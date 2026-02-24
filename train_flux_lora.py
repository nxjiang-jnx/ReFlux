import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import yaml
import copy
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import transformers
import logging

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)

from utils.lora_dataset import LoraDataset, collate_data_fn
from utils.calc_loss import calculate_loss
from tools.prompt_process import encode_prompt
from tools.scheduler_process import CustomFlowMatchEulerDiscreteScheduler

print("All modules imported")

if is_wandb_available():
    import wandb

check_min_version("0.32.0.dev0")
logger = logging.getLogger(__name__)

def load_text_encoders(class_one, class_two, args):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def finetune(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    model_id = Path(args.output_dir).name
    
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        cache_dir=cache_dir
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        cache_dir=cache_dir
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    
    noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
        cache_dir=cache_dir
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two, args)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        cache_dir=cache_dir
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant, cache_dir=cache_dir
    ).to(devices[0])
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(transformer.device, dtype=weight_dtype)
    transformer.to(transformer.device, dtype=weight_dtype)
    text_encoder_one.to(transformer.device, dtype=weight_dtype)
    text_encoder_two.to(transformer.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            # "attn.to_k",
            # "attn.to_q",
            # "attn.to_v",
            # "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            # "attn.add_v_proj",
            # "attn.to_add_out",
            # "ff.net.0.proj",
            # "ff.net.2",
            # "ff_context.net.0.proj",
            # "ff_context.net.2",
        ]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # if args.train_text_encoder:
            # models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # If neither --train_text_encoder nor --train_text_encoder_ti, text_encoders remain frozen during training
    freeze_text_encoder = True # not (args.train_text_encoder or args.train_text_encoder_ti)
    print("[20250805] free text encoder", freeze_text_encoder)
    

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": float(args.learning_rate)}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=float(args.adam_weight_decay),
            eps=float(args.adam_epsilon),
        )

    criteria = torch.nn.MSELoss()

    # Dataset and DataLoaders creation:
    train_dataset = LoraDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        key_word=args.key_word,
        tokenizer_t5=tokenizer_two,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_data_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    if freeze_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(transformer.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(transformer.device)
                text_ids = text_ids.to(transformer.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # if --train_text_encoder_ti we need add_special_tokens to be True for textual inversion
    add_special_tokens_clip = False # True if args.train_text_encoder_ti else False
    add_special_tokens_t5 = False # True if (args.train_text_encoder_ti and args.enable_t5_ti) else False

    if not train_dataset.custom_instance_prompts:
        if freeze_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    transformer.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # print("Num examples = ", len(train_dataset))
    # print("Num batches each epoch = ", len(train_dataloader))
    # print("Num Epochs = ", args.num_train_epochs)
    # print("Instantaneous batch size per device = ", args.train_batch_size)
    # print("Total train batch size (w. parallel, distributed & accumulation) = ", total_batch_size)
    # print("Gradient Accumulation steps = ", args.gradient_accumulation_steps)
    # print("Total optimization steps = ", args.max_train_steps)
    
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            upper_loss, t_enc_ddpm = calculate_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler_copy, batch["prompts"], vae, criteria, negative_guidance=args.negative_guidance, weight_dtype=weight_dtype, neg_prompts=str(args.prompt_b), start_guidance=3, ddim_steps=28, lamb1=float(args.lamb1), lamb2=float(args.lamb2))
            
            loss = float(args.lamb1) * upper_loss[0] + float(args.lamb2) * upper_loss[1] + float(args.lamb3) * upper_loss[2]
            
            esd_loss = float(args.lamb1) * upper_loss[0].detach().item()
            attn_loss = float(args.lamb2) * upper_loss[1].detach().item()
            lora_loss = float(args.lamb3) * upper_loss[2].detach().item()
            
            logs = {
                "esd": esd_loss, 
                "attn": attn_loss,
                "lora": lora_loss,
                "prompt": batch["prompts"], 
                "index": batch['remove_indices'][0], 
                "lr": lr_scheduler.get_last_lr()[0]
            }
            lambda_entropy = getattr(args, 'lambda_attn_entropy', 0.0)
            if lambda_entropy > 0:
                logs["λ_H"] = lambda_entropy
                
            progress_bar.set_postfix(**logs)
            
            loss.backward()
            
            attention_params = []
            other_params = []
            
            for name, param in transformer.named_parameters():
                if param.requires_grad:
                    if 'attn' in name:  # attention layer parameters
                        attention_params.append(param)
                    else:
                        other_params.append(param)
            
            # Apply stronger gradient clipping to attention layers
            attn_grad_norm = getattr(args, 'attn_max_grad_norm', 0.5)
            other_grad_norm = getattr(args, 'max_grad_norm', 1.0)
            
            if attention_params:
                torch.nn.utils.clip_grad_norm_(attention_params, attn_grad_norm)
            if other_params:
                torch.nn.utils.clip_grad_norm_(other_params, other_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            progress_bar.update(1)
            global_step += 1
            
            
            if global_step % args.checkpointing_steps == 0:

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                # Save the lora layers
                transformer = transformer.to(weight_dtype)
                transformer_lora_layers = get_peft_model_state_dict(transformer)
                text_encoder_lora_layers = None

                FluxPipeline.save_lora_weights(
                    save_directory=args.output_dir,
                    transformer_lora_layers=transformer_lora_layers,
                    text_encoder_lora_layers=text_encoder_lora_layers,
                )
                logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break
            
    # Save the lora layers
    transformer = transformer.to(weight_dtype)
    transformer_lora_layers = get_peft_model_state_dict(transformer)
    text_encoder_lora_layers = None

    FluxPipeline.save_lora_weights(
        save_directory=args.output_dir,
        transformer_lora_layers=transformer_lora_layers,
        text_encoder_lora_layers=text_encoder_lora_layers,
    )
    print("All done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with given parameters.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    
    def read_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    config = read_config(args.config)
    for key, value in config.items():
        setattr(args, key, value)
    finetune(args)