# coding: UTF-8
"""
    @func: loss
"""

import random
import torch
from typing import Dict, Optional
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from .esd_utils import latent_sample, predict_noise

def calculate_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, transformer, noise_scheduler, prompts, vae, criteria, negative_guidance, weight_dtype, neg_prompts, start_guidance=3, ddim_steps=28, lamb1=1, lamb2=1):
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents:
        # Use cached latents if available in batch
        model_input = batch.get("latents", None)
        if model_input is None:
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
            model_input = vae.encode(pixel_values).latent_dist.sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()

    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
                neg_prompts, text_encoders, tokenizers
            )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
                prompts, text_encoders, tokenizers
            )

    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

    start_guidance = 3
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    with torch.no_grad():
        # generate an image with the concept from ESD model
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor)
        # e_0 & e_p
        e_0 = predict_noise(transformer, z, emb_0, pooled_emb_0, text_ids_0, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
        e_p = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)

    e_n = predict_noise(transformer, z, emb_p, pooled_emb_p, text_ids_p, latent_image_ids, guidance=start_guidance, timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True)
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    total_loss = []
    
    loss_attack = -torch.mean(e_n) + negative_guidance * torch.mean(torch.norm(e_p.to(transformer.device) - e_0.to(transformer.device), p=2, dim=list(range(1, len(e_p.shape)))) ** 2)

    total_loss.append(loss_attack)
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            transformer.device,
            weight_dtype,
        )
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    noisy_model_input = noise_scheduler.add_noise(model_input,
                                                  noise,
                                                  t_enc_ddpm)

    packed_noisy_model_input = FluxPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

    if transformer.config.guidance_embeds:
        guidance = torch.tensor([args.guidance_scale], device=transformer.device)
        guidance = guidance.expand(model_input.shape[0])
    else:
        guidance = None

    remove_indices = batch['remove_indices'][0]

    model_pred, attn_maps = transformer(
        hidden_states=packed_noisy_model_input.to(dtype=weight_dtype, device=transformer.device),
        timestep=t_enc_ddpm / 1000,
        guidance=guidance.to(dtype=weight_dtype, device=transformer.device),
        pooled_projections=pooled_emb_p.to(dtype=weight_dtype, device=transformer.device),
        encoder_hidden_states=emb_p.to(dtype=weight_dtype, device=transformer.device),
        txt_ids=text_ids_p.to(dtype=weight_dtype, device=transformer.device),
        img_ids=latent_image_ids.to(dtype=weight_dtype, device=transformer.device),
        return_dict=False,
    )[0:2]

    attn_map_mask = torch.ones_like(attn_maps).to(transformer.device)
    attn_map_mask[..., remove_indices] = 0
    attn_map_mask = 1 - attn_map_mask
    # import pdb; pdb.set_trace()

    # Compute attention loss with L2 and entropy regularization
    lambda_l2 = getattr(args, 'lambda_attn_l2', 1e-4) 
    lambda_entropy = getattr(args, 'lambda_attn_entropy', 0.05) 
    
    # Original attention loss (negative sum for target indices)
    loss_attn_base = -sum(torch.norm(attn_map_mask*attn_maps, dim=(0, 1))).sum()
    
    # L2 regularization term for attention maps
    loss_attn_l2 = lambda_l2 * torch.norm(attn_maps, p=2) ** 2
    
    # Entropy regularization term
    eps = 1e-8  # Small constant for numerical stability
    
    # Apply softmax along the key dimension (last dimension)
    attn_probs = torch.softmax(attn_maps, dim=-1)  # [H, Q, K]
    
    # Compute entropy for each head and query
    log_probs = torch.log(attn_probs + eps)
    entropy_per_head_query = -torch.sum(attn_probs * log_probs, dim=-1)
    
    total_entropy = torch.sum(entropy_per_head_query)
    loss_entropy = -lambda_entropy * total_entropy
    combined_attn_loss = loss_attn_base + loss_attn_l2 + loss_entropy
        
    loss_attn = combined_attn_loss
    total_loss.append(loss_attn)
    
    # LoRA Loss
    model_pred_unpacked = FluxPipeline._unpack_latents(
        model_pred,
        height=model_input.shape[2] * vae_scale_factor,
        width=model_input.shape[3] * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )
    
    loss_lora = torch.mean(
        ((noise.float() - model_pred_unpacked.float()) ** 2).reshape(noise.shape[0], -1),
        1,
    )[0]
    
    total_loss.append(loss_lora)
            
    return total_loss, t_enc_ddpm