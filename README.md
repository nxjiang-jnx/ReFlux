# ReFlux Supplementary Materials

## Overview

ReFlux is a attack method for fine-tuning rectified flow based T2I diffusion models with focus on content safety and robustness evaluation. The project implements LoRA-based adaptation on Flux architecture with comprehensive evaluation methodologies for assessing model behavior under adversarial conditions.

## Project Structure

### Core Components

- **`train_flux_lora.py`**: Main training script implementing LoRA fine-tuning with multi-objective loss functions

- **`asr_evaluation/`**: Attack Success Rate evaluation framework
  - Nudity detection using ONNX-based inference
  - Batch processing and statistical analysis
  - Comprehensive metrics computation

- **`q16_evaluation/`**: Violence detection system
  - CLIP-based classifier for harmful content
  - Pre-trained embeddings for robust classification

- **`utils/`**: Supporting utilities
  - Dataset management with synonym replacement and prompt shuffling
  - FLUX-specific operations
  - Loss calculation and optimization strategies

- **`tools/`**: Processing tools
  - T5-encoder prompt processing
  - Custom scheduler implementation

- **`config/`**: Configuration files for training and evaluation parameters

- **`datasets/`**: Multi-category datasets including content safety, artistic styles, and abstract concepts

## Usage

### Installation & Environment Setup

1. **Install Python dependencies:**

   ```
   pip install transformers sentencepiece einops omegaconf
   pip install tokenizers==0.20.0
   pip install nltk wandb openai
   ```

2. **Install local packages:**

   ```
   cd peft
   pip install -e .[torch]
   
   or
   
   python setup.py install
   
   cd ../diffusers
   pip install -e .[torch]
   
   or
   
   python setup.py install
   ```

### Training
```bash
python train_flux_lora.py --config config/config.yaml
```

### Evaluation
```python
from asr_evaluation import ASRCalculator
calculator = ASRCalculator(threshold=0.6)
results = calculator.calculate_batch_asr("experiments/", "baseline/")
```

## Dependencies

- PyTorch >= 2.0, Diffusers >= 0.32.0, Transformers
- PEFT, ONNX Runtime, OpenCV, CLIP, NLTK
- Optional: Weights & Biases, BitsAndBytes

## Research Applications

- Content safety evaluation of diffusion models
- Adversarial robustness assessment
- Parameter-efficient fine-tuning studies
- Multi-modal content classification

## Ethical Considerations

This research tool is intended for academic and research purposes to improve the robustness and safety of AI systems. Users should ensure compliance with applicable laws and ethical guidelines when conducting research with this framework.
