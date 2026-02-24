# Erased, But Not Forgotten: Erased Rectified Flow Transformers Still Remain Unsafe Under Concept Attack

<p align="center">
  <a href="https://arxiv.org/abs/2510.00635">
    <img src='https://img.shields.io/badge/Paper-arXiv%20Preprint-green?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
  </a>
    <a href="https://github.com/nxjiang-jnx/ReFlux">
  <img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github">
</a>
    <a href="LICENSE">
  <img src="https://img.shields.io/badge/License-CC--BY-green?style=for-the-badge">
</a>
</p>


![Teaser](teaser.png)

## Overview

ReFlux is a attack method for fine-tuning rectified flow based T2I diffusion models with focus on content safety and robustness evaluation. The project implements LoRA-based adaptation on Flux architecture with comprehensive evaluation methodologies for assessing model behavior under adversarial conditions.

## Features

- ✅ Supports **[diffusers]** (You need to use my version of diffusers)
- ✅ Easy to extend and integrate

## Installation & Environment Setup

1. **Install Rust (if required):**

   ```bash
   curl https://sh.rustup.rs -sSf | sh
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

2. **Install Python dependencies:**

   ```bash
   pip install transformers sentencepiece einops omegaconf
   pip install tokenizers==0.20.0
   pip install nltk wandb openai
   ```

3. **Install local packages:**

   ```bash
   cd peft
   pip install -e .[torch]
   
   or
   
   python setup.py install
   
   cd ../diffusers
   pip install -e .[torch]
   
   or
   
   python setup.py install
   ```

## Quick Start

1. **Train the adversarial checkpoint (LoRA):**

```bash
python train_flux_lora.py --config config/config.yaml
```

2. **Run attack inference:**

For quick visualization, please run `attack_inference.py` .

For batch experiments, please run:

```python
from asr_evaluation import ASRCalculator
calculator = ASRCalculator(threshold=0.6)
results = calculator.calculate_batch_asr("experiments/", "baseline/")
```

## Acknowledgments

This project is inspired by and builds upon the work of [Erase Anything](https://github.com/tomguluson92/EraseAnything) and [Erasing Concepts from Diffusion Models](https://github.com/rohitgandikota/erasing) and other open-source projects. We thank the community for their valuable contributions.

## Ethical Considerations

This research tool is intended for academic and research purposes to improve the robustness and safety of AI systems. Users should ensure compliance with applicable laws and ethical guidelines when conducting research with this framework.

## Citation

If you use this project in your research, please cite:

```bibtex
@article{jiang2025ReFlux,
  title={Erased, But Not Forgotten: Erased Rectified Flow Transformers Still Remain Unsafe Under Concept Attack},
  author={Nanxiang Jiang and Zhaoxin Fan and Enhan Kang and Daiheng Gao and Yun Zhou and Yanxia Chang and Zheng Zhu and Yeying Jin and Wenjun Wu},
  journal={CVPR findings},
  year={2026}
}
```

## Contact

For technical questions, please contact **jiangnx@buaa.edu.cn**.
