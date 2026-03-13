# UniDD

Official PyTorch implementation of:

**Understanding Dataset Distillation via Spectral Filtering**  
Deyu Bo, Songhua Liu, Xinchao Wang  
ICLR 2026

UniDD is a research codebase for understanding and improving dataset distillation through spectral filtering. The repository includes teacher pretraining, feature-statistics extraction, synthetic data generation, and distilled-data training with online soft labels across CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet settings.

## News

- `ICLR 2026`: **Understanding Dataset Distillation via Spectral Filtering** was accepted to ICLR 2026.

## Overview

This repository currently provides:

- teacher pretraining scripts for small and large-scale settings,
- feature-statistics hooks for real-data distribution modeling,
- synthetic image generation pipelines based on spectral filtering,
- training and evaluation on synthesized datasets with online soft labels.

Supported dataset families:

- `CIFAR-10`
- `CIFAR-100`
- `Tiny-ImageNet`
- `ImageNet`

## Repository Layout

```text
UniDD/
+-- pretrain_small.py             # teacher training for CIFAR-10 / CIFAR-100
+-- pretrain_large.py             # teacher training for Tiny / ImageNet
+-- synthesis_small.py            # distilled data synthesis for CIFAR-10 / CIFAR-100
+-- synthesis_large.py            # distilled data synthesis for Tiny / ImageNet
+-- generate_soft_label_online.py # train/evaluate on synthesized data
+-- hook.py                       # feature statistic hooks and spectral filtering logic
+-- utils.py                      # datasets, image saving, synthetic data loader, helpers
+-- rded_models.py                # ConvNet variants used in this project
+-- tiny_imagenet_dataset.py      # Tiny-ImageNet dataset wrapper
+-- ckpt/                         # teacher checkpoints
+-- statistic/                    # cached running feature statistics
+-- hf_dataset/                   # local Hugging Face dataset shard
+-- save/                         # training outputs
`-- archive/                      # older experiments and validation code
```

## Installation

Recommended environment:

- Python >= 3.10
- PyTorch >= 2.0
- torchvision
- numpy
- scipy
- tqdm
- Pillow
- kornia
- transformers
- datasets

Install the main dependencies with:

```bash
pip install torch torchvision numpy scipy tqdm pillow kornia transformers datasets
```

## Data Preparation

Expected data layout:

```text
data/
+-- CIFAR-10 / CIFAR-100         # auto-downloaded by torchvision
+-- tiny-imagenet-200/           # auto-downloaded by tiny_imagenet_dataset.py
`-- ImageNet/
    +-- train/
    `-- val/
```

Notes:

- CIFAR datasets are downloaded automatically by `torchvision`.
- Tiny-ImageNet can be downloaded automatically by `tiny_imagenet_dataset.py`.
- ImageNet should be prepared manually under `data/ImageNet/train` and `data/ImageNet/val`.
- `datasets` is required by `synthesis_large.py` when using `hf_dataset/rded-ipc-10`.

## Supported Settings

| Script | Datasets | Models |
| --- | --- | --- |
| `pretrain_small.py` | `CIFAR-10`, `CIFAR-100` | `ResNet18`, `ConvNetW128` |
| `pretrain_large.py` | `Tiny`, `ImageNet` | `Tiny`: `ResNet18`, `ConvNetW128D4`; `ImageNet`: `ResNet18` |
| `synthesis_small.py` | `CIFAR-10`, `CIFAR-100` | `ResNet18`, `ConvNetW128D3` |
| `synthesis_large.py` | `Tiny`, `ImageNet` | `Tiny`: `ResNet18`, `ConvNetW128D4`; `ImageNet`: `ResNet18` |
| `generate_soft_label_online.py` | `CIFAR-10`, `CIFAR-100`, `Tiny`, `ImageNet` | dataset-dependent |

## Quick Start

The standard workflow is:

1. pretrain or prepare a teacher model,
2. synthesize the distilled dataset,
3. train or evaluate on the synthesized data.

### 1. Pretrain a Teacher

For CIFAR-10 / CIFAR-100:

```bash
python pretrain_small.py \
  --dataset CIFAR-10 \
  --model ResNet18 \
  --data_path ./data \
  --buffer_path ./ckpt
```

For Tiny-ImageNet / ImageNet:

```bash
python pretrain_large.py \
  --dataset Tiny \
  --model ResNet18 \
  --data_path ./data \
  --buffer_path ./ckpt
```

Teacher checkpoints are saved to `ckpt/<dataset>/`.

### 2. Synthesize Distilled Data

For CIFAR-style datasets:

```bash
python synthesis_small.py \
  --dataset CIFAR-10 \
  --model ResNet18 \
  --ipc 50 \
  --data_path ./data \
  --model_path ./ckpt \
  --statistic_path ./statistic \
  --syn_path ./syn_data \
  --filter HFM \
  --signal mean \
  --beta 0.1 \
  --cos
```

For Tiny-ImageNet / ImageNet:

```bash
python synthesis_large.py \
  --dataset Tiny \
  --model ResNet18 \
  --ipc 50 \
  --data_path ./data \
  --model_path ./ckpt \
  --statistic_path ./statistic \
  --syn_path ./syn_data \
  --filter HFM \
  --signal mean \
  --scheduler cos \
  --beta 0.1
```

The synthesized images are written to:

```text
syn_data/<dataset>/conv_<model>_<ipc>_<filter>_<scheduler_or_flag>_<beta>/
```

Each class is saved in a separate folder so the output can be loaded directly with `ImageFolder`.

### 3. Train / Evaluate on Synthesized Data

Example:

```bash
python generate_soft_label_online.py \
  --dataset Tiny \
  --model ResNet18 \
  --val_model ResNet18 \
  --ipc 50 \
  --teacher_path ./ckpt \
  --syn_data_path ./syn_data \
  --syn_folder conv_ResNet18_50_HFM_cos_0.1 \
  --output_dir ./save
```

This stage loads the synthesized dataset, applies augmentation, generates soft targets online with a teacher model, and saves outputs to `save/<dataset>/`.

## Implementation Notes

- `synthesis_small.py` uses the boolean flag `--cos` for beta scheduling.
- `synthesis_large.py` uses `--scheduler` with options such as `none`, `cos`, and `linear`.
- Feature statistics are cached automatically under `statistic/` if they do not already exist.
- `generate_soft_label_online.py` expects the synthesized folder name to match the output naming convention of the synthesis scripts.
- For `ImageNet` with `ipc == 1`, `synthesis_large.py` uses the local shard under `hf_dataset/rded-ipc-10`.

## Artifacts

The current repository already includes several generated artifacts:

- `ckpt/` for teacher checkpoints,
- `statistic/` for cached feature statistics,
- `save/` for training outputs,
- `hf_dataset/` for a local dataset shard.

If you want a lighter public release later, a good next step is to move large artifacts to Hugging Face Hub, GitHub Releases, or external storage and link them here.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{bo2026understanding,
  title     = {Understanding Dataset Distillation via Spectral Filtering},
  author    = {Deyu Bo and Songhua Liu and Xinchao Wang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## Acknowledgements

- Parts of `baseline.py` are adapted from [DatasetCondensation](https://github.com/VICO-UoE/DatasetCondensation).
- `tiny_imagenet_dataset.py` is adapted from an existing Tiny-ImageNet dataset wrapper implementation.

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
