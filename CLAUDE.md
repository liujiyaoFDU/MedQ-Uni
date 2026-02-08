# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Uni-MedVL** (also known as MedQ-Uni) is a unified medical vision-language model that combines both:
- **Visual Understanding**: Medical image analysis, VQA, diagnostic reasoning, report generation
- **Visual Generation**: Medical image editing, cross-modal translation (e.g., H&E to IHC staining)

The model uses a Bagel architecture combining Qwen2 language model, SigLIP vision transformer, and VAE for image generation.

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yaml
conda activate medq_uni_junzhin  # or unimedvl_gzy

# Install Flash Attention (must be compatible with your PyTorch/CUDA)
pip install --no-build-isolation flash-attn==2.8.3

# Or use pip directly with requirements.txt
pip install -r requirements.txt
```

## Training Commands

### Basic Training

Training uses PyTorch distributed with torchrun:

```bash
# Single-node training (8 GPUs)
bash scripts/training/train_sft_generic.sh

# Multi-node training
bash scripts/training/train_sft_stage1_medq_unif_multinode.sh
```

### Key Training Parameters

Training scripts are in `scripts/training/` and use `train/main.py` or `train/main_sr_pixel_loss.py`:

```bash
torchrun --nproc_per_node=8 \
  train/main.py \
  --dataset_config_file configs/finetuned_example.yaml \
  --model_path /path/to/checkpoint \
  --checkpoint_dir output/experiment_name \
  --total_steps 100 \
  --lr 1e-5 \
  --ce_weight 0.25 \  # Cross-entropy loss weight
  --mse_weight 1.0 \   # MSE loss weight for generation
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --visual_gen True \  # Enable image generation
  --visual_und True    # Enable image understanding
```

### Important Training Script Paths to Configure

Before running training scripts, update these paths in `scripts/training/train_sft_*.sh`:
- Line 2: `path2user_bashrc` - Your bashrc file path
- Line 5: `path2project_root` - This project root directory
- Line 7: Conda environment name (if different)
- Line 10: `path2model_checkpoint` - Model checkpoint path

## Inference Commands

### Interactive VQA Inference

For medical image analysis and diagnostic reasoning:

```bash
# Direct execution (Jupyter-style cells with #%%)
python interactive_vqa_inferencer.py

# Key configuration in the script (lines 42, 60):
# - ROOT: UniMedVL installation directory
# - model_path: Model checkpoint path
```

### Interactive Image Generation

For medical image editing and translation:

```bash
# Direct execution (Jupyter-style cells with #%%)
python interactive_image_generator.py

# Two inference modes:
# - Understanding mode (default): Faster, context-aware editing
# - Thinking mode: Slower, includes internal reasoning process
```

### Batch Inference for Evaluation

```bash
# Single checkpoint, sequential processing
bash inference_pipeline/MedQ-Uni_run_batch_test_ver1.sh

# Multiple checkpoints, parallel GPU processing
bash inference_pipeline/MedQ-Uni_run_batch_test_ver2.sh

# Aggregate results to CSV
python inference_pipeline/parse_statistics_to_csv.py
```

## Dataset System Architecture

The dataset system uses a three-tier registry pattern:

### 1. Dataset Registration (`data/dataset_info.py`)

```python
# Register dataset classes
DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'CounterfactualMedicalIterableDataset_ver1': CounterfactualMedicalIterableDataset_ver1,
    # Add new dataset types here
}

# Register dataset paths and metadata
DATASET_INFO = {
    'CounterfactualMedicalIterableDataset_ver1': {
        'counterfactual_cxr_chexpertplus_train': {
            'data_dir': '/path/to/images',
            'jsonl_path': '/path/to/annotations.jsonl',
        },
    }
}
```

### 2. Dataset Configuration (`configs/*.yaml`)

YAML files define dataset combinations, weights, and transform parameters:

```yaml
CounterfactualMedicalIterableDataset_ver1:
  dataset_names:
    counterfactual_train:
      - counterfactual_cxr_chexpertplus_train
      - counterfactual_cxr_mimic_cxr_train

  # VAE image transform parameters
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512

  # ViT image transform parameters
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2_007_040

  weight:
    counterfactual_train: 10  # Higher weight = larger proportion

  num_used_data:
    counterfactual_train: [0, 0]  # [start, end]; [0,0] means use all

  is_mandatory: true
  shuffle_lines: true
  shuffle_seed: 42
```

### 3. Training Script Reference

Training scripts pass the config file path:

```bash
torchrun train/main.py --dataset_config_file configs/finetuned_example.yaml ...
```

### Data Flow

```
Training Script
    ↓ reads
YAML Config (configs/*.yaml)
    ↓ references dataset class name
DATASET_REGISTRY (data/dataset_info.py)
    ↓ looks up dataset paths
DATASET_INFO (data/dataset_info.py)
    ↓ instantiates
Dataset Class (data/vlm_dataset.py, data/interleave_datasets/)
```

### Adding a New Dataset

1. **Define dataset class** (if needed) in `data/vlm_dataset.py` or `data/interleave_datasets/`
2. **Register class** in `DATASET_REGISTRY` in `data/dataset_info.py`
3. **Add dataset paths** in `DATASET_INFO` in `data/dataset_info.py`
4. **Create YAML config** in `configs/` or modify existing one
5. **Reference config** in training script with `--dataset_config_file`

## JSONL Data Format

The model expects JSONL files with one JSON object per line:

```json
{
  "main_task_type": "multimodal_input_output",
  "input_img": [
    {
      "path": "path/to/input/image.png",
      "height": 512,
      "width": 512
    }
  ],
  "output_img": [
    {
      "path": "path/to/output/image.png",
      "height": 512,
      "width": 512
    }
  ],
  "message": [
    {
      "from": "human",
      "value": "<image>\n\nAnalyze this chest X-ray."
    },
    {
      "from": "gpt",
      "value": "The chest X-ray shows... <image>"
    }
  ]
}
```

**Key fields:**
- `main_task_type`: Task identifier (e.g., "multimodal_input_output", "image_understanding", "text_only")
- `input_img`: Array of input images (optional, for VQA/understanding)
- `output_img`: Array of output images (optional, for generation)
- `message`: Conversation array with "human" and "gpt" turns
- `<image>` token: Placeholder for image in text

## Code Architecture

### Model Structure (`modeling/`)

```
modeling/
├── bagel/                    # Main model implementation
│   ├── bagel.py             # Bagel model (main architecture)
│   ├── qwen2_navit.py       # Qwen2 language model with NaViT
│   ├── siglip_navit.py      # SigLIP vision encoder
│   ├── losses.py            # Loss functions (CE, MSE, pixel loss)
│   └── modeling_utils.py    # MLP connector, timestep embedder
├── autoencoder.py           # VAE for image generation
├── qwen2/                   # Qwen2 tokenizer and config
└── siglip/                  # SigLIP processor
```

**Bagel Model Components:**
- **Language Model**: Qwen2 (LLM backbone for text generation and reasoning)
- **Vision Encoder**: SigLIP NaViT (for image understanding)
- **VAE**: Autoencoder (for image generation, with latent diffusion)
- **Connectors**: MLP layers bridging vision/VAE to LLM hidden states

### Training (`train/`)

```
train/
├── main.py                       # Main training script
├── main_sr_pixel_loss.py        # Training with pixel-level losses
├── main_benchmark.py            # Benchmark evaluation
├── fsdp_utils.py                # FSDP (Fully Sharded Data Parallel) utilities
├── param_manager.py             # Parameter group management
└── train_utils.py               # Training utilities
```

**Training uses:**
- PyTorch FSDP for distributed training
- EMA (Exponential Moving Average) for model weights
- Mixed precision training
- TensorBoard logging
- Checkpoint saving and resuming

### Dataset Loading (`data/`)

```
data/
├── dataset_info.py              # Dataset registry and metadata
├── dataset_base.py              # Base dataset class
├── vlm_dataset.py               # VLM SFT datasets
├── t2i_dataset.py               # Text-to-image datasets
├── interleave_datasets/         # Multimodal interleaved datasets
├── transforms.py                # Image transformations
└── data_utils.py                # Data processing utilities
```

**Key concepts:**
- **IterableDataset**: All datasets inherit from PyTorch IterableDataset for streaming
- **Sharding**: Data is sharded across ranks and workers for distributed training
- **Transforms**: Separate transforms for VAE (generation) and ViT (understanding)

### Inference (`inference_pipeline/`, root level)

```
inference_pipeline/
├── MedQ-Uni_run_batch_test1.py    # Batch inference (VAE min=512)
├── MedQ-Uni_run_batch_test2.py    # Batch inference (VAE min=256)
├── MedQ-Uni_run_batch_test_ver*.sh # Shell scripts for batch inference
└── parse_statistics_to_csv.py      # Aggregate evaluation results

Root level:
├── interactive_vqa_inferencer.py   # Interactive VQA inference
├── interactive_image_generator.py  # Interactive image generation
└── inferencer.py                   # Base inferencer class
```

## Key Design Patterns

### 1. Dual-Mode Architecture

The model supports two modes simultaneously:
- `visual_und=True`: Vision understanding (VQA, image analysis) via ViT
- `visual_gen=True`: Image generation (editing, translation) via VAE

Both can be enabled together during training for unified multimodal capabilities.

### 2. Position Encoding Strategies

- **Interpolate**: For ViT patches when image size varies (standard scaling)
- **Extrapolate**: For VAE latents (novel position encoding beyond training size)

Set in config: `interpolate_pos=False` (default uses extrapolation for VAE)

### 3. Conditional Dropout

Training supports dropout for different modalities:
- `vit_cond_dropout_prob`: Dropout ViT features (image understanding)
- `vae_cond_dropout_prob`: Dropout VAE conditioning (image generation)
- `text_cond_dropout_prob`: Dropout text conditioning

This enables unconditional/classifier-free guidance during inference.

### 4. Timestep Shifting

For diffusion-based image generation:
- `timestep_shift`: Shifts the noise schedule (default 3.0 for inference)
- Higher values → cleaner initial generations, less denoising steps needed

### 5. Loss Weighting

Training combines multiple losses:
- `ce_weight`: Cross-entropy loss (language modeling)
- `mse_weight`: MSE loss (VAE latent prediction)
- Pixel-level losses: Direct image space supervision (in `main_sr_pixel_loss.py`)

## Common Workflows

### Train a New Model

1. Prepare JSONL dataset following the format above
2. Add dataset to `data/dataset_info.py` (DATASET_REGISTRY and DATASET_INFO)
3. Create or modify YAML config in `configs/`
4. Update paths in training script: `scripts/training/train_sft_*.sh`
5. Run training: `bash scripts/training/train_sft_generic.sh`

### Evaluate a Checkpoint

```bash
# Interactive VQA
python interactive_vqa_inferencer.py
# (Modify model_path and ROOT in the script)

# Batch evaluation
bash inference_pipeline/MedQ-Uni_run_batch_test_ver2.sh
# (Modify CHECKPOINTS array in the script)

# Aggregate results
python inference_pipeline/parse_statistics_to_csv.py
```

### Resume Training

Training scripts support auto-resume:
```bash
--auto_resume True \
--resume_from /path/to/checkpoint \
--resume_model_only False \      # Resume both model and optimizer
--resume_model_optimizer True
```

### Freeze/Unfreeze Components

Control which parts of the model are trainable:
```bash
--freeze_llm False \   # Train language model
--freeze_vit True \    # Freeze vision encoder
--freeze_vae True      # Freeze VAE
```

## Important Notes

### Module Imports in Inference Scripts

The inference scripts (`interactive_*.py`, `inference_pipeline/*.py`) use absolute path imports:
```python
ROOT = "/path/to/UniMedVL/"  # Shared module library
sys.path.append(ROOT)

from data.transforms import ImageTransform
from modeling.bagel import Bagel
from inferencer import InterleaveInferencer
```

**When modifying inference scripts**, update the `ROOT` path to match your installation.

### Shell Script Execution

Training and inference shell scripts always `cd` to project root first:
```bash
cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni
```

This ensures relative paths work correctly regardless of where the script is invoked from.

### FSDP Sharding Strategies

Training uses FSDP with configurable strategies:
- `FULL_SHARD`: Maximum memory efficiency, slower
- `HYBRID_SHARD`: Balance of memory and speed (default)
- `SHARD_GRAD_OP`: Only shard gradients and optimizer states

Set via: `--sharding_strategy HYBRID_SHARD`

### VAE Transform Size

When running inference, the VAE transform `min_size` affects quality vs. speed tradeoff:
- `min_size=512`: Better quality, more VRAM, slower (test1.py)
- `min_size=256`: Faster, less VRAM, potential quality loss (test2.py)

Choose based on your hardware and requirements.
