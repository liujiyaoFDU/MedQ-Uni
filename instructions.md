# UniMedVL Environment Setup and Training Instructions

## 🔧 Environment Setup

### 1. Create Conda Environment
```bash
conda env create -f environment.yaml -y
conda activate unimedvl_gzy
```

### 2. Install Flash Attention
```bash
pip install --no-build-isolation flash-attn==2.8.3
```

**Note**: flash-attn version must be compatible with your PyTorch and cuda versions. If 2.8.3 has issues, please try others!


## 🚀 Training Script Configuration

### train_sft_generic.sh -  Configuration Guide

The script requires you to replace the following paths with your actual paths:

1. **Line 2**: `path2user_bashrc` - Your bashrc file path (e.g., `/home/username/.bashrc`)
2. **Line 5**: `path2project_root` - Project root directory path (e.g., `/home/username/projects/Uni-MedVL`)
3. **Line 7**: `unimedvl_gzy` - Your conda environment name (modify if different)
4. **Line 10**: `path2model_checkpoint` - Model checkpoint path (e.g., `/data/models/unimedvl_checkpoint`)

### Path Configuration Methods

```bash
# Method 1: Directly edit path variables in the script
# Method 2: Use sed commands for batch replacement
sed -i 's|path2user_bashrc|/home/yourname/.bashrc|g' scripts/train_sft_generic.sh
sed -i 's|path2project_root|/path/to/your/Uni-MedVL|g' scripts/train_sft_generic.sh
sed -i 's|path2model_checkpoint|/path/to/your/model/checkpoint|g' scripts/train_sft_generic.sh
```

### Running Commands
```bash
# Basic run
cd path2project_root  # Replace with your actual project path
bash scripts/train_sft_generic.sh
```

---

## 📊 Dataset Registration and Configuration

Uni-MedVL uses a flexible data configuration system that allows you to add and manage training datasets by modifying configuration files, without changing core code. Here is the complete dataset registration process:

### 📋 Data Configuration Flow Overview

```
1. dataset_info.py (Register dataset classes and config info)
       ↓
2. finetuned_example.yaml (Configure dataset combinations and parameters)
       ↓
3. train_sft_generic.sh (Reference config file to start training)
       ↓
4. Training script (Load config and build datasets)
```

---

## 🗂️ Step 1: Register Datasets in `data/dataset_info.py`

### 1.1 Register Dataset Classes

Add new dataset class mappings in the `DATASET_REGISTRY` dictionary. If you do not intend to add new dataset classes for a new task, you can skip this step:

```python
DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    # Add new dataset types
    "MyCustomMedicalDataset": MyCustomMedicalDataset,
    "CounterfactualMedicalIterableDataset_ver1": CounterfactualMedicalIterableDataset_ver1
}
```

### 1.2 Configure Dataset Information

Add specific configuration information for datasets in the `DATASET_INFO` dictionary:

```python
DATASET_INFO = {
    # Existing datasets...

    # New dataset configuration
    "CounterfactualMedicalIterableDataset_ver1": {
        'counterfactual_cxr_chexpertplus_train': {
            'data_dir': '/path/to/your/images',
            'jsonl_path': '/path/to/your/annotations.jsonl',
        },
        'counterfactual_cxr_mimic_cxr_train': {
            'data_dir': '/path/to/mimic/images',
            'jsonl_path': '/path/to/mimic/annotations.jsonl',
        }
    },

    "MyCustomMedicalDataset": {
        'my_medical_data_train': {
            'data_dir': '/path/to/medical/images',
            'jsonl_path': '/path/to/medical/annotations.jsonl',
        }
    }
}
```

**Key Configuration Fields:**
- `data_dir`: Root directory path for image files
- `jsonl_path`: Path to JSONL format annotation file

---

## ⚙️ Step 2: Configure Dataset Combinations in `configs/finetuned_example.yaml`

The YAML configuration file defines dataset combinations and their parameters used during training. Below is a complete configuration example:

```yaml
# Dataset class name - must match the key name in DATASET_REGISTRY
CounterfactualMedicalIterableDataset_ver1:
  # Dataset group names
  dataset_names:
    counterfactual_train:
      - counterfactual_cxr_chexpertplus_train  # Specific dataset name, corresponds to key in DATASET_INFO
      - counterfactual_cxr_mimic_cxr_train

  # VAE image transformation parameters
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512

  # ViT image transformation parameters
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2_007_040

  # Dataset weights - for multi-dataset mixed training
  weight:
    counterfactual_train: 10  # Higher weight means larger proportion in training

  # Data usage control [start_sample, end_sample], [0, 0] means use all data
  num_used_data:
    counterfactual_train: [0, 0]

  # Whether this is a mandatory dataset
  is_mandatory: true

  # Whether to shuffle data order
  shuffle_lines: true

  # Shuffle seed
  shuffle_seed: 42
```

**YAML Configuration Parameters:**

| Parameter | Type | Description | Example Value |
|-----------|------|-------------|---------------|
| `dataset_names` | dict | Define dataset groups, key is group name, value is list of dataset names | `{"train": ["dataset1", "dataset2"]}` |
| `image_transform_args` | dict | VAE image preprocessing parameters | - |
| `vit_image_transform_args` | dict | ViT image preprocessing parameters | - |
| `weight` | dict | Weights of each dataset group | `{"train": 5}` |
| `num_used_data` | dict | Control data usage for each dataset group | `{"train": [0, 1000]}` |
| `is_mandatory` | bool | Whether it's a mandatory dataset | `true` |
| `shuffle_lines` | bool | Whether to shuffle data order | `true` |
| `shuffle_seed` | int | Random seed | `42` |

---

## 🚀 Step 3: Reference Configuration File in `scripts/train_sft_generic.sh`

The training script references the YAML configuration file through command line arguments:

```bash
#!/bin/bash

# Configuration file path - point to your YAML file
CONFIG_FILE="${SCRIPT_DIR}/configs/finetuned_example.yaml"

# Training script path
TRAIN_SCRIPT="${PROJECT_DIR}/train/pretrain_unified_navit_simple_ver6.py"

# Start training, passing configuration file
torchrun \
  --nproc_per_node="${num_gpus}" \
  --master_addr=127.0.0.1 \
  --master_port=29503 \
  "${TRAIN_SCRIPT}" \
  --dataset_config_file "${CONFIG_FILE}" \  # Key: pass configuration file parameter
  --resume_from_ema "${MODEL_CHECKPOINT}" \
  # ... other parameters
```

**Key Parameter Description:**
- `--dataset_config_file`: Specifies the path to the dataset configuration file
- The training script will read this file and automatically build corresponding datasets

---

## 📄 JSONL Data Format Specification

Uni-MedVL uses JSONL format data files, with one JSON object per line. Here is the standard data format:

### Basic Format
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
      "value": "<image>\n\nYou are an experienced radiologist specializing in chest X-ray analysis. Given a reference chest X-ray image, analyze and predict the expected radiological changes described in the clinical context.\n\nClinical context: Decreased cardiomegaly and resolution of interstitial edema."
    },
    {
      "from": "gpt",
      "value": "The heart size appears reduced, and there is a decrease in the hazy opacities in the lung fields, indicating reduced interstitial markings. <image>"
    }
  ]
}
```

### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `main_task_type` | string | ✅ | Task type identifier, such as "multimodal_input_output" |
| `input_img` | array | ❌ | List of input images, each element contains path and size information |
| `output_img` | array | ❌ | List of output images, same format as input_img |
| `message` | array | ✅ | Array of conversation messages, containing user and model interactions |

### Multiple Data Format Examples

#### 1. Text-only Conversation (No Images)
```json
{
  "main_task_type": "text_only",
  "message": [
    {"from": "human", "value": "What are the common symptoms of pneumonia?"},
    {"from": "gpt", "value": "Common symptoms of pneumonia include cough, fever, chest pain, and difficulty breathing..."}
  ]
}
```

#### 2. Single Input Image (Image Understanding)
```json
{
  "main_task_type": "image_understanding",
  "input_img": [
    {
      "path": "medical/images/cxr_001.png",
      "height": 1024,
      "width": 1024
    }
  ],
  "message": [
    {"from": "human", "value": "<image>\n\nAnalyze this chest X-ray and describe your findings."},
    {"from": "gpt", "value": "This chest X-ray shows clear lung fields with normal cardiac silhouette..."}
  ]
}
```

#### 3. Image-to-Image Conversion
```json
{
  "main_task_type": "image_to_image",
  "input_img": [
    {
      "path": "input/he_stain.png",
      "height": 512,
      "width": 512
    }
  ],
  "output_img": [
    {
      "path": "output/ihc_stain.png",
      "height": 512,
      "width": 512
    }
  ],
  "message": [
    {"from": "human", "value": "<image>\n\nConvert this H&E stained image to HER2 IHC staining with 3+ expression level."},
    {"from": "gpt", "value": "I'll convert the H&E stained image to HER2 IHC staining with strong 3+ membranous expression pattern. <image>"}
  ]
}
```

---

## 🔄 Complete Data Flow Explanation

### Data Loading Process

1. **Configuration File Parsing**
   ```python
   # Training script reads YAML configuration
   with open(data_args.dataset_config_file, "r") as stream:
       dataset_meta = yaml.safe_load(stream)
   ```

2. **Dataset Class Lookup**
   ```python
   # Find dataset class based on key name in YAML
   dataset_class = DATASET_REGISTRY["CounterfactualMedicalIterableDataset_ver1"]
   ```

3. **Dataset Information Retrieval**
   ```python
   # Get configuration information for specific dataset
   dataset_info = DATASET_INFO["CounterfactualMedicalIterableDataset_ver1"]["counterfactual_cxr_chexpertplus_train"]
   ```

4. **Dataset Instantiation**
   ```python
   # Create dataset instance
   dataset = dataset_class(
       dataset_name="counterfactual_train",
       data_dir=dataset_info['data_dir'],
       jsonl_path=dataset_info['jsonl_path'],
       **yaml_config_args
   )
   ```

### Key Mapping Relationships

```
YAML config file key name    → DATASET_REGISTRY key name
Dataset name in YAML        → DATASET_INFO configuration key name
Path in DATASET_INFO        → Actual data file location
```

---

## 💡 Practical Operation Example

### Example: Adding a New Medical Imaging Dataset

Let's say we want to add a new dataset called `radiology_reports` containing chest X-ray images and corresponding diagnostic reports.

#### Step 1: Update `data/dataset_info.py`

```python
# Add to DATASET_REGISTRY
DATASET_REGISTRY = {
    # ... existing registrations
    "RadiologyReportDataset": RadiologyReportDataset,
}

# Add configuration to DATASET_INFO
DATASET_INFO = {
    # ... existing configurations

    "RadiologyReportDataset": {
        'radiology_reports_train': {
            'data_dir': '/data/radiology/images',
            'jsonl_path': '/data/radiology/train_annotations.jsonl',
        },
        'radiology_reports_val': {
            'data_dir': '/data/radiology/images',
            'jsonl_path': '/data/radiology/val_annotations.jsonl',
        }
    }
}
```

#### Step 2: Create `configs/radiology_config.yaml`

```yaml
RadiologyReportDataset:
  dataset_names:
    train:
      - radiology_reports_train
    val:
      - radiology_reports_val

  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512

  vit_image_transform_args:
    image_stride: 14
    max_image_size: 980
    min_image_size: 378
    max_pixels: 2_007_040

  weight:
    train: 8
    val: 1

  num_used_data:
    train: [0, 0]
    val: [0, 0]

  is_mandatory: true
  shuffle_lines: true
  shuffle_seed: 42
```

#### Step 3: Prepare JSONL Data File

Create `/data/radiology/train_annotations.jsonl`:
```json
{"main_task_type": "image_understanding", "input_img": [{"path": "patient_001_cxr.png", "height": 1024, "width": 1024}], "message": [{"from": "human", "value": "<image>\n\nPlease analyze this chest X-ray film and provide a detailed diagnostic report."}, {"from": "gpt", "value": "Chest X-ray shows: Clear lung textures bilaterally, no obvious consolidation or exudative lesions. Heart size and shape are normal, mediastinum is centered. Diaphragmatic surface is smooth, costophrenic angles are sharp. Impression: No obvious abnormalities found on chest X-ray examination."}]}

{"main_task_type": "image_understanding", "input_img": [{"path": "patient_002_cxr.png", "height": 1024, "width": 1024}], "message": [{"from": "human", "value": "<image>\n\nAnalyze the abnormal findings on this chest X-ray."}, {"from": "gpt", "value": "Chest X-ray shows: Patchy high-density opacity in the right upper lobe with blurred borders, considered inflammatory changes. Heart size is normal, no mediastinal shift. Impression: Right upper lobe pneumonia."}]}
```

#### Step 4: Modify Training Script

```bash
# Modify configuration file path in train_sft_generic.sh
CONFIG_FILE="${SCRIPT_DIR}/configs/radiology_config.yaml"
```

## 🔬 Inference Scripts

UniMedVL provides two interactive inference scripts that use a Jupyter notebook-like cell structure. Simply run the Python files directly - they're organized with `#%%` cell separators for interactive execution. Note that these scripts are not the only way to perform inference; you may want to modify them according to your specific evaluation requirements. 

### 1. Medical VQA Inference (`interactive_vqa_inferencer.py`)

**Purpose**: Medical image analysis, diagnostic reasoning, and report generation through visual question answering.

#### Configuration Parameters
```python
DEFAULT_CONFIG = {
    # TODO: Set your model checkpoint path
    "model_path": "/path/to/unimedvl_checkpoint",
    "target_gpu_device": "0",
    "max_mem_per_gpu": "40GiB",
    "temperature": 1.0,
    "max_new_tokens": 512,
    "do_sample": True,
    "seed": 42
}
```

#### Required Path Modifications
1. **Line 42**: `ROOT = "/path/to/UniMedVL"` - Your UniMedVL installation directory
2. **Line 60**: `"model_path": "/path/to/unimedvl_checkpoint"` - Model checkpoint directory

#### Usage Example
```python
# Cell 4: VQA inference
image_path = "/path/to/your/test_image.png"
prompt = "Please analyze this chest X-ray using professional medical knowledge: 1) How is the transparency of the lung fields? 2) Are there any nodules, masses, or infiltrative lesions present? 3) Is the pleura smooth? 4) Is the contour of the cardiac silhouette normal? 5) How is the mediastinal structure? Please provide a detailed radiological report and diagnostic opinions."

temperature = 0.1
max_new_tokens = 512
show_image = True

result = inferencer.infer_single(
    image_path=image_path,
    prompt=prompt,
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    show_image=show_image
)
```

#### Cell Structure
- **Cell 1**: Environment setup and imports
- **Cell 2**: Configuration and VQAInferencer class
- **Cell 3**: Model initialization and loading
- **Cell 4**: VQA inference example

### 2. Medical Image Generation (`interactive_image_generator.py`)

**Purpose**: Medical image editing and cross-modal translation (e.g., H&E to IHC staining conversion).

**Inference Modes**:
- **Understanding Mode** (default): Context-aware editing, faster execution
- **Thinking Mode**: Deep reasoning with internal thought process, slower but more thorough

#### Configuration Parameters
```python
DEFAULT_CONFIG = {
    # TODO: Set your model checkpoint path
    "model_path": "/path/to/unimedvl_checkpoint",
    "target_gpu_device": "0",
    "max_mem_per_gpu": "40GiB",
    "seed": 42,
    "vae_transform_size": (1024, 32, 16),
    "text_do_sample": False,  # False = greedy (deterministic)
    "text_temperature": 0.3   # Only effective when text_do_sample=True
}
```

#### Required Path Modifications
1. **Line 40**: `ROOT = "/path/to/UniMedVL"` - Your UniMedVL installation directory
2. **Line 58**: `"model_path": "/path/to/unimedvl_checkpoint"` - Model checkpoint directory

#### Usage Example
```python
# Cell 4: Image editing example
image_path = "/path/to/your/test_image.png"
edit_instruction = """
Synthesize a HER2 IHC image with 2+ expression level from the given H&E stained pathology input, maintaining all anatomical structures and generating realistic 2+ immunohistochemistry patterns.
"""

# Inference mode selection
use_thinking = False  # True for deep reasoning mode

# Generation parameters
cfg_text_scale = 4.0
cfg_img_scale = 2.0
num_timesteps = 50
timestep_shift = 3.0
seed = 42

# Execute generation
result = generator.infer_single(
    image=input_image,
    instruction=final_instruction,
    cfg_text_scale=cfg_text_scale,
    cfg_img_scale=cfg_img_scale,
    timestep_shift=timestep_shift,
    num_timesteps=num_timesteps
)
```

#### Cell Structure
- **Cell 1**: Environment setup and imports
- **Cell 2**: Configuration and ImageGenerator class
- **Cell 3**: Model initialization and loading
- **Cell 4**: Image editing/generation examples

---

## 🔬 SSIM Loss 集成计划

### 📋 需求分析

目前框架支持的损失函数：
- **CE Loss**: 语言模型交叉熵损失
- **MSE Loss**: 扩散模型速度匹配损失
- **Pixel Loss**: 像素空间保真损失 (L1/L2)

计划添加：
- **SSIM Loss**: 结构相似性损失，更好地保持图像结构信息

### 🏗️ 架构分析

```
losses.py                    bagel.py                      main_sr_pixel_loss.py
┌─────────────────┐         ┌─────────────────────┐        ┌─────────────────────────┐
│ compute_ce_loss │──────▶  │                     │        │ TrainingArguments:      │
│ compute_mse_loss│──────▶  │ forward() 方法       │◀───────│  - pixel_loss_weight    │
│ compute_pixel_loss│────▶  │ 返回 loss dict      │        │  - pixel_loss_type      │
│ [新] compute_ssim_loss│──▶│                     │        │  - [新] ssim_loss_weight│
└─────────────────┘         └─────────────────────┘        └─────────────────────────┘
```

### 📝 实现步骤

#### Step 1: 在 `losses.py` 中添加 SSIM Loss 函数

**文件**: `modeling/bagel/losses.py`

```python
def compute_ssim_loss(
    x_pred: torch.Tensor,      # 预测图像 [B, C, H, W], 范围 [0, 1]
    x_gt: torch.Tensor,        # 真实图像 [B, C, H, W], 范围 [0, 1]
    mask: Optional[torch.Tensor] = None,  # 权重掩码 [B, 1, H, W]
    window_size: int = 11,     # SSIM 窗口大小
    channel: int = 3,          # 图像通道数
    size_average: bool = True, # 是否对批次求平均
) -> torch.Tensor:
    """
    Compute SSIM loss for image quality assessment.

    SSIM measures structural similarity between two images, considering:
    - Luminance: mean intensity comparison
    - Contrast: variance comparison
    - Structure: covariance comparison

    Returns:
        1 - SSIM (so that minimizing this loss maximizes SSIM)
    """
    ...
```

#### Step 2: 扩展 `compute_pixel_loss()` 以支持多种损失类型

**修改策略**: 将 SSIM 集成到现有的 `compute_pixel_loss()` 框架中，通过 `pixel_loss_type` 参数控制

```python
# pixel_loss_type 支持的值:
# - "l1": L1 像素损失
# - "l2" / "mse": L2 像素损失
# - "ssim": 纯 SSIM 损失
# - "l1+ssim": L1 + SSIM 组合损失
# - "l2+ssim": L2 + SSIM 组合损失
```

#### Step 3: 修改 `bagel.py` 的 `forward()` 方法

**文件**: `modeling/bagel/bagel.py`

添加新参数到 `forward()` 签名:
```python
def forward(
    ...
    # 现有像素损失参数
    pixel_loss_weight: float = 0.0,
    pixel_loss_type: str = "l1",
    pixel_loss_max_t: float = 0.0,
    # 新增 SSIM 损失参数
    ssim_loss_weight: float = 0.0,
    ssim_window_size: int = 11,
    ...
)
```

#### Step 4: 修改训练脚本参数

**文件**: `train/main_sr_pixel_loss.py`

在 `TrainingArguments` 中添加:
```python
@dataclass
class TrainingArguments:
    ...
    # SSIM Loss 配置
    ssim_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for SSIM loss. Set > 0 to enable."}
    )

    ssim_window_size: int = field(
        default=11,
        metadata={"help": "Window size for SSIM computation (default: 11)."}
    )

    ssim_loss_max_t: float = field(
        default=0.3,
        metadata={"help": "Apply SSIM loss only when timestep t <= this value."}
    )
```

#### Step 5: 更新训练脚本配置

**文件**: `scripts/training/train_sft_stage1_medq_unif_multinode_eyeQ1_sr_pixel_loss_small_max_T_large_pixel_weight.sh`

添加 SSIM 配置:
```bash
# SSIM Loss 配置
SSIM_LOSS_WEIGHT=1.0
SSIM_WINDOW_SIZE=11
SSIM_LOSS_MAX_T=0.3

# 在 torchrun 命令中添加参数
--ssim_loss_weight "${SSIM_LOSS_WEIGHT}" \
--ssim_window_size "${SSIM_WINDOW_SIZE}" \
--ssim_loss_max_t "${SSIM_LOSS_MAX_T}" \
```

### 🔧 灵活的损失组合设计

为了支持多种损失函数的灵活组合，建议采用以下设计模式:

**方案 A: 独立权重控制（推荐）**
```python
loss = 0
loss += ce * ce_weight
loss += mse * mse_weight
loss += pixel * pixel_loss_weight    # L1/L2
loss += ssim * ssim_loss_weight      # SSIM
```

**方案 B: 复合损失类型**
```python
# pixel_loss_type = "l1+ssim"
# 内部自动组合，通过 ssim_ratio 控制比例
pixel_loss = (1 - ssim_ratio) * l1_loss + ssim_ratio * ssim_loss
```

### 📊 训练配置示例

```bash
# 仅使用 Pixel Loss (L2)
--pixel_loss_weight 10000 \
--pixel_loss_type "l2" \
--ssim_loss_weight 0 \

# 仅使用 SSIM Loss
--pixel_loss_weight 0 \
--ssim_loss_weight 1.0 \

# 组合使用 Pixel + SSIM
--pixel_loss_weight 5000 \
--pixel_loss_type "l2" \
--ssim_loss_weight 0.5 \
```

### ⚠️ 注意事项

1. **SSIM 计算成本**: SSIM 比 L1/L2 计算量大，建议使用分块计算
2. **数值范围**: SSIM 输出在 [0, 1]，需要合理设置权重
3. **时间步过滤**: 与 pixel loss 类似，SSIM 也应只在低噪声时间步应用
4. **内存管理**: 复用 pixel loss 的分块 VAE decode 机制

### 📁 需要修改的文件清单

| 文件路径 | 修改内容 |
|---------|---------|
| `modeling/bagel/losses.py` | 添加 `compute_ssim_loss()` 函数 |
| `modeling/bagel/bagel.py` | 扩展 `forward()` 支持 SSIM |
| `train/main_sr_pixel_loss.py` | 添加 SSIM 相关训练参数 |
| `scripts/training/*.sh` | 添加 SSIM 配置选项 |

---
