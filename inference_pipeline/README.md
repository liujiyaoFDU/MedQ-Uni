# 推理流程 (Inference Pipeline)

本目录包含 MedQ-Uni 模型的批量推理脚本、执行脚本和结果统计工具。

## 📁 目录结构

```
inference_pipeline/
├── MedQ-Uni_run_batch_test1.py          # Python 批量测试脚本 (VAE min=512)
├── MedQ-Uni_run_batch_test2.py          # Python 批量测试脚本 (VAE min=256)
├── MedQ-Uni_run_batch_test_ver1.sh      # Shell 脚本: 单检查点推理
├── MedQ-Uni_run_batch_test_ver2.sh      # Shell 脚本: 多检查点并行推理
├── parse_statistics_to_csv.py            # 统计结果聚合工具
└── README.md                             # 本文档
```

## 📄 脚本说明

### Python 测试脚本

#### MedQ-Uni_run_batch_test1.py
- **功能**: 批量推理测试脚本，支持 PSNR/SSIM 指标计算
- **特点**: VAE transform min size = **512**
- **核心类**:
  - `ImageGenerator`: 模型加载和图像生成
  - `BatchTester`: 批量处理，支持断点续传
- **指标计算**: `calculate_psnr()`, `calculate_ssim()`
- **输出**: `results.jsonl`, `statistics.json`, `images/`

#### MedQ-Uni_run_batch_test2.py
- **功能**: 与 test1.py 相同的批量推理功能
- **特点**: VAE transform min size = **256**
- **其他**: 与 test1.py 代码几乎完全相同

> **注意**: 两个脚本的唯一区别在于 VAE transform 的最小尺寸配置 (line 618)，影响图像预处理的下采样策略。

### Shell 执行脚本

#### MedQ-Uni_run_batch_test_ver1.sh
- **用途**: 单检查点顺序推理
- **特点**:
  - 单个模型检查点 (0016000)
  - 单 GPU 执行 (GPU 3)
  - 顺序处理多个数据集 (8 个测试集)
  - 简单的循环执行
- **输出目录**: `MedQ-Uni_results_16000/{DATASET_NAME}_{TIMESTAMP}/`
- **适用场景**: 快速测试单个检查点，调试推理流程

#### MedQ-Uni_run_batch_test_ver2.sh
- **用途**: 多检查点并行推理
- **特点**:
  - 多个模型检查点 (5个: 0004000, 0008000, 0012000, 0016000, 0020000)
  - 多 GPU 并行执行 (GPU 0, GPU 1)
  - 处理多个数据集 (12 个训练集)
  - 轮询任务分配 (round-robin)
  - 每个检查点独立执行，GPU 间并行
  - 详细的日志记录 (按 GPU 和检查点分文件)
- **输出目录**: `stage1_train_50_ver1/{CHECKPOINT_NAME}/{DATASET_NAME}/`
- **适用场景**: 批量评估多个检查点，生产环境大规模推理

### 统计分析脚本

#### parse_statistics_to_csv.py
- **功能**: 聚合多个 `statistics.json` 文件到统一 CSV 文件
- **处理流程**:
  1. 递归扫描指定目录下所有 `statistics.json` 文件
  2. 从文件路径提取元数据 (model_id, split)
  3. 解析 overall 和 by_task_type 指标
  4. 横向展开多任务类型 (task1_*, task2_*, ...)
  5. 生成统一的 CSV 汇总表
- **输出**: `summary.csv` (包含所有检查点的性能指标)
- **配置**: 可通过脚本顶部的 `DEFAULT_INPUT_DIRECTORIES` 修改默认输入路径

## 🔍 版本差异对比

### test1.py vs test2.py

| 特性 | test1.py | test2.py |
|------|----------|----------|
| **VAE Transform Min Size** | 512 | **256** |
| **代码行数** | 816 | 813 |
| **核心功能** | ✅ 相同 | ✅ 相同 |
| **ImageGenerator 类** | ✅ | ✅ |
| **BatchTester 类** | ✅ | ✅ |
| **PSNR/SSIM 计算** | ✅ | ✅ |
| **断点续传** | ✅ | ✅ |

**关键区别** (test2.py:618):
```python
# test1.py
vae_transform = ImageTransform(img_size=1024, min_size=512, num_buckets=16)

# test2.py
vae_transform = ImageTransform(img_size=1024, min_size=256, num_buckets=16)
```

**选择建议**:
- `min_size=256`: 更激进的下采样，速度更快，显存占用更小，可能损失细节
- `min_size=512`: 保留更多细节，显存占用更大，推理速度较慢

### ver1.sh vs ver2.sh

| 特性 | ver1.sh | ver2.sh |
|------|---------|---------|
| **检查点数量** | 1 个 | 5 个 |
| **检查点版本** | 0016000 | 0004000-0020000 |
| **GPU 使用** | 单 GPU (3) | 双 GPU (0, 1) |
| **数据集数量** | 8 个 | 12 个 |
| **执行方式** | 顺序执行 | 并行执行 |
| **任务分配** | N/A | 轮询 (round-robin) |
| **日志管理** | 标准输出 | 独立日志文件 |
| **输出结构** | 扁平 | 按检查点分层 |
| **脚本行数** | 139 | 291 |

**执行流程对比**:

**ver1.sh**:
```
for 每个数据集:
    推理 -> 等待完成 -> 下一个
```

**ver2.sh**:
```
for 每个检查点:
    for 每个 GPU:
        分配任务列表 -> 后台执行
    等待所有 GPU 完成
    等待 15 秒 (GPU 清理)
```

## 🚀 使用方法

### 前置要求

1. **环境激活**:
   ```bash
   cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni
   source .venv/bin/activate
   ```

2. **依赖检查**:
   - PyTorch >= 2.0
   - transformers, accelerate
   - safetensors
   - PIL, numpy, pandas
   - scikit-image (SSIM 计算)
   - tqdm

### 方法 1: 单检查点测试 (ver1)

**适用场景**: 快速测试单个检查点，调试推理参数

```bash
# 从项目根目录运行
cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni

# 执行 ver1 脚本 (会自动 cd 到正确目录)
bash inference_pipeline/MedQ-Uni_run_batch_test_ver1.sh
```

**脚本内部流程**:
1. `cd` 到项目根目录 (line 9)
2. 激活虚拟环境
3. 设置环境变量 (CUDA_VISIBLE_DEVICES, TOKENIZERS_PARALLELISM 等)
4. 循环处理每个数据集:
   - 检查 annotation 文件存在性
   - 创建时间戳输出目录
   - 调用 `python inference_pipeline/MedQ-Uni_run_batch_test2.py`
   - 保存结果到 `MedQ-Uni_results_16000/{DATASET}_{TIMESTAMP}/`

**自定义配置** (修改 ver1.sh):
```bash
# 选择不同的 GPU
TARGET_GPU="0"  # 默认是 3

# 修改显存限制
MAX_MEM="80GiB"  # 默认是 130GiB

# 添加/删除数据集
ANNOTATION_FILES=(
    "/path/to/your/dataset1.jsonl"
    "/path/to/your/dataset2.jsonl"
)

# 限制测试样本数 (快速验证)
NUM_SAMPLES=10  # 默认是 50
```

### 方法 2: 多检查点并行测试 (ver2)

**适用场景**: 批量评估多个检查点，对比不同训练阶段的性能

```bash
# 从项目根目录运行
cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni

# 执行 ver2 脚本
bash inference_pipeline/MedQ-Uni_run_batch_test_ver2.sh
```

**脚本内部流程**:
1. `cd` 到项目根目录 (line 11)
2. 激活虚拟环境
3. 外层循环: 遍历每个检查点 (5 个)
4. 内层循环: 为每个 GPU 分配任务
   - GPU 0: 处理数据集 0, 2, 4, 6, 8, 10 (偶数索引)
   - GPU 1: 处理数据集 1, 3, 5, 7, 9, 11 (奇数索引)
5. 并行执行: 两个 GPU 同时推理 (后台进程)
6. 等待当前检查点所有任务完成
7. GPU 清理: 等待 15 秒后处理下一个检查点
8. 保存日志: `{BASE_OUTPUT_DIR}/{CHECKPOINT}/gpu_{GPU_ID}.log`

**任务分配示例** (假设 12 个数据集):
```
Checkpoint: 0004000
  GPU 0 后台任务: dataset_0, dataset_2, dataset_4, ..., dataset_10
  GPU 1 后台任务: dataset_1, dataset_3, dataset_5, ..., dataset_11
  等待两个 GPU 完成

Checkpoint: 0008000
  GPU 0 后台任务: ...
  GPU 1 后台任务: ...
  ...
```

**自定义配置** (修改 ver2.sh):
```bash
# 修改检查点列表
CHECKPOINTS=(
    "/path/to/checkpoint1"
    "/path/to/checkpoint2"
)

# 修改 GPU 配置
GPUS=("0" "1" "2" "3")  # 使用 4 个 GPU

# 修改数据集列表
ANNOTATION_FILES=(
    "/path/to/dataset1.jsonl"
    # ...
)

# 修改基础输出目录
BASE_OUTPUT_DIR="my_experiment_results"
```

### 方法 3: 统计结果聚合

推理完成后，使用 `parse_statistics_to_csv.py` 聚合所有 `statistics.json` 文件:

```bash
# 方式 A: 使用脚本内置的默认路径
python inference_pipeline/parse_statistics_to_csv.py

# 方式 B: 指定自定义路径
python inference_pipeline/parse_statistics_to_csv.py \
    -d ./MedQ-Uni_results_16000 \
    -d ./stage1_train_50_ver1 \
    -o my_summary.csv

# 方式 C: 启用详细日志
python inference_pipeline/parse_statistics_to_csv.py -v
```

**输出示例** (`summary.csv`):
```csv
model_id,split,total_samples,psnr_mean,psnr_std,ssim_mean,ssim_std,avg_inference_time,task1_type,task1_count,task1_psnr_mean,...,timestamp
stage1_medq_2nodes_unif_combined_v1_0008000,test,100,28.45,3.21,0.876,0.043,1.23,denoising,50,29.12,...,2024-12-23T10:15:32
stage1_medq_2nodes_unif_combined_v1_0016000,test,100,29.87,2.98,0.891,0.038,1.18,denoising,50,30.45,...,2024-12-23T12:30:45
```

**自定义默认路径** (修改 parse_statistics_to_csv.py):
```python
# Line 29-32
DEFAULT_INPUT_DIRECTORIES = [
    "/your/custom/path/checkpoint_0008000",
    "/your/custom/path/checkpoint_0016000",
]
```

## 📊 输出结果

### 单次推理输出 (每个数据集)

每次推理会在输出目录中生成以下文件:

```
{OUTPUT_DIR}/
├── results.jsonl                  # 每个样本的详细结果 (JSONL 格式)
├── statistics.json                # 聚合统计指标 (JSON 格式)
└── images/                        # 生成的图像文件
    ├── sample_0001.png
    ├── sample_0002.png
    └── ...
```

#### results.jsonl 格式
```jsonl
{"id": "sample_0001", "psnr": 28.45, "ssim": 0.876, "task_type": "denoising", "inference_time": 1.23}
{"id": "sample_0002", "psnr": 29.12, "ssim": 0.881, "task_type": "super_resolution", "inference_time": 1.18}
...
```

#### statistics.json 格式
```json
{
  "overall": {
    "total_samples": 100,
    "psnr_mean": 28.45,
    "psnr_std": 3.21,
    "ssim_mean": 0.876,
    "ssim_std": 0.043,
    "avg_inference_time": 1.23
  },
  "by_task_type": {
    "denoising": {
      "count": 50,
      "psnr_mean": 29.12,
      "psnr_std": 2.87,
      "ssim_mean": 0.889,
      "ssim_std": 0.038
    },
    "super_resolution": {
      "count": 50,
      "psnr_mean": 27.78,
      "psnr_std": 3.45,
      "ssim_mean": 0.863,
      "ssim_std": 0.047
    }
  },
  "timestamp": "2024-12-23T10:15:32"
}
```

### 目录结构示例

#### ver1 输出结构 (扁平)
```
MedQ-Uni_results_16000/
├── AAPM-CT-MAR_test_20241223_101532/
│   ├── results.jsonl
│   ├── statistics.json
│   └── images/
├── AMIR_MRI_super-resolution_test_20241223_103045/
│   ├── results.jsonl
│   ├── statistics.json
│   └── images/
└── ...
```

#### ver2 输出结构 (分层)
```
stage1_train_50_ver1/
├── stage1_medq_2nodes_unif_combined_v1_0004000/
│   ├── AAPM-CT-MAR_test/
│   │   ├── results.jsonl
│   │   ├── statistics.json
│   │   └── images/
│   ├── AMIR_MRI_super-resolution_test/
│   │   └── ...
│   ├── gpu_0.log                    # GPU 0 的执行日志
│   └── gpu_1.log                    # GPU 1 的执行日志
├── stage1_medq_2nodes_unif_combined_v1_0008000/
│   └── ...
├── stage1_medq_2nodes_unif_combined_v1_0012000/
│   └── ...
├── stage1_medq_2nodes_unif_combined_v1_0016000/
│   └── ...
└── stage1_medq_2nodes_unif_combined_v1_0020000/
    └── ...
```

### 聚合后的 CSV 输出

`parse_statistics_to_csv.py` 生成的 `summary.csv`:

**列结构**:
- 基础列: `model_id`, `split`, `total_samples`, `psnr_mean`, `psnr_std`, `ssim_mean`, `ssim_std`, `avg_inference_time`
- 任务列 (动态): `task1_type`, `task1_count`, `task1_psnr_mean`, `task1_psnr_std`, `task1_ssim_mean`, `task1_ssim_std`
- 更多任务: `task2_*`, `task3_*`, ... (根据实际任务类型数量动态生成)
- 时间戳: `timestamp`

**示例数据**:
```csv
model_id,split,total_samples,psnr_mean,psnr_std,ssim_mean,ssim_std,avg_inference_time,task1_type,task1_count,task1_psnr_mean,task1_psnr_std,task1_ssim_mean,task1_ssim_std,task2_type,task2_count,task2_psnr_mean,task2_psnr_std,task2_ssim_mean,task2_ssim_std,timestamp
stage1_medq_2nodes_unif_combined_v1_0004000,test,100,26.32,3.45,0.854,0.052,1.45,denoising,50,27.15,3.12,0.867,0.048,super_resolution,50,25.49,3.67,0.841,0.055,2024-12-23T10:15:32
stage1_medq_2nodes_unif_combined_v1_0008000,test,100,28.45,3.21,0.876,0.043,1.23,denoising,50,29.12,2.87,0.889,0.038,super_resolution,50,27.78,3.45,0.863,0.047,2024-12-23T12:30:45
...
```

## ⚙️ 技术说明

### Python 脚本导入机制

**关键代码** (test1.py:43-54, test2.py:43-54):
```python
# UniMedVL imports
ROOT = "/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/MedQ-Uni/"
sys.path.append(ROOT)

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, ...
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer
```

**工作原理**:
1. `ROOT` 定义了共享模块库的绝对路径
2. `sys.path.append(ROOT)` 将该路径添加到 Python 模块搜索路径
3. 后续的 `from data.xxx import yyy` 会从 ROOT 路径下查找模块

**重要特性**:
- ✅ ROOT 是绝对路径，不受脚本所在位置影响
- ✅ 移动脚本到 `inference_pipeline/` 不会破坏导入
- ✅ 无需修改任何 Python 导入语句

**为什么使用两个不同的路径?**
- **共享模块库**: `/inspire/hdd/.../MedQ-Uni/` (存放 data, modeling, inferencer 模块)
- **项目工作目录**: `/mnt/shared-storage-user/.../MedQ-Uni/` (存放脚本、数据、结果)

这种设计允许多个项目共享同一套模型代码库，同时保持各自独立的数据和实验结果。

### Shell 脚本执行流程

**关键机制** (ver1.sh:9, ver2.sh:11):
```bash
cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni
```

**工作原理**:
1. Shell 脚本首先 `cd` 到项目根目录 (绝对路径)
2. 然后调用 `python inference_pipeline/MedQ-Uni_run_batch_test2.py` (相对于根目录)
3. Python 脚本内部使用绝对路径 `ROOT` 导入模块

**重要说明**:
- ⚠️ Shell 脚本应该从任意位置运行，它会自动 `cd` 到正确位置
- ⚠️ 输出目录 (如 `BASE_OUTPUT_DIR="MedQ-Uni_results_16000"`) 是相对路径，会在根目录创建
- ✅ 修改后的 Shell 脚本已更新 Python 调用路径为 `inference_pipeline/...`

### 统计数据处理流程

**parse_statistics_to_csv.py 处理流程**:

1. **第一遍扫描** (确定最大任务类型数量):
   ```python
   def determine_max_task_count(json_files):
       max_tasks = 0
       for file in json_files:
           data = json.load(file)
           max_tasks = max(max_tasks, len(data['by_task_type']))
       return max_tasks
   ```

2. **生成列名** (动态列):
   ```python
   def generate_column_names(max_tasks):
       columns = ['model_id', 'split', 'total_samples', ...]
       for i in range(1, max_tasks + 1):
           columns += [f'task{i}_type', f'task{i}_count', ...]
       return columns
   ```

3. **第二遍扫描** (解析数据):
   - 提取元数据 (model_id, split)
   - 解析 overall 指标
   - 横向展开 by_task_type (按字母顺序排序)
   - 填充空白列 (如果某文件任务数少于 max_tasks)

4. **输出 CSV**:
   ```python
   df = pd.DataFrame(rows, columns=columns)
   df.to_csv(output_csv, index=False)
   ```

**处理示例**:
```
输入:
- file1.json: 1 个任务类型 (denoising)
- file2.json: 2 个任务类型 (denoising, super_resolution)
- file3.json: 3 个任务类型 (denoising, super_resolution, restoration)

第一遍扫描 -> max_tasks = 3

生成列名 -> [..., task1_*, task2_*, task3_*]

第二遍扫描:
- file1 -> task1=denoising, task2=空, task3=空
- file2 -> task1=denoising, task2=super_resolution, task3=空
- file3 -> task1=denoising, task2=restoration, task3=super_resolution
```

## 📋 依赖要求

### Python 环境
- Python >= 3.8
- PyTorch >= 2.0 (CUDA 支持)

### 核心依赖
```txt
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
safetensors>=0.3.0
Pillow>=9.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-image>=0.19.0  # SSIM 计算
tqdm>=4.62.0
```

### 系统要求
- GPU: NVIDIA GPU with CUDA support (推荐 24GB+ 显存)
- 磁盘: 充足的存储空间用于保存推理结果和图像
- 内存: 建议 32GB+ RAM

### 安装依赖
```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate safetensors
pip install Pillow numpy pandas scikit-image tqdm
```

## 📝 常见问题 (FAQ)

### Q1: 如何选择 test1.py 还是 test2.py?
**A**: 两者的区别仅在于 VAE transform 的 min_size 参数:
- `test2.py` (min=256): 更快，显存占用小，适合快速测试
- `test1.py` (min=512): 保留更多细节，适合最终评估

推荐使用 `test2.py` (ver1.sh 和 ver2.sh 默认都调用 test2.py)。

### Q2: 如何修改要测试的数据集?
**A**: 编辑对应的 Shell 脚本 (ver1.sh 或 ver2.sh):
```bash
# 修改 ANNOTATION_FILES 数组
ANNOTATION_FILES=(
    "/path/to/your/dataset1.jsonl"
    "/path/to/your/dataset2.jsonl"
)
```

### Q3: 如何使用不同的 GPU?
**A**:
- **ver1.sh**: 修改 `TARGET_GPU="3"` 为你想要的 GPU 编号
- **ver2.sh**: 修改 `GPUS=("0" "1")` 数组

### Q4: 推理速度太慢怎么办?
**A**:
1. 减少 `NUM_SAMPLES` (测试更少样本)
2. 使用 `test2.py` (min_size=256, 更快)
3. 增加 GPU 数量 (ver2.sh 支持多 GPU 并行)
4. 调整 `max_mem_per_gpu` (增大显存分配)

### Q5: 如何添加新的检查点到 ver2.sh?
**A**: 修改 `CHECKPOINTS` 数组:
```bash
CHECKPOINTS=(
    "/path/to/checkpoint1"
    "/path/to/checkpoint2"
    "/path/to/your/new/checkpoint"
)
```

### Q6: parse_statistics_to_csv.py 找不到文件怎么办?
**A**:
1. 检查默认路径配置 (parse_statistics_to_csv.py:29-32)
2. 使用 `-d` 参数手动指定目录:
   ```bash
   python inference_pipeline/parse_statistics_to_csv.py -d /your/results/path
   ```

### Q7: 如何处理 CUDA Out of Memory 错误?
**A**:
1. 减少 `max_mem_per_gpu` 值
2. 减少 batch size (如果脚本支持)
3. 使用 `test2.py` (min_size=256, 显存占用更小)
4. 关闭其他占用 GPU 的进程

### Q8: 如何并行处理更多 GPU (ver2.sh)?
**A**: 修改 `GPUS` 数组:
```bash
# 使用 4 个 GPU
GPUS=("0" "1" "2" "3")
```
脚本会自动进行轮询任务分配。

## 🔧 故障排除

### 问题 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'data'
```
**原因**: sys.path 没有正确添加 ROOT 路径
**解决**: 检查 Python 脚本的 ROOT 路径是否正确，确保 `/inspire/hdd/.../MedQ-Uni/` 路径存在且包含 `data`, `modeling`, `inferencer` 模块

### 问题 2: Shell 脚本找不到 Python 脚本
```
python: can't open file 'MedQ-Uni_run_batch_test2.py': [Errno 2] No such file or directory
```
**原因**: Shell 脚本没有从项目根目录运行，或 Python 路径未更新
**解决**: 确保 Shell 脚本已更新为 `python inference_pipeline/MedQ-Uni_run_batch_test2.py`

### 问题 3: parse_statistics_to_csv.py 输出为空
```
WARNING: No statistics.json files found
```
**原因**: 默认路径不正确，或推理尚未完成
**解决**:
1. 确认推理已完成并生成 `statistics.json`
2. 使用 `-v` 查看详细日志: `python parse_statistics_to_csv.py -v`
3. 手动指定路径: `python parse_statistics_to_csv.py -d /your/results/path`

## 📌 最佳实践

1. **测试流程**:
   - 先用 ver1.sh 测试单个检查点，确保配置正确
   - 再用 ver2.sh 批量处理多个检查点

2. **GPU 管理**:
   - 使用 `nvidia-smi` 监控 GPU 使用情况
   - 合理分配 GPU (避免过载)
   - ver2.sh 会在检查点间等待 15 秒，允许 GPU 清理显存

3. **结果管理**:
   - 定期备份推理结果
   - 使用 `parse_statistics_to_csv.py` 及时聚合统计数据
   - 使用有意义的输出目录名称

4. **调试技巧**:
   - 使用 `NUM_SAMPLES=10` 快速验证
   - 使用 `--verbose` 查看详细日志
   - 检查 `gpu_*.log` 日志文件排查问题

## 📞 联系方式

如有问题或建议，请联系项目维护者或提交 Issue。

---

**最后更新**: 2024-12-23
**维护者**: MedQ-Uni Team
