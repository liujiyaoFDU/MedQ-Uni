# MedQ-Uni 批量测试脚本 Ver3 使用说明

## 📌 概述

Ver3 版本是一个支持多模型、多GPU批次并行的批量测试脚本。相比 Ver2，它实现了真正的多设备并行运行。

## ✨ 主要特性

1. **多模型支持** - 通过 `MODEL_PATHS` 数组配置多个模型checkpoint
2. **多GPU批次并行** - 真正实现多GPU同时运行，而非顺序执行
3. **笛卡尔积任务分配** - 自动生成 模型 × 测试文件 的所有组合
4. **智能命名** - 输出目录自动包含模型路径信息
5. **独立日志** - 每个任务有专属日志文件，便于追踪调试
6. **任务追踪** - 自动记录任务状态，生成汇总报告

## 🚀 快速开始

### 1. 配置脚本

编辑 `MedQ-Uni_run_batch_test_ver3.sh`，修改以下配置：

```bash
# 配置多个模型路径
MODEL_PATHS=(
    "/path/to/model1/checkpoint/0008000"
    "/path/to/model2/checkpoint/0012000"
    # 添加更多模型...
)

# 配置要使用的GPU
TARGET_GPUS=("0" "1")  # 使用GPU 0和1

# 测试文件已预配置12个数据集，可根据需要调整
```

### 2. 运行脚本

```bash
cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni
./MedQ-Uni_run_batch_test_ver3.sh
```

### 3. 查看结果

脚本运行后，输出结构如下：

```
MedQ-Uni_results_ver3/
├── logs/                           # 日志目录
│   ├── training_stage1_stage1_medq_v1_0008000_AAPM-CT-MAR_test_gpu0.log
│   └── ...
├─�� training_stage1_stage1_medq_v1_0008000_AAPM-CT-MAR_test_20251223_143025/
│   ├── images/                     # 生成的图像
│   ├── results.jsonl              # 详细结果
│   └── statistics.json            # 统计指标
├── task_status.txt                # 任务状态记录
└── summary_report.txt             # 汇总报告
```

## ��� 运行示例

假设配置如下：
- 2个模型 checkpoint
- 12个测试数据集
- 2个GPU (0, 1)

脚本将：
1. 生成 2 × 12 = 24 个任务
2. 分成 24 ÷ 2 = 12 个批次
3. 每批次并行运行2个任务（分别在GPU 0和1上）

运行输出示例：
```
============================================================
任务调度信息
============================================================
总任务数: 24
可用GPU数: 2 (0 1)
批次数: 12
调度策略: 批次并行（每批 2 个任务）
============================================================

------------------------------------------------------------
批次 1/12 开始 (2025-12-23 14:30:25)
------------------------------------------------------------
  → 启动任务 0 (GPU 0): training_stage1_stage1_medq_v1_0008000 / AAPM-CT-MAR_test
  → 启动任务 1 (GPU 1): training_stage1_stage1_medq_v1_0008000 / AMIR_CT_Low-Dose_CT_denoising_test

等待批次 1 完成...

批次 1 完成: ✓ 2 个成功, ✗ 0 个失败

...
```

## 🔧 关键参数说明

### MODEL_PATHS
指定要测试的模型checkpoint路径。可以添加多个模型，脚本会自动提取模型信息用于命名。

**提取规则**：倒数第三级目录 + 倒数第二级目录 + checkpoint步数

例如：
```
路径: /path/training_stage1/stage1_medq_v1/0008000
提取: training_stage1_stage1_medq_v1_0008000
```

### TARGET_GPUS
指定要使用的GPU设备。可以是：
- 单GPU: `TARGET_GPUS=("0")`
- 双GPU: `TARGET_GPUS=("0" "1")`
- 四GPU: `TARGET_GPUS=("0" "1" "2" "3")`

### NUM_SAMPLES
控制每个数据集测试的样本数量：
- `NUM_SAMPLES=50` - 每个数据集只测试前50个样本（快速验证）
- `NUM_SAMPLES=-1` - 测试全部样本（完整测试）

## 📝 输出文件说明

### 1. 日志文件 (logs/)
每个任务有独立的日志文件，格式：
```
{模型信息}_{数据集名称}_gpu{GPU编号}.log
```

包含：
- 任务开始/结束时间
- 模型和数据集信息
- Python脚本的完整输出
- 错误信息（如果有）

### 2. 结果目录
每个任务生成一个独立的结果目录，格式：
```
{模型信息}_{数据集名称}_{时间戳}/
```

包含：
- `images/` - 输入、真值、预测图像
- `results.jsonl` - 每个样本的详细结果
- `statistics.json` - PSNR、SSIM等统计指标

### 3. 任务状态文件 (task_status.txt)
记录每个任务的执行状态，格式：
```
任务ID|状态|模型信息|数据集名称|耗时|退出码
```

例如：
```
0|SUCCESS|training_stage1_stage1_medq_v1_0008000|AAPM-CT-MAR_test|1245
1|FAILED|training_stage1_stage1_medq_v1_0008000|AMIR_CT_denoising_test|523|1
```

### 4. 汇总报告 (summary_report.txt)
包含：
- 配置信息（模型数量、测试集数量、GPU设备）
- 任务执行统计（成功/失败数量）
- 失败任务列表
- 总耗时

## 🆚 与 Ver2 的对比

| 特性 | Ver2 | Ver3 |
|------|------|------|
| 多模型支持 | ✗ 仅支持单个模型 | ✓ 支持多个模型 |
| GPU并行 | ✗ 顺序执行 | ✓ 批次并行执行 |
| 任务调度 | 简单循环 | 智能批次调度器 |
| 日志管理 | 所有输出混在一起 | 每个任务独立日志 |
| 输出命名 | 仅数据集名称 | 包含模型信息 |
| 状态追踪 | ✗ 无 | ✓ 完整的状态记录 |
| 汇总报告 | ✗ 无 | ✓ 自动生成报告 |

## ⚙️ 高级配置

### 自定义生成参数
```bash
CFG_TEXT_SCALE=4.0      # 文本CFG scale
CFG_IMG_SCALE=2.0       # 图像CFG scale
NUM_TIMESTEPS=50        # 扩散步数
TIMESTEP_SHIFT=1.0      # 时间步偏移
SEED=42                 # 随机种子
```

### 显存管理
```bash
MAX_MEM="130GiB"        # 每GPU显存限制
```

### 环境变量
```bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
```

## 🐛 故障排查

### 1. 任务失败
检查对应的日志文件：
```bash
cat MedQ-Uni_results_ver3/logs/{模型信息}_{数据集}_gpu{N}.log
```

### 2. GPU显存不足
减小 `MAX_MEM` 或减少同时运行的任务数（使用更少的GPU）。

### 3. 路径不存在
检查 `MODEL_PATHS` 和 `ANNOTATION_FILES` 中的路径是否正确。

### 4. 查看任务状态
```bash
cat MedQ-Uni_results_ver3/task_status.txt
```

## 💡 使用技巧

1. **快速验证**：设置 `NUM_SAMPLES=5` 快速测试脚本是否正常工作
2. **分批测试**：可以将不同的模型分到不同的脚本实例中运行
3. **监控进度**：使用 `tail -f` 实时查看日志文件
4. **资源利用**：根据GPU显存大小调整并行任务数

## 📞 支持

如有问题，请检查：
1. 日志文件 `logs/*.log`
2. 任务状态 `task_status.txt`
3. 汇总报告 `summary_report.txt`
