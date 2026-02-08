# MedQ-Uni 批量测试脚本 Ver2 使用说明

## 功能特性

Ver2 版本支持以下功能：

1. **多 Checkpoint 支持**：可以同时测试多个不同的模型 checkpoint
2. **多数据集支持**：可以测试多个不同的 jsonl 文件
3. **按 Checkpoint 分文件夹**：每个 checkpoint 的推理结果保存在独立的文件夹中
4. **多 GPU 并行推理**：使用多个 GPU 并行执行不同的推理任务，每个任务使用单个 GPU

## 核心机制

### GPU 并行执行策略

- **每个推理任务使用单个 GPU**：脚本不使用多 GPU 模型并行，而是每个任务独占一个 GPU
- **任务队列管理**：当指定多个 GPU 时，脚本会智能分配空闲的 GPU 来执行不同的推理任务
- **自动等待机制**：当所有 GPU 都在使用时，新任务会等待直到有 GPU 空闲

#### 示例

如果你指定了 4 个 GPU：
```bash
GPUS=(
    "0"
    "1"
    "2"
    "3"
)
```

并且有 6 个 jsonl 文件要测试，那么：
- GPU 0、1、2、3 会同时开始执行前 4 个任务
- 当其中一个 GPU 完成任务后，会自动开始执行第 5 个任务
- 所有任务会充分利用可用的 GPU 并行执行

### 输出目录结构

```
MedQ-Uni_results_multi_ckpt/
├── checkpoint1_name/
│   ├── dataset1_test/
│   ├── dataset2_test/
│   └── dataset3_test/
├── checkpoint2_name/
│   ├── dataset1_test/
│   ├── dataset2_test/
│   └── dataset3_test/
```

每个推理任务的日志会保存在：`{output_dir}_gpu{N}.log`

## 配置说明

### 1. 配置 Checkpoints

在脚本的 `CHECKPOINTS` 数组中添加你要测试的所有 checkpoint 路径：

```bash
CHECKPOINTS=(
    "/path/to/checkpoint1"
    "/path/to/checkpoint2"
    "/path/to/checkpoint3"
)
```

### 2. 配置测试数据集

在 `ANNOTATION_FILES` 数组中添加你要测试的所有 jsonl 文件：

```bash
ANNOTATION_FILES=(
    "/path/to/dataset1_test.jsonl"
    "/path/to/dataset2_test.jsonl"
    "/path/to/dataset3_test.jsonl"
)
```

### 3. 配置可用的 GPU

在 `GPUS` 数组中指定可用的 GPU 编号：

```bash
GPUS=(
    "0"
    "1"
    "2"
    "3"
)
```

**注意**：
- 可以指定任意数量的 GPU
- GPU 编号对应物理 GPU 的编号
- 更多的 GPU 意味着更多的任务可以并行执行

### 4. 其他配置参数

```bash
# 图像根目录
IMAGE_ROOT="/path/to/images"

# 输出根目录
BASE_OUTPUT_DIR="MedQ-Uni_results_multi_ckpt"

# 每张卡的显存限制
MAX_MEM="130GiB"

# 样本数量控制（-1表示全部，>0表示测试前N个样本）
NUM_SAMPLES=50

# 生成参数
CFG_TEXT_SCALE=4.0
CFG_IMG_SCALE=2.0
NUM_TIMESTEPS=50
TIMESTEP_SHIFT=1.0
SEED=42
```

## 使用方法

### 1. 基本使用

```bash
# 赋予执行权限
chmod +x MedQ-Uni_run_batch_test_ver2.sh

# 执行脚本
./MedQ-Uni_run_batch_test_ver2.sh
```

### 2. 后台运行

```bash
# 使用 nohup 在后台运行
nohup ./MedQ-Uni_run_batch_test_ver2.sh > batch_test.log 2>&1 &

# 查看日志
tail -f batch_test.log
```

### 3. 使用 screen 或 tmux

```bash
# 创建新的 screen 会话
screen -S medq_test

# 运行脚本
./MedQ-Uni_run_batch_test_ver2.sh

# 断开会话：Ctrl+A, D
# 重新连接：screen -r medq_test
```

## 执行流程

1. **初始化**：加载配置，初始化 GPU 状态
2. **遍历 Checkpoints**：对每个 checkpoint 创建输出文件夹
3. **遍历数据集**：对每个 jsonl 文件创建推理任务
4. **GPU 分配**：等待获取空闲的 GPU
5. **并行执行**：在后台启动推理任务
6. **等待完成**：等待所有任务完成
7. **统计结果**：输出成功/失败任务数和总耗时

## 输出信息

脚本会实时输出以下信息：

```
============================================================
批量测试配置
============================================================
Checkpoints 数量: 2
测试集数量: 6
可用 GPU 数量: 4
可用 GPUs: 0 1 2 3
样本模式: 前 50 个样本
============================================================

总任务数: 12

[GPU 0] 开始推理: dataset1_test (checkpoint: ckpt1_0016000)
[GPU 1] 开始推理: dataset2_test (checkpoint: ckpt1_0016000)
...
[GPU 0] ✓ 完成: dataset1_test (checkpoint: ckpt1_0016000)
...

============================================================
所有任务完成!
============================================================
总任务数: 12
成功: 11
失败: 1
总耗时: 2时30分15秒
结果保存在: MedQ-Uni_results_multi_ckpt/
============================================================
```

## 常见使用场景

### 场景 1：测试单个 Checkpoint，使用 4 个 GPU 并行

```bash
CHECKPOINTS=(
    "/path/to/checkpoint/0016000"
)

GPUS=(
    "0"
    "1"
    "2"
    "3"
)
```

### 场景 2：测试多个 Checkpoints，使用 2 个 GPU 并行

```bash
CHECKPOINTS=(
    "/path/to/checkpoint/0010000"
    "/path/to/checkpoint/0016000"
    "/path/to/checkpoint/0020000"
)

GPUS=(
    "0"
    "1"
)
```

### 场景 3：只使用单个 GPU（顺序执行）

```bash
GPUS=(
    "3"
)
```

## 注意事项

1. **GPU 显存**：确保每个 GPU 有足够的显存运行模型（默认需要 130GB）
2. **磁盘空间**：确保有足够的磁盘空间存储推理结果
3. **文件路径**：确保所有的 checkpoint 和 jsonl 文件路径都正确
4. **日志文件**：每个任务的详细日志保存在 `{output_dir}_gpu{N}.log`
5. **任务失败**：如果某个任务失败，脚本会继续执行其他任务

## 与 Ver1 的区别

| 特性 | Ver1 | Ver2 |
|------|------|------|
| Checkpoint 数量 | 单个 | 多个 |
| GPU 并行 | 不支持 | 支持 |
| 结果组织 | 按时间戳 | 按 checkpoint 分文件夹 |
| 任务管理 | 顺序执行 | 后台并行执行 |
| GPU 使用 | 固定单个 GPU | 动态分配多个 GPU |

## 故障排查

### 问题：任务一直等待，没有 GPU 可用

**原因**：所有 GPU 都被占用

**解决方案**：
- 检查 GPU 使用情况：`nvidia-smi`
- 增加可用的 GPU 数量
- 等待正在运行的任务完成

### 问题：某些任务失败

**原因**：可能是显存不足、文件不存在等

**解决方案**：
- 查看对应的日志文件：`{output_dir}_gpu{N}.log`
- 检查 checkpoint 和 jsonl 文件是否存在
- 检查 GPU 显存是否足够

### 问题：脚本意外终止

**原因**：可能是系统资源不足或其他错误

**解决方案**：
- 使用 `nohup` 或 `screen`/`tmux` 运行脚本
- 检查系统日志
- 减少并行任务数量（减少可用 GPU 数量）

## 性能建议

1. **并行度**：根据可用 GPU 数量和显存大小调整
2. **任务调度**：脚本会自动等待空闲 GPU，无需手动管理
3. **日志管理**：定期清理旧的日志文件以节省磁盘空间
4. **监控**：使用 `watch nvidia-smi` 实时监控 GPU 使用情况
