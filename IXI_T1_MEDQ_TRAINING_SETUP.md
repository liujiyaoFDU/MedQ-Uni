# IXI T1 MedQ医学图像质量增强训练配置 - 完成总结

## ✅ 完成的工作

### 1. 数据集注册（dataset_info.py）✓
**文件**: `/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL/data/dataset_info.py`

已在第813-824行添加两个新数据集配置：
- `ixi_t1_medq_train`: 58,377个训练样本
- `ixi_t1_medq_test`: 302个测试样本

**数据来源**:
```
图像目录: /inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/images
训练JSONL: /inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/annotation/ixi_t1_sr_4x_train.jsonl
测试JSONL: /inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/annotation/ixi_t1_sr_4x_test.jsonl
```

### 2. 配置文件（YAML）✓
**文件**: `/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL/configs/train_ixi_t1_medq_ver1.yaml`

配置MedicalImageEditingIterableDataset_ver1，主要参数：
- 数据集分组: `ixi_t1_medq`
- 图像尺寸: VAE (512-1024), ViT (224-518)
- 采样权重: 20
- 使用全部数据: num_used_data: [0]

### 3. 训练脚本（Shell）✓
**文件**: `/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL/scripts/training/train_sft_ixi_t1_medq_ver1.sh`

脚本特点：
- ✓ Debug参数默认值（TOTAL_STEPS=50快速验证）
- ✓ 正式训练参数注释（TOTAL_STEPS=2000推荐值）
- ✓ 脚本开头定义所有可自定义变量
- ✓ 支持3个命令行参数：EXP_NAME, NUM_GPUS, MASTER_PORT

### 4. 完整性验证 ✓
已验证：
- ✅ Python语法无误
- ✅ YAML语法正确
- ✅ Bash脚本无误
- ✅ 数据集注册成功
- ✅ 配置文件引用正确
- ✅ 图像目录存在
- ✅ JSONL文件格式正确
- ✅ 图像文件可访问

---

## 📝 使用指南

### 必须配置的变量
在运行前，编辑训练脚本第21行：
```bash
MODEL_PATH="/path/to/pretrained/checkpoint"  # 指定预训练模型路径
```

### Debug模式运行（推荐先验证）
```bash
cd /inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL

# 方法1：使用默认参数
bash scripts/training/train_sft_ixi_t1_medq_ver1.sh

# 方法2：指定实验名称
bash scripts/training/train_sft_ixi_t1_medq_ver1.sh ixi_t1_debug_v1

# 方法3：指定GPU数量（为快速测试，可用1或4卡）
bash scripts/training/train_sft_ixi_t1_medq_ver1.sh ixi_t1_debug_v1 4 23456
```

### 正式训练（修改脚本后）
1. 编辑脚本，取消注释正式参数（第28-30行）：
```bash
TOTAL_STEPS=2000       # 取消注释
SAVE_EVERY=500         # 取消注释
LOG_EVERY=10           # 取消注释
```

2. 同时注释掉Debug参数（第21-23行）

3. 启动8卡训练：
```bash
bash scripts/training/train_sft_ixi_t1_medq_ver1.sh ixi_t1_medq_full_v1 8 23456
```

### 监控训练
```bash
# 监控进程和显存
nvidia-smi

# 查看tensorboard日志
tensorboard --logdir output/ixi_t1_medq_full_v1/tensorboard --port 6006

# 监控输出日志
tail -f output/ixi_t1_medq_full_v1/train.log
```

---

## 🔧 关键参数说明

### 训练参数
| 参数 | Debug值 | 正式值 | 说明 |
|------|--------|--------|------|
| TOTAL_STEPS | 50 | 2000+ | 总训练步数 |
| SAVE_EVERY | 25 | 500 | 保存checkpoint间隔 |
| LOG_EVERY | 1 | 10 | 日志记录间隔 |
| LEARNING_RATE | 1e-5 | 1e-5 | 微调学习率 |

### 损失权重
- **CE_WEIGHT=0.25**: 文本交叉熵损失（权重较小因为任务以图像为主）
- **MSE_WEIGHT=1.0**: 图像重建MSE损失（主要优化目标）

### 模块冻结策略
- freeze_llm=False（微调LLM）
- freeze_vit=True（冻结条件编码器）
- freeze_vae=True（冻结重建器）
- freeze_und=False（训练理解分支）

---

## 📊 数据统计

**训练集**: 58,377个样本
- 任务类型分布：motion_correction, denoising, accelerating_mri
- 图像尺寸：大多数256×256或512×512
- 格式：JSONL（每行一个样本的JSON对象）

**测试集**: 302个样本（用于评估）

**样本结构**:
```json
{
    "main_task_type": "motion_correction|denoising|accelerating_mri",
    "degrade_type": "motion|noise|undersampling",
    "input_img": [{"path": "...", "height": 256, "width": 256}],
    "output_img": [{"path": "...", "height": 256, "width": 256}],
    "message": [
        {"from": "human", "value": "<image>...指令..."},
        {"from": "gpt", "value": "...响应...<image>"}
    ]
}
```

---

## ⚠️ 常见问题

### Q1: 显存不足（OOM）
A: 修改脚本的这些变量：
```bash
EXPECTED_NUM_TOKENS=12000  # 从18000降至12000
MAX_NUM_TOKENS=14000       # 相应调整
```

### Q2: 数据加载很慢
A: 增加num_workers参数，在torchrun后面添加：
```bash
--num_workers 4
```

### Q3: Loss不下降
A: 尝试降低学习率：
```bash
LEARNING_RATE=5e-6
```

### Q4: "找不到数据集"错误
A: 确保：
1. dataset_info.py已正确保存（`python -m py_compile`验证）
2. 数据集名称拼写正确（大小写敏感）
3. YAML配置中的数据集名称与dataset_info.py一致

---

## 📍 关键文件位置

```
/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL/
├── data/
│   └── dataset_info.py                    ← 修改（第813-824行添加）
├── configs/
│   └── train_ixi_t1_medq_ver1.yaml       ← 新建
├── scripts/training/
│   └── train_sft_ixi_t1_medq_ver1.sh     ← 新建
├── train/
│   └── main.py                           ← 训练入口（无需修改）
└── output/
    └── {EXP_NAME}/                       ← 训练输出目录（自动创建）
```

---

## 🎯 后续步骤

### 立即可做：
1. ✓ 设置MODEL_PATH（预训练模型路径）
2. ✓ 运行debug模式验证数据流
3. ✓ 检查loss曲线是否正常下降

### 后续优化：
1. 超参数搜索（学习率、批次大小、损失权重）
2. 实现学习率scheduler（warmup + cosine decay）
3. 混合其他医学图像编辑数据集进行多数据集训练
4. 实现自动评估指标（PSNR, SSIM等）

---

## 📋 快速命令参考

```bash
# 验证配置
python -c "from data.dataset_info import DATASET_INFO; print('ixi_t1_medq_train' in DATASET_INFO['MedicalImageEditingIterableDataset_ver1'])"

# debug训练（50步）
bash scripts/training/train_sft_ixi_t1_medq_ver1.sh debug_test 8 23456

# 查看脚本参数
grep "TOTAL_STEPS\|SAVE_EVERY\|LEARNING_RATE" scripts/training/train_sft_ixi_t1_medq_ver1.sh

# 监控GPU
watch -n 1 nvidia-smi

# 检查输出
ls -lh output/*/checkpoint*.pth | head -5
```

---

**完成时间**: 2025-12-12
**配置状态**: ✅ 完成且通过验证
**准备就绪**: 可随时启动训练
