#!/usr/bin/env python3
"""
数据集分割脚本：将JSONL文件按80/20比例分割为训练集和验证集
确保验证集样本不会出现在训练集中

使用方法：
    python split_train_val_datasets.py

功能：
    - 读取指定的3个核心数据集JSONL文件
    - 使用固定随机种子打乱数据
    - 按80/20比例分割为训练集和验证集
    - 验证分割结果无重叠
    - 生成 *_train.jsonl 和 *_val.jsonl 文件
"""

import json
import random
from pathlib import Path

# ============================================================================
# 配置参数
# ============================================================================

ANNOTATION_ROOT = Path("/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation")
RANDOM_SEED = 42
TRAIN_RATIO = 0.8

# 需要分割的数据集
DATASETS = [
    {
        "name": "VQA_RAD",
        "jsonl_path": ANNOTATION_ROOT / "stage1_ver2/VQA_RAD.jsonl",
        "output_dir": ANNOTATION_ROOT / "stage1_ver2"
    },
    {
        "name": "Slake",
        "jsonl_path": ANNOTATION_ROOT / "stage1_ver2/Slake.jsonl",
        "output_dir": ANNOTATION_ROOT / "stage1_ver2"
    },
    {
        "name": "GMAI_Reasoning10K",
        "jsonl_path": ANNOTATION_ROOT / "stage2_part1_ver1/multi_model_cot/GMAI_Reasoning10K.jsonl",
        "output_dir": ANNOTATION_ROOT / "stage2_part1_ver1/multi_model_cot"
    }
]


def split_dataset(dataset_info):
    """
    分割单个数据集

    Args:
        dataset_info: 包含name, jsonl_path, output_dir的字典
    """
    name = dataset_info["name"]
    jsonl_path = dataset_info["jsonl_path"]
    output_dir = dataset_info["output_dir"]

    print(f"\n处理数据集: {name}")
    print(f"  源文件: {jsonl_path}")

    # 检查源文件是否存在
    if not jsonl_path.exists():
        print(f"  ⚠️  警告: 源文件不存在，跳过")
        return False

    # 读取所有行
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"  总样本数: {total}")

    if total == 0:
        print(f"  ⚠️  警告: 文件为空，跳过")
        return False

    # 随机打乱（使用固定种子保证可重复性）
    random.seed(RANDOM_SEED)
    indices = list(range(total))
    random.shuffle(indices)

    # 计算分割点
    train_count = int(total * TRAIN_RATIO)
    val_count = total - train_count

    train_indices = set(indices[:train_count])
    val_indices = set(indices[train_count:])

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写入训练集
    train_path = output_dir / f"{name}_train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for i in sorted(train_indices):  # 排序以保持一致性
            f.write(lines[i])
    print(f"  ✅ 训练集: {train_count} 样本 → {train_path}")

    # 写入验证集
    val_path = output_dir / f"{name}_val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for i in sorted(val_indices):  # 排序以保持一致性
            f.write(lines[i])
    print(f"  ✅ 验证集: {val_count} 样本 → {val_path}")

    # 验证无重叠
    overlap = train_indices & val_indices
    if len(overlap) > 0:
        print(f"  ❌ 错误: 训练集和验证集有 {len(overlap)} 个重叠样本！")
        return False

    print(f"  ✓ 验证通过：训练集和验证集无重叠")

    # 验证总数一致
    if train_count + val_count != total:
        print(f"  ❌ 错误: 分割后样本总数不一致！")
        return False

    return True


def main():
    """主函数"""
    print("=" * 70)
    print("数据集分割工具 - Train/Val Split")
    print("=" * 70)
    print(f"训练集比例: {TRAIN_RATIO * 100:.0f}%")
    print(f"验证集比例: {(1 - TRAIN_RATIO) * 100:.0f}%")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"注释根目录: {ANNOTATION_ROOT}")

    success_count = 0
    total_datasets = len(DATASETS)

    for dataset in DATASETS:
        if split_dataset(dataset):
            success_count += 1

    print("\n" + "=" * 70)
    if success_count == total_datasets:
        print(f"✅ 所有 {total_datasets} 个数据集分割完成！")
    else:
        print(f"⚠️  分割完成：{success_count}/{total_datasets} 个数据集成功")
    print("=" * 70)

    print("\n下一步操作：")
    print("1. 检查生成的 *_train.jsonl 和 *_val.jsonl 文件")
    print("2. 修改 data/dataset_info.py 注册新数据集")
    print("3. 更新训练和验证配置文件")


if __name__ == "__main__":
    main()
