#!/bin/bash

# 批量测试脚本
# 用法: ./run_batch_test.sh
source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc
conda activate bagel

# ============================================================
# 默认参数
# ============================================================
ANNOTATION_FILE="/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/annotation/ixi_t1_sr_4x_test.jsonl"
IMAGE_ROOT="/inspire/hdd/global_user/hejunjun-24017/junzhin/data/bagel/annotation_medq-Uni/images"
OUTPUT_DIR="MedQ-Uni_results/ixi_t1_medq_4ep_ver1_5000_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL/output/ixi_t1_medq_4ep_ver1/ixi_t1_medq_4ep_ver1/0005000"

# GPU 设置
TARGET_GPU="1"
MAX_MEM="40GiB"

# ============================================================
# 环境变量设置
# ============================================================
export CUDA_VISIBLE_DEVICES=$TARGET_GPU
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

echo "============================================================"
echo "Environment Variables"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo ""

# 生成参数
CFG_TEXT_SCALE=4.0
CFG_IMG_SCALE=2.0
NUM_TIMESTEPS=50
TIMESTEP_SHIFT=1.0
SEED=42

# 运行测试
python MedQ-Uni_run_batch_test.py \
    --annotation_file "$ANNOTATION_FILE" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --target_gpu_device "0" \
    --max_mem_per_gpu "$MAX_MEM" \
    --cfg_text_scale "$CFG_TEXT_SCALE" \
    --cfg_img_scale "$CFG_IMG_SCALE" \
    --num_timesteps "$NUM_TIMESTEPS" \
    --timestep_shift "$TIMESTEP_SHIFT" \
    --seed "$SEED"

echo "测试完成! 结果保存在: $OUTPUT_DIR"
