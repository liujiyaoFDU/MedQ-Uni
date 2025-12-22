#!/bin/bash

# 批量测试脚本
# 用法: ./run_batch_test.sh
# TODO: For 光语启智
# source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc
# conda activate bagel


source .venv/bin/activate

# ============================================================
# 默认参数
# ============================================================
# 多个测试文件列表 - 在这里添加你要测试的所有jsonl文件
ANNOTATION_FILES=(
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AAPM-CT-MAR_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_CT_Low-Dose_CT_denoising_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_MRI_super-resolution_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_PET_low-dose_PET_denoising_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/eyeq_restoration_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/ixi_t1_sr_4x_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/ixi_t2_sr_4x_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/mr_art_motion_correction_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/real_fundus_restoration_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/refuge_restoration_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/UDPET_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/xray-bone-shadow-suppression_test.jsonl"
)

IMAGE_ROOT="/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/images"
BASE_OUTPUT_DIR="MedQ-Uni_results"  # 基础输出目录
MODEL_PATH="/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_v1/stage1_medq_2nodes_unif_combined_v1/0004000"

# GPU 设置 - 使用两张卡的全部显存
TARGET_GPU="0,1"  # 使用 GPU 0 和 1
MAX_MEM="130GiB"   # 增大显存限制（两张卡，每张最多可用显存）

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

# TODO:样本数量控制（-1表示测试全部样本，>0表示只测试前N个样本）
# NUM_SAMPLES=-1  # 默认测试全部样本
NUM_SAMPLES=50  

# 记录总体开始时间
TOTAL_START_TIME=$(date +%s)

echo "============================================================"
echo "开始批量测试，共 ${#ANNOTATION_FILES[@]} 个测试集"
echo "============================================================"
if [ "$NUM_SAMPLES" -eq -1 ]; then
    echo "样本模式: 全部样本 (Full Testing)"
else
    echo "样本模式: 部分样本 (每个测试集前 $NUM_SAMPLES 个样本)"
fi
echo ""

# 循环处理每个测试文件
for i in "${!ANNOTATION_FILES[@]}"; do
    ANNOTATION_FILE="${ANNOTATION_FILES[$i]}"
    
    # 检查文件是否存在
    if [ ! -f "$ANNOTATION_FILE" ]; then
        echo "警告: 文件不存在，跳过: $ANNOTATION_FILE"
        continue
    fi
    
    # 从annotation文件路径中提取数据集名称
    DATASET_NAME=$(basename "$ANNOTATION_FILE" .jsonl)
    
    # 为每个测试集创建独立的输出目录
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"
    
    echo "------------------------------------------------------------"
    echo "测试 $((i+1))/${#ANNOTATION_FILES[@]}: $DATASET_NAME"
    echo "------------------------------------------------------------"
    echo "Annotation文件: $ANNOTATION_FILE"
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    
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
        --seed "$SEED" \
        --num_samples "$NUM_SAMPLES"
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "✓ $DATASET_NAME 测试完成! 结果保存在: $OUTPUT_DIR"
    else
        echo "✗ $DATASET_NAME 测试失败!"
    fi
    echo ""
done

# 计算总耗时
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo "============================================================"
echo "所有测试完成!"
echo "总耗时: ${HOURS}时${MINUTES}分${SECONDS}秒"
echo "结果保存在: $BASE_OUTPUT_DIR/"
echo "============================================================"
