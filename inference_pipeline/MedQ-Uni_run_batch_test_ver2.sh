#!/bin/bash

# 批量测试脚本 - 支持多个 checkpoints 和多 GPU 并行推理
# 用法: ./MedQ-Uni_run_batch_test_ver2.sh
# 特性:
#   1. 支持多个 checkpoint 模型
#   2. 支持多个 jsonl 测试文件
#   3. 结果按 checkpoint 分文件夹存储
#   4. 多 GPU 并行执行（每个任务用单个 GPU）

cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni
source .venv/bin/activate

# 获取脚本自身的路径(用于后续备份)
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")

# ============================================================
# 配置参数
# ============================================================

# 多个 checkpoint 路径列表 - 在这里添加你要测试的所有 checkpoint
CHECKPOINTS=(
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_v1/stage1_medq_2nodes_unif_combined_v1/0020000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_v1/stage1_medq_2nodes_unif_combined_v1/0016000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_v1/stage1_medq_2nodes_unif_combined_v1/0012000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_v1/stage1_medq_2nodes_unif_combined_v1/0008000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_v1/stage1_medq_2nodes_unif_combined_v1/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0002000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0001500"
# "/"mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/"stage1_medq_2nodes_unif_combined_eyeQ_v1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0001000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0000500"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0008000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0006000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0004000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0002000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0001000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0004000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0006000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0008000"
)


# 多个测试文件列表 - 在这里添加你要测试的所有jsonl文件
ANNOTATION_FILES=(
    # test
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AAPM-CT-MAR_test.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_CT_Low-Dose_CT_denoising_test.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_MRI_super-resolution_test.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_PET_low-dose_PET_denoising_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/eyeq_restoration_test.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/ixi_t1_sr_4x_test.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/ixi_t2_sr_4x_test.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/mr_art_motion_correction_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/real_fundus_restoration_test.jsonl"
    "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/refuge_restoration_test.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/UDPET_test.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/xray-bone-shadow-suppression_test.jsonl"

    # # train
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_CT_Low-Dose_CT_denoising_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AAPM-CT-MAR_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_MRI_super-resolution_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/AMIR_PET_low-dose_PET_denoising_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/eyeq_restoration_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/ixi_t1_sr_4x_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/ixi_t2_sr_4x_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/mr_art_motion_correction_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/real_fundus_restoration_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/refuge_restoration_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/UDPET_train.jsonl"
    # "/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/annotation/stage1/xray-bone-shadow-suppression_train.jsonl"
)

# 可用的 GPU 列表（用于并行执行）
GPUS=(
    "0"
    "1" 
    "2"
    "3" 
)

IMAGE_ROOT="/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/images"
BASE_OUTPUT_DIR="stage1_train_50_eye_ctu_stage1_comb_v1"  # 基础输出目录

# 每张卡的显存限制
MAX_MEM="130GiB"

# ============================================================
# 全局环境变量设置
# ============================================================
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# 生成参数
CFG_TEXT_SCALE=4.0
CFG_IMG_SCALE=2.0
NUM_TIMESTEPS=50
TIMESTEP_SHIFT=1.0
SEED=42

# TODO:样本数量控制（-1表示测试全部样本，>0表示只测试前N个样本）
# NUM_SAMPLES=-1  # 默认测试全部样本
NUM_SAMPLES=50

# ============================================================
# 任务管理函数
# ============================================================

# 任务计数器
TOTAL_TASKS=0
COMPLETED_TASKS=0
FAILED_TASKS=0

# 执行推理任务的函数
run_inference() {
    local checkpoint=$1
    local annotation_file=$2
    local gpu=$3
    local ckpt_name=$4
    local dataset_name=$5

    # 创建输出目录：按 checkpoint 分文件夹
    local output_dir="${BASE_OUTPUT_DIR}/${ckpt_name}/${dataset_name}"

    # 创建日志文件的父目录（如果不存在）
    mkdir -p "${BASE_OUTPUT_DIR}/${ckpt_name}"

    echo "[GPU $gpu] 开始推理: $dataset_name (checkpoint: $ckpt_name)"

    # 设置该任务专用的 CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$gpu python inference_pipeline/MedQ-Uni_run_batch_test2.py \
        --annotation_file "$annotation_file" \
        --image_root "$IMAGE_ROOT" \
        --output_dir "$output_dir" \
        --model_path "$checkpoint" \
        --target_gpu_device "0" \
        --max_mem_per_gpu "$MAX_MEM" \
        --cfg_text_scale "$CFG_TEXT_SCALE" \
        --cfg_img_scale "$CFG_IMG_SCALE" \
        --num_timesteps "$NUM_TIMESTEPS" \
        --timestep_shift "$TIMESTEP_SHIFT" \
        --seed "$SEED" \
        --num_samples "$NUM_SAMPLES" \
        > "${output_dir}_gpu${gpu}.log" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu] ✓ 完成: $dataset_name (checkpoint: $ckpt_name)"
        ((COMPLETED_TASKS++))
    else
        echo "[GPU $gpu] ✗ 失败: $dataset_name (checkpoint: $ckpt_name)"
        ((FAILED_TASKS++))
    fi

    return $exit_code
}

# ============================================================
# 主执行逻辑
# ============================================================

TOTAL_START_TIME=$(date +%s)

echo "============================================================"
echo "批量测试配置"
echo "============================================================"
echo "Checkpoints 数量: ${#CHECKPOINTS[@]}"
echo "测试集数量: ${#ANNOTATION_FILES[@]}"
echo "可用 GPU 数量: ${#GPUS[@]}"
echo "可用 GPUs: ${GPUS[*]}"
echo "样本模式: $([ "$NUM_SAMPLES" -eq -1 ] && echo '全部样本' || echo "前 $NUM_SAMPLES 个样本")"
echo "============================================================"
echo ""

# 计算总任务数
TOTAL_TASKS=$((${#CHECKPOINTS[@]} * ${#ANNOTATION_FILES[@]}))
echo "总任务数: $TOTAL_TASKS"
echo ""

# 可用 GPU 数量
NUM_GPUS=${#GPUS[@]}

# 为每个 GPU 创建任务处理函数（顺序执行分配给它的任务）
process_gpu_tasks() {
    local gpu=$1
    local gpu_idx=$2
    local checkpoint=$3
    local ckpt_name=$4

    # 该 GPU 需要处理的任务列表
    local tasks_for_this_gpu=()

    # 按轮询方式分配任务给该 GPU
    for ((i=gpu_idx; i<${#ANNOTATION_FILES[@]}; i+=NUM_GPUS)); do
        tasks_for_this_gpu+=("${ANNOTATION_FILES[$i]}")
    done

    # 该 GPU 的任务数量
    local num_tasks=${#tasks_for_this_gpu[@]}

    if [ $num_tasks -eq 0 ]; then
        echo "[GPU $gpu] 无分配任务"
        return 0
    fi

    echo "[GPU $gpu] 分配到 $num_tasks 个任务，开始顺序执行..."

    # 顺序处理该 GPU 的所有任务
    for ((task_idx=0; task_idx<num_tasks; task_idx++)); do
        local annotation_file="${tasks_for_this_gpu[$task_idx]}"

        # 检查文件是否存在
        if [ ! -f "$annotation_file" ]; then
            echo "[GPU $gpu] 警告: 文件不存在，跳过: $annotation_file"
            continue
        fi

        # 提取数据集名称
        local dataset_name=$(basename "$annotation_file" .jsonl)

        echo "[GPU $gpu] 任务 $((task_idx+1))/$num_tasks: $dataset_name"

        # 执行推理（前台执行，等待完成）
        run_inference "$checkpoint" "$annotation_file" "$gpu" "$ckpt_name" "$dataset_name"

        echo "[GPU $gpu] 任务 $((task_idx+1))/$num_tasks 完成，等待 GPU 清理... (5秒)"
        sleep 5
    done

    echo "[GPU $gpu] ✓ 所有 $num_tasks 个任务完成"
}

# 遍历每个 checkpoint（按批次执行）
for ckpt_idx in "${!CHECKPOINTS[@]}"; do
    CHECKPOINT="${CHECKPOINTS[$ckpt_idx]}"

    # 检查 checkpoint 是否存在
    if [ ! -d "$CHECKPOINT" ]; then
        echo "警告: Checkpoint 不存在，跳过: $CHECKPOINT"
        continue
    fi

    # 从 checkpoint 路径提取名称（使用最后两级目录）
    CKPT_NAME=$(basename "$(dirname "$CHECKPOINT")")_$(basename "$CHECKPOINT")

    echo "============================================================"
    echo "处理 Checkpoint $((ckpt_idx+1))/${#CHECKPOINTS[@]}: $CKPT_NAME"
    echo "============================================================"
    echo "路径: $CHECKPOINT"
    echo "任务分配模式: 每个 GPU 顺序执行其分配的任务"
    echo ""

    # 复制脚本到checkpoint输出目录以便追溯和复现
    mkdir -p "${BASE_OUTPUT_DIR}/${CKPT_NAME}"
    cp "$SCRIPT_PATH" "${BASE_OUTPUT_DIR}/${CKPT_NAME}/"
    echo "✓ 脚本已备份到: ${BASE_OUTPUT_DIR}/${CKPT_NAME}/$(basename "$SCRIPT_PATH")"
    echo ""

    # 为每个 GPU 启动一个后台任务处理函数
    declare -a gpu_pids=()

    for gpu_idx in "${!GPUS[@]}"; do
        GPU="${GPUS[$gpu_idx]}"
        echo "[GPU $GPU] 启动任务处理进程..."

        # 在后台运行该 GPU 的任务处理函数
        process_gpu_tasks "$GPU" "$gpu_idx" "$CHECKPOINT" "$CKPT_NAME" &
        gpu_pids+=($!)

        sleep 2  # 避免同时启动
    done

    # 等待所有 GPU 完成其任务
    echo ""
    echo "等待所有 GPU 完成其任务..."
    for pid in "${gpu_pids[@]}"; do
        wait $pid
    done

    echo ""
    echo "✓✓ Checkpoint $CKPT_NAME 的所有任务完成"
    echo ""

    # Checkpoint 间等待，确保显存完全释放
    if [ $((ckpt_idx + 1)) -lt ${#CHECKPOINTS[@]} ]; then
        echo "切换到下一个 Checkpoint 前等待 GPU 显存清理... (15秒)"
        sleep 15
    fi
done

echo "============================================================"
echo "所有 Checkpoints 处理完成"
echo "============================================================"

# 计算总耗时
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "============================================================"
echo "所有任务完成!"
echo "============================================================"
echo "总任务数: $TOTAL_TASKS"
echo "成功: $COMPLETED_TASKS"
echo "失败: $FAILED_TASKS"
echo "总耗时: ${HOURS}时${MINUTES}分${SECONDS}秒"
echo "结果保存在: $BASE_OUTPUT_DIR/"
echo "============================================================"
