#!/bin/bash

# 批量测试脚本 ver3 - 支持多个 checkpoints 和多 GPU 并行推理
# 用法: ./MedQ-Uni_run_batch_test_ver3.sh
# 特性:
#   1. 支持多个 checkpoint 模型
#   2. 支持多个 jsonl 测试文件
#   3. 结果按 checkpoint 分文件夹存储
#   4. 多 GPU 并行执行（每个任务用单个 GPU）
#   5. [NEW] 支持指定每个 GPU 上运行多个并行任务
#   6. [NEW] 自动任务调度：当一个任务完成后自动从队列获取下一个任务

cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni
source .venv/bin/activate

# 获取脚本自身的路径(用于后续备份)
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")

# ============================================================
# 配置参数
# ============================================================

# [NEW] 每个 GPU 上的并行任务数（根据显存大小和任务内存需求调整）
TASKS_PER_GPU=3

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
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0008000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0006000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0002000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1/0001000"

# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0002000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0001500"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0001000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0000500"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0006000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_eyeQ_v1/0008000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0010000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0008000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0006000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0004000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6/0060000"
)


# 多个测试文件列表 - 在这里添加你要测试的所有jsonl文件
ANNOTATION_FILES=(
    # test
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
BASE_OUTPUT_DIR="stage1_train_pixel_loss_l2_50_eye_0_5_max_T_lr_2_5e-6_ver1"  # 基础输出目录
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
NUM_TIMESTEPS=25
TIMESTEP_SHIFT=1.0
SEED=42

# TODO:样本数量控制（-1表示测试全部样本，>0表示只测试前N个样本）
# NUM_SAMPLES=-1  # 默认测试全部样本
NUM_SAMPLES=500

# ============================================================
# 任务队列和统计变量
# ============================================================

# 临时目录用于任务队列管理
TEMP_DIR=$(mktemp -d)
TASK_QUEUE_FILE="${TEMP_DIR}/task_queue.txt"
TASK_LOCK_FILE="${TEMP_DIR}/task_queue.lock"
STATS_FILE="${TEMP_DIR}/stats.txt"
STATS_LOCK_FILE="${TEMP_DIR}/stats.lock"

# 初始化统计文件
echo "0 0" > "$STATS_FILE"  # completed failed

# 清理函数
cleanup() {
    echo ""
    echo "清理临时文件..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# ============================================================
# 任务管理函数
# ============================================================

# 从队列获取下一个任务（线程安全）
get_next_task() {
    local task=""

    # 使用文件锁确保线程安全
    (
        flock -x 200

        if [ -s "$TASK_QUEUE_FILE" ]; then
            # 读取第一行作为任务
            task=$(head -n 1 "$TASK_QUEUE_FILE")
            # 删除第一行
            tail -n +2 "$TASK_QUEUE_FILE" > "${TASK_QUEUE_FILE}.tmp"
            mv "${TASK_QUEUE_FILE}.tmp" "$TASK_QUEUE_FILE"
            echo "$task"
        fi
    ) 200>"$TASK_LOCK_FILE"
}

# 更新统计信息（线程安全）
update_stats() {
    local success=$1  # 1 for success, 0 for failure

    (
        flock -x 200

        read completed failed < "$STATS_FILE"
        if [ "$success" -eq 1 ]; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
        fi
        echo "$completed $failed" > "$STATS_FILE"
    ) 200>"$STATS_LOCK_FILE"
}

# 获取当前统计信息
get_stats() {
    (
        flock -s 200
        cat "$STATS_FILE"
    ) 200>"$STATS_LOCK_FILE"
}

# 执行推理任务的函数
run_inference() {
    local checkpoint=$1
    local annotation_file=$2
    local gpu=$3
    local worker_id=$4
    local ckpt_name=$5
    local dataset_name=$6

    # 创建输出目录：按 checkpoint 分文件夹
    local output_dir="${BASE_OUTPUT_DIR}/${ckpt_name}/${dataset_name}"

    # 创建日志文件的父目录（如果不存在）
    mkdir -p "${BASE_OUTPUT_DIR}/${ckpt_name}"

    echo "[GPU $gpu Worker $worker_id] 开始推理: $dataset_name (checkpoint: $ckpt_name)"

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
        > "${output_dir}_gpu${gpu}_worker${worker_id}.log" 2>&1

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu Worker $worker_id] ✓ 完成: $dataset_name (checkpoint: $ckpt_name)"
        update_stats 1
    else
        echo "[GPU $gpu Worker $worker_id] ✗ 失败: $dataset_name (checkpoint: $ckpt_name)"
        update_stats 0
    fi

    return $exit_code
}

# Worker 进程：持续从队列获取任务并执行
worker_process() {
    local gpu=$1
    local worker_id=$2
    local checkpoint=$3
    local ckpt_name=$4

    echo "[GPU $gpu Worker $worker_id] 启动，等待任务..."

    while true; do
        # 从队列获取下一个任务
        local task=$(get_next_task)

        if [ -z "$task" ]; then
            # 队列为空，退出 worker
            echo "[GPU $gpu Worker $worker_id] 队列为空，退出"
            break
        fi

        # 解析任务（任务格式：annotation_file）
        local annotation_file="$task"

        # 检查文件是否存在
        if [ ! -f "$annotation_file" ]; then
            echo "[GPU $gpu Worker $worker_id] 警告: 文件不存在，跳过: $annotation_file"
            continue
        fi

        # 提取数据集名称
        local dataset_name=$(basename "$annotation_file" .jsonl)

        # 执行推理
        run_inference "$checkpoint" "$annotation_file" "$gpu" "$worker_id" "$ckpt_name" "$dataset_name"

        # 短暂等待，避免 GPU 资源竞争
        sleep 2
    done

    echo "[GPU $gpu Worker $worker_id] 完成所有任务"
}

# ============================================================
# 主执行逻辑
# ============================================================

TOTAL_START_TIME=$(date +%s)

echo "============================================================"
echo "批量测试配置 (ver3 - 支持每 GPU 多任务并行)"
echo "============================================================"
echo "Checkpoints 数量: ${#CHECKPOINTS[@]}"
echo "测试集数量: ${#ANNOTATION_FILES[@]}"
echo "可用 GPU 数量: ${#GPUS[@]}"
echo "可用 GPUs: ${GPUS[*]}"
echo "每 GPU 并行任务数: $TASKS_PER_GPU"
echo "总并行度: $((${#GPUS[@]} * TASKS_PER_GPU))"
echo "样本模式: $([ "$NUM_SAMPLES" -eq -1 ] && echo '全部样本' || echo "前 $NUM_SAMPLES 个样本")"
echo "============================================================"
echo ""

# 计算总任务数
TOTAL_TASKS=$((${#CHECKPOINTS[@]} * ${#ANNOTATION_FILES[@]}))
echo "总任务数: $TOTAL_TASKS"
echo ""

# 可用 GPU 数量
NUM_GPUS=${#GPUS[@]}

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
    echo "任务分配模式: 动态任务队列，每 GPU $TASKS_PER_GPU 个并行 worker"
    echo ""

    # 复制脚本到checkpoint输出目录以便追溯和复现
    mkdir -p "${BASE_OUTPUT_DIR}/${CKPT_NAME}"
    cp "$SCRIPT_PATH" "${BASE_OUTPUT_DIR}/${CKPT_NAME}/"
    echo "✓ 脚本已备份到: ${BASE_OUTPUT_DIR}/${CKPT_NAME}/$(basename "$SCRIPT_PATH")"
    echo ""

    # 重置统计
    echo "0 0" > "$STATS_FILE"

    # 初始化任务队列（将所有 annotation files 放入队列）
    > "$TASK_QUEUE_FILE"
    for annotation_file in "${ANNOTATION_FILES[@]}"; do
        echo "$annotation_file" >> "$TASK_QUEUE_FILE"
    done
    echo "✓ 任务队列已初始化: ${#ANNOTATION_FILES[@]} 个任务"
    echo ""

    # 启动所有 worker 进程
    declare -a worker_pids=()

    for gpu_idx in "${!GPUS[@]}"; do
        GPU="${GPUS[$gpu_idx]}"

        for ((worker_id=0; worker_id<TASKS_PER_GPU; worker_id++)); do
            echo "[GPU $GPU] 启动 Worker $worker_id..."

            # 在后台启动 worker
            worker_process "$GPU" "$worker_id" "$CHECKPOINT" "$CKPT_NAME" &
            worker_pids+=($!)

            # 稍微错开启动时间，避免同时启动造成资源竞争
            sleep 1
        done
    done

    echo ""
    echo "共启动 ${#worker_pids[@]} 个 worker 进程"
    echo "等待所有任务完成..."
    echo ""

    # 等待所有 worker 完成
    for pid in "${worker_pids[@]}"; do
        wait $pid
    done

    # 读取统计信息
    read completed failed < "$STATS_FILE"

    echo ""
    echo "✓✓ Checkpoint $CKPT_NAME 的所有任务完成"
    echo "   成功: $completed, 失败: $failed"
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

# 最终统计
read COMPLETED_TASKS FAILED_TASKS < "$STATS_FILE"

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
