#!/bin/bash

# 批量测试脚本 ver4 - 命令队列 + GPU 动态调度
# 用法: ./MedQ-Uni_run_batch_test_ver4.sh
# 特性:
#   1. 预生成所有完整的推理命令
#   2. 每个 GPU 维护自己的命令队列
#   3. 每个 GPU 可运行多个并行任务
#   4. 任务完成后自动从队列取下一个命令执行
#   5. 更高效：无需重复解析参数，直接执行命令

cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni
source .venv/bin/activate

# 获取脚本自身的路径(用于后续备份)
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")

# ============================================================
# 配置参数
# ============================================================

# 每个 GPU 上的并行任务数
TASKS_PER_GPU=3
# 可用的 GPU 列表
GPUS=(0 1 2 3)

# 多个 checkpoint 路径列表
CHECKPOINTS=(
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0010000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0008000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0006000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6_pixel_weight_10000/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6_pixel_weight_10000/0002000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0008000"
#  "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0012000"
#     "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0016000"
#     "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0020000"
#     "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6_pixel_weight_10000/0004000"
#     "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6_pixel_weight_10000/0006000"
#     "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6_pixel_weight_10000/0008000"
#     "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6_pixel_weight_10000/0010000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0008000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0012000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0016000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6/0020000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0002000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0006000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0008000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss/0010000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6_pixel_weight_10000/0002000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T_lr_2_5e-6_pixel_weight_10000/0004000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_2_max_T_lr_2_5e-6_pixel_weight_10000/0010000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_2_max_T_lr_2_5e-6_pixel_weight_10000/0008000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_2_max_T_lr_2_5e-6_pixel_weight_10000/0006000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6/0012500"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6/0060000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6/0080000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/0010000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/0030000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/0060000"
# "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/0100000"
"/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000/0140000"
)


# 测试文件列表
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



# 其他配置
IMAGE_ROOT="/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/Dataset/images"
BASE_OUTPUT_DIR="stage1_medq_2nodes_unif_sr_pixel_loss_0_2_max_T_lr_2_5e-6_PIXEL_LOSS_WEIGHT_1000_ver1_part2"
MAX_MEM="130GiB"

# 生成参数
CFG_TEXT_SCALE=4.0
CFG_IMG_SCALE=2.0
NUM_TIMESTEPS=50
TIMESTEP_SHIFT=1.0
SEED=42
NUM_SAMPLES=50

# ============================================================
# 全局环境变量
# ============================================================
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# ============================================================
# 临时目录和队列管理
# ============================================================
TEMP_DIR=$(mktemp -d)
GLOBAL_QUEUE="${TEMP_DIR}/global_queue.txt"
GLOBAL_LOCK="${TEMP_DIR}/global.lock"
STATS_FILE="${TEMP_DIR}/stats.txt"
STATS_LOCK="${TEMP_DIR}/stats.lock"

echo "0 0" > "$STATS_FILE"

cleanup() {
    echo ""
    echo "清理临时文件: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# ============================================================
# 队列操作函数
# ============================================================

# 从全局队列获取下一个任务
pop_task() {
    (
        flock -x 200
        if [ -s "$GLOBAL_QUEUE" ]; then
            head -n 1 "$GLOBAL_QUEUE"
            tail -n +2 "$GLOBAL_QUEUE" > "${GLOBAL_QUEUE}.tmp"
            mv "${GLOBAL_QUEUE}.tmp" "$GLOBAL_QUEUE"
        fi
    ) 200>"$GLOBAL_LOCK"
}

# 更新统计
update_stats() {
    local success=$1
    (
        flock -x 200
        read completed failed < "$STATS_FILE"
        if [ "$success" -eq 1 ]; then
            ((completed++))
        else
            ((failed++))
        fi
        echo "$completed $failed" > "$STATS_FILE"
    ) 200>"$STATS_LOCK"
}

# ============================================================
# GPU Worker 函数
# ============================================================

gpu_worker() {
    local gpu=$1
    local worker_id=$2

    echo "[GPU:$gpu W:$worker_id] 启动"

    while true; do
        # 获取任务行 (格式: JOB_ID|CKPT_NAME|DATASET|CMD)
        local task_line=$(pop_task)

        if [ -z "$task_line" ]; then
            echo "[GPU:$gpu W:$worker_id] 队列空，退出"
            break
        fi

        # 解析任务
        local job_id=$(echo "$task_line" | cut -d'|' -f1)
        local ckpt_name=$(echo "$task_line" | cut -d'|' -f2)
        local dataset=$(echo "$task_line" | cut -d'|' -f3)
        local cmd=$(echo "$task_line" | cut -d'|' -f4-)

        echo "[GPU:$gpu W:$worker_id] Job#$job_id 开始: $dataset ($ckpt_name)"

        # 创建输出目录
        local output_dir="${BASE_OUTPUT_DIR}/${ckpt_name}/${dataset}"
        local log_file="${output_dir}_gpu${gpu}_w${worker_id}.log"
        mkdir -p "${BASE_OUTPUT_DIR}/${ckpt_name}"

        # 执行命令
        local start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu eval "$cmd" > "$log_file" 2>&1
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if [ $exit_code -eq 0 ]; then
            echo "[GPU:$gpu W:$worker_id] Job#$job_id 完成 (${duration}s): $dataset"
            update_stats 1
        else
            echo "[GPU:$gpu W:$worker_id] Job#$job_id 失败 (${duration}s): $dataset [exit=$exit_code]"
            update_stats 0
        fi

        sleep 1
    done
}

# ============================================================
# 主逻辑
# ============================================================

TOTAL_START=$(date +%s)

echo "============================================================"
echo "批量测试 ver4 - 命令队列模式"
echo "============================================================"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "测试集: ${#ANNOTATION_FILES[@]}"
echo "GPUs: ${GPUS[*]}"
echo "每GPU并行: $TASKS_PER_GPU"
echo "总并行度: $((${#GPUS[@]} * TASKS_PER_GPU))"
echo "============================================================"
echo ""

# ============================================================
# 生成所有任务命令并放入队列
# ============================================================

echo "生成任务队列..."
JOB_ID=0

for checkpoint in "${CHECKPOINTS[@]}"; do
    if [ ! -d "$checkpoint" ]; then
        echo "警告: Checkpoint 不存在: $checkpoint"
        continue
    fi

    ckpt_name=$(basename "$(dirname "$checkpoint")")_$(basename "$checkpoint")

    # 备份脚本
    mkdir -p "${BASE_OUTPUT_DIR}/${ckpt_name}"
    cp "$SCRIPT_PATH" "${BASE_OUTPUT_DIR}/${ckpt_name}/" 2>/dev/null

    for annotation_file in "${ANNOTATION_FILES[@]}"; do
        if [ ! -f "$annotation_file" ]; then
            echo "警告: 文件不存在: $annotation_file"
            continue
        fi

        dataset_name=$(basename "$annotation_file" .jsonl)
        output_dir="${BASE_OUTPUT_DIR}/${ckpt_name}/${dataset_name}"

        # 生成完整命令
        cmd="python inference_pipeline/MedQ-Uni_run_batch_test2.py \
--annotation_file \"$annotation_file\" \
--image_root \"$IMAGE_ROOT\" \
--output_dir \"$output_dir\" \
--model_path \"$checkpoint\" \
--target_gpu_device \"0\" \
--max_mem_per_gpu \"$MAX_MEM\" \
--cfg_text_scale $CFG_TEXT_SCALE \
--cfg_img_scale $CFG_IMG_SCALE \
--num_timesteps $NUM_TIMESTEPS \
--timestep_shift $TIMESTEP_SHIFT \
--seed $SEED \
--num_samples $NUM_SAMPLES"

        # 写入队列 (JOB_ID|CKPT_NAME|DATASET|CMD)
        echo "${JOB_ID}|${ckpt_name}|${dataset_name}|${cmd}" >> "$GLOBAL_QUEUE"
        ((JOB_ID++))
    done
done

TOTAL_JOBS=$JOB_ID
echo "✓ 共生成 $TOTAL_JOBS 个任务"
echo ""

if [ $TOTAL_JOBS -eq 0 ]; then
    echo "没有任务需要执行"
    exit 0
fi

# ============================================================
# 启动所有 GPU Workers
# ============================================================

echo "启动 Workers..."
declare -a PIDS=()

for gpu in "${GPUS[@]}"; do
    for ((w=0; w<TASKS_PER_GPU; w++)); do
        gpu_worker "$gpu" "$w" &
        PIDS+=($!)
        sleep 0.5
    done
done

echo "✓ 启动 ${#PIDS[@]} 个 workers"
echo ""
echo "============================================================"
echo "执行中... (可用 tail -f ${BASE_OUTPUT_DIR}/*/*_gpu*.log 查看)"
echo "============================================================"
echo ""

# 等待所有 workers 完成
for pid in "${PIDS[@]}"; do
    wait $pid
done

# ============================================================
# 统计结果
# ============================================================

TOTAL_END=$(date +%s)
DURATION=$((TOTAL_END - TOTAL_START))

read COMPLETED FAILED < "$STATS_FILE"

echo ""
echo "============================================================"
echo "完成!"
echo "============================================================"
echo "总任务: $TOTAL_JOBS"
echo "成功: $COMPLETED"
echo "失败: $FAILED"
printf "耗时: %02d:%02d:%02d\n" $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60))
echo "输出: $BASE_OUTPUT_DIR/"
echo "============================================================"
