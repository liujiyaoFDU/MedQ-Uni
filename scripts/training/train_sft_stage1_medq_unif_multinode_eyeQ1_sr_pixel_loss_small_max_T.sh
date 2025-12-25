#!/usr/bin/env bash

# ============================================================================
# Stage1 MedQ Unified Multi-Node Training Script (EyeQ1 + SR Pixel Loss)
# Stage1 MedQ统一多节点训练脚本（EyeQ1 + 像素保真loss）
# ============================================================================
# This script is copied from:
#   scripts/training/train_sft_stage1_medq_unif_multinode_eyeQ1.sh
# Changes:
#   - Use train/main_sr_pixel_loss.py as the training entrypoint
#   - Enable pixel-space fidelity loss by default (for PSNR/SSIM-oriented restoration)
#   - Keep all other settings consistent with the original script

# ============================================================================
# 节点环境检测 / Node Environment Detection
# ============================================================================

if [[ -n "${NODE_COUNT}" && -n "${NODE_RANK}" && -n "${MASTER_ADDR}" ]]; then
    RUNNING_ON_H_CLUSTER=true
    echo "[INFO] Detected H Cluster environment: ${NODE_COUNT} nodes, rank ${NODE_RANK}"
else
    RUNNING_ON_H_CLUSTER=false
    echo "[INFO] Running in standalone mode"
fi

if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    REQUIRED_VARS=("NODE_COUNT" "NODE_RANK" "MASTER_ADDR")
    MISSING_VARS=()

    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -z "${!var}" ]]; then
            MISSING_VARS+=("$var")
        fi
    done

    if [[ ${#MISSING_VARS[@]} -gt 0 ]]; then
        echo "[ERROR] Missing required H cluster variables: ${MISSING_VARS[*]}"
        echo "[ERROR] Please ensure you're running with: rjob -e DISTRIBUTED_JOB=true"
        exit 1
    fi
fi

# ============================================================================
# 可自定义变量（脚本内修改） / Customizable Variables (Modify in Script)
# ============================================================================

cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni

source /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/.venv/bin/activate

SCRIPT_DIR="/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni"

# MODEL_PATH="/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/unimedvl_model_checkpoint_upload"
MODEL_PATH="/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_v1/stage1_medq_2nodes_unif_combined_v1/0024000"

CONFIG_FILE="${SCRIPT_DIR}/configs/train_stage1_medq_unif_trainonly_eyeQ.yaml"

TOTAL_STEPS=24000
SAVE_EVERY=4000
LOG_EVERY=1

LEARNING_RATE=2.5e-6

EXPECTED_NUM_TOKENS=18000
MAX_NUM_TOKENS=20000
MAX_NUM_TOKENS_PER_SAMPLE=18000

CE_WEIGHT=0.25
MSE_WEIGHT=0.3

# Pixel-space fidelity loss (enabled by default for SR/restoration metrics like PSNR/SSIM)
# NOTE: The loss is gated internally to apply only on low-noise timesteps.
PIXEL_LOSS_WEIGHT=50
PIXEL_LOSS_TYPE="l2"

PIXEL_LOSS_MAX_T=0.5  # 增加到 1.0，覆盖几乎所有时间步

EMA_DECAY=0.995

# ============================================================================
# Pixel Loss Debugging (optional)
# ============================================================================
# Enable detailed pixel-loss diagnostics in modeling/bagel/bagel.py.
# Set to 0/empty to disable for long runs.
export PIXEL_LOSS_DEBUG="${PIXEL_LOSS_DEBUG:-0}"
export PIXEL_LOSS_DEBUG_VERBOSE="${PIXEL_LOSS_DEBUG_VERBOSE:-0}"
export PIXEL_LOSS_DEBUG_VERBOSE_MAX="${PIXEL_LOSS_DEBUG_VERBOSE_MAX:-2}"
export PIXEL_LOSS_DEBUG_ABNORMAL_MAX="${PIXEL_LOSS_DEBUG_ABNORMAL_MAX:-5}"

# ============================================================================
# 命令行传入参数（可选） / Command-line Arguments (Optional)
# ============================================================================
EXP_NAME="${1:-stage1_medq_2nodes_unif_eyeQ1_sr_pixel_loss_0_5_max_T}"  # Experiment name
NUM_GPUS="${2:-8}"
MASTER_PORT="${3:-23456}"

if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    NUM_NODES="${NODE_COUNT}"
else
    NUM_NODES="${4:-1}"
fi

# ============================================================================
# NCCL自动配置 / NCCL Auto-Configuration
# ============================================================================

if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    ACTUAL_PROC_PER_NODE="${PROC_PER_NODE:-${NUM_GPUS}}"

    if [[ ${ACTUAL_PROC_PER_NODE} -lt 8 ]]; then
        echo "[NCCL CONFIG] Running auto-config for ${ACTUAL_PROC_PER_NODE} GPUs/node"

        NCCL_CONFIG_OUTPUT=$(curl -s http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export 2>&1)

        if [[ $? -eq 0 ]]; then
            eval "${NCCL_CONFIG_OUTPUT}"
        else
            echo "[ERROR] NCCL auto-configuration failed!"
            echo "${NCCL_CONFIG_OUTPUT}"
            exit 1
        fi
    fi

    export NCCL_DEBUG=WARN
    export NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
fi

# ============================================================================
# 环境准备 / Environment Preparation
# ============================================================================

cd "${SCRIPT_DIR}"

# ============================================================================
# GPU设备配置 / GPU Device Configuration
# ============================================================================

if [[ "${RUNNING_ON_H_CLUSTER}" == true && -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    ACTUAL_NUM_GPUS=${#GPU_ARRAY[@]}

    if [[ ${ACTUAL_NUM_GPUS} -ne ${NUM_GPUS} ]]; then
        echo "[WARNING] NUM_GPUS mismatch: expected ${NUM_GPUS}, got ${ACTUAL_NUM_GPUS}. Using ${ACTUAL_NUM_GPUS}"
        NUM_GPUS=${ACTUAL_NUM_GPUS}
    fi
else
    VISIBLE_GPUS=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
fi

# ============================================================================
# 分布式通信配置 / Distributed Communication Configuration
# ============================================================================

if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    ACTUAL_MASTER_ADDR="${MASTER_ADDR}"
    ACTUAL_NODE_RANK="${NODE_RANK}"
    ACTUAL_NNODES="${NODE_COUNT}"
else
    ACTUAL_MASTER_ADDR="127.0.0.1"
    ACTUAL_NODE_RANK=0
    ACTUAL_NNODES=1
fi

# ============================================================================
# 训练脚本路径 / Training Script Path
# ============================================================================
TRAIN_SCRIPT="${SCRIPT_DIR}/train/main_sr_pixel_loss.py"

# ============================================================================
# 训练信息输出 / Training Information Display
# ============================================================================

echo "============================================================================"
echo "[INFO] Stage1 MedQ Unified Training (EyeQ1 + SR Pixel Loss)"
echo "[CONFIG] Exp: ${EXP_NAME} | Steps: ${TOTAL_STEPS} | LR: ${LEARNING_RATE}"
echo "[CONFIG] Nodes: ${NUM_NODES} | GPUs/node: ${NUM_GPUS} | Total GPUs: $((NUM_NODES * NUM_GPUS))"
echo "[CONFIG] Pixel loss: w=${PIXEL_LOSS_WEIGHT} type=${PIXEL_LOSS_TYPE} max_t=${PIXEL_LOSS_MAX_T}"
if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    echo "[CONFIG] Node rank: ${ACTUAL_NODE_RANK} | Master: ${ACTUAL_MASTER_ADDR}:${MASTER_PORT}"
fi
echo "============================================================================"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_TIMESTAMP=$(date +%s)
echo "[INFO] Training started at ${START_TIME}"
echo "============================================================================"

# ============================================================================
# 启动分布式训练 / Launch Distributed Training
# ============================================================================

torchrun \
  --nnodes="${ACTUAL_NNODES}" \
  --node_rank="${ACTUAL_NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${ACTUAL_MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --dataset_config_file "${CONFIG_FILE}" \
  --data_seed 3432 \
  --max_checkpoints 5 \
  --checkpoint_dir "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/${EXP_NAME}" \
  --model_path "${MODEL_PATH}" \
  --resume_from  "${MODEL_PATH}" \
  --resume_model_only True \
  --resume_model_optimizer False \
  --finetune_from_hf True \
  --finetune_from_ema False \
  --auto_resume True \
  --wandb_name "${EXP_NAME}" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --total_steps "${TOTAL_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY}" \
  --lr "${LEARNING_RATE}" \
  --num_workers 1 \
  --expected_num_tokens "${EXPECTED_NUM_TOKENS}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}" \
  --vit_cond_dropout_prob 0 \
  --vae_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --ce_weight "${CE_WEIGHT}" \
  --mse_weight "${MSE_WEIGHT}" \
  --pixel_loss_weight "${PIXEL_LOSS_WEIGHT}" \
  --pixel_loss_type "${PIXEL_LOSS_TYPE}" \
  --pixel_loss_max_t "${PIXEL_LOSS_MAX_T}" \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --freeze_und False \
  --copy_init_moe True \
  --visual_gen True \
  --visual_und True \
  --ema "${EMA_DECAY}" \
  --num_replicate "${NUM_NODES}" \
  --num_shard "${NUM_GPUS}" \
  --sharding_strategy HYBRID_SHARD \
  --backward_prefetch BACKWARD_PRE \
  --cpu_offload False \
  --enable_tensorboard

# ============================================================================
# 训练完成统计 / Training Completion Statistics
# ============================================================================

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
END_TIMESTAMP=$(date +%s)
ELAPSED_SECONDS=$((END_TIMESTAMP - START_TIMESTAMP))
ELAPSED_HOURS=$((ELAPSED_SECONDS / 3600))
ELAPSED_MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
ELAPSED_SECS=$((ELAPSED_SECONDS % 60))

echo "============================================================================"
echo "[INFO] Training completed!"
echo "[INFO] Started: ${START_TIME} | Ended: ${END_TIME}"
echo "[INFO] Elapsed time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m ${ELAPSED_SECS}s"
if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    echo "[INFO] Node ${ACTUAL_NODE_RANK}/${ACTUAL_NNODES} finished"
fi
echo "============================================================================"
