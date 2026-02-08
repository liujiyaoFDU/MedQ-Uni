#!/usr/bin/env bash

# ============================================================================
# Stage1 MedQ Unified Multi-Node Training Script (SR SSIM Loss v3 — DeepSpeed ZeRO-2)
# Stage1 MedQ统一多节点训练脚本（SSIM v3 大规模数据 — DeepSpeed ZeRO-2）
# ============================================================================
# Based on: train_sft_stage1_medq_unif_multinode_sr_ssim_loss_v3.sh (FSDP)
# Changes from FSDP v3:
#   - Uses DeepSpeed ZeRO-2 instead of FSDP HYBRID_SHARD
#   - Removes FSDP-specific args (sharding_strategy, backward_prefetch, etc.)
#   - Adds DeepSpeed args (zero_stage)
#   - Keeps torchrun launcher (DeepSpeed supports torchrun)
#   - All loss weights, model paths, freezing args unchanged

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

MODEL_PATH="/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/unimedvl_model_checkpoint_upload"

CONFIG_FILE="${SCRIPT_DIR}/configs/train_stage1_medq_unif_trainonly.yaml"

# 训练参数配置 / Training parameters (identical to FSDP v3)
TOTAL_STEPS=288000         # 总训练步数 / Total training steps
SAVE_EVERY=5000
LOG_EVERY=1

LEARNING_RATE=2.5e-6

EXPECTED_NUM_TOKENS=13000
MAX_NUM_TOKENS=14000
MAX_NUM_TOKENS_PER_SAMPLE=12000

CE_WEIGHT=0.25
MSE_WEIGHT=1

# Pixel-space fidelity loss: DISABLED (replaced by SSIM)
PIXEL_LOSS_WEIGHT=0
PIXEL_LOSS_TYPE="l2"
PIXEL_LOSS_MAX_T=0.0

# SSIM Loss Configuration: ENABLED
# SSIM measures structural similarity for better perceptual quality.
# Loss = 1 - SSIM, so minimizing drives SSIM toward 1.0.
# NOTE: Three layers of attenuation to prevent SSIM gradient explosion:
#   1. ssim_loss_weight=0.1 (10x reduction from original 1.0)
#   2. ssim_loss_max_t=0.1 (only clean samples where SSIM stats are reliable)
#   3. ssim_grad_scale=0.01 (gradient scaling through VAE decoder)
# Why max_t matters: at t=0.3, 30% noise corrupts SSIM local statistics (sigma),
# causing denominator (sigma_x^2+sigma_y^2+C2) ≈ C2=0.0009 → gradient O(1e6) per-element.
# At t=0.1, 90% signal keeps statistics stable → well-behaved gradients.
SSIM_LOSS_WEIGHT=0.1      # SSIM loss weight (10x reduction from 1.0)
SSIM_LOSS_MAX_T=0.1       # Apply SSIM only when timestep t <= 0.1 (90% signal, 10% noise)
SSIM_WINDOW_SIZE=11       # Gaussian window size for SSIM computation
SSIM_GRAD_SCALE=0.01       # Gradient scaling through VAE decoder (prevents 200-2000x gradient spike)

EMA_DECAY=0.995

# DeepSpeed 配置 / DeepSpeed configuration
ZERO_STAGE=2

# ============================================================================
# Debugging (optional)
# ============================================================================
export PIXEL_LOSS_DEBUG="${PIXEL_LOSS_DEBUG:-0}"
export PIXEL_LOSS_DEBUG_VERBOSE="${PIXEL_LOSS_DEBUG_VERBOSE:-0}"
export PIXEL_LOSS_DEBUG_VERBOSE_MAX="${PIXEL_LOSS_DEBUG_VERBOSE_MAX:-2}"
export PIXEL_LOSS_DEBUG_ABNORMAL_MAX="${PIXEL_LOSS_DEBUG_ABNORMAL_MAX:-5}"
export SSIM_LOSS_DEBUG="${SSIM_LOSS_DEBUG:-0}"

# ============================================================================
# 命令行传入参数（可选） / Command-line Arguments (Optional)
# ============================================================================
EXP_NAME="${1:-stage1_medq_2nodes_unif_sr_ssim_loss_v3_deepspeed}"  # v3: weight=0.1, max_t=0.1, grad_scale=0.01 + DeepSpeed ZeRO-2
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

# CUDA_HOME 自动检测 / Auto-detect CUDA_HOME for DeepSpeed
if [[ -z "${CUDA_HOME}" ]]; then
    # Try common CUDA paths
    for cuda_dir in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.8 /usr/local/cuda-12.4; do
        if [[ -f "${cuda_dir}/bin/nvcc" ]]; then
            export CUDA_HOME="${cuda_dir}"
            break
        fi
    done
    # Fallback: find nvcc in PATH
    if [[ -z "${CUDA_HOME}" ]]; then
        NVCC_PATH=$(which nvcc 2>/dev/null)
        if [[ -n "${NVCC_PATH}" ]]; then
            export CUDA_HOME=$(dirname "$(dirname "${NVCC_PATH}")")
        fi
    fi
    # Fallback: create nvcc shim for DeepSpeed import check
    # (ZeRO-2 doesn't need nvcc at runtime, only at import for op compatibility check)
    if [[ -z "${CUDA_HOME}" ]]; then
        CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        if [[ -n "${CUDA_VER}" ]]; then
            SHIM_DIR="${SCRIPT_DIR}/.cuda_shim"
            mkdir -p "${SHIM_DIR}/bin"
            cat > "${SHIM_DIR}/bin/nvcc" << NVCC_EOF
#!/bin/bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release ${CUDA_VER}, V${CUDA_VER}.0"
NVCC_EOF
            chmod +x "${SHIM_DIR}/bin/nvcc"
            export CUDA_HOME="${SHIM_DIR}"
            echo "[INFO] No CUDA toolkit found, created nvcc shim (PyTorch CUDA ${CUDA_VER})"
        fi
    fi
    if [[ -n "${CUDA_HOME}" ]]; then
        echo "[INFO] CUDA_HOME=${CUDA_HOME}"
    fi
fi

# 检查DeepSpeed安装 / Verify DeepSpeed is installed
python3 -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
if [[ $? -ne 0 ]]; then
    echo "[ERROR] DeepSpeed import failed. See error above."
    exit 1
fi

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
# 训练脚本路径 / Training Script Path (DeepSpeed version)
# ============================================================================
TRAIN_SCRIPT="${SCRIPT_DIR}/train/main_sr_pixel_loss_deepspeed.py"

# ============================================================================
# 训练信息输出 / Training Information Display
# ============================================================================

echo "============================================================================"
echo "[INFO] Stage1 MedQ Unified Training (SR SSIM Loss v3 — DeepSpeed ZeRO-${ZERO_STAGE})"
echo "[CONFIG] Exp: ${EXP_NAME} | Steps: ${TOTAL_STEPS} | LR: ${LEARNING_RATE}"
echo "[CONFIG] Nodes: ${NUM_NODES} | GPUs/node: ${NUM_GPUS} | Total GPUs: $((NUM_NODES * NUM_GPUS))"
echo "[CONFIG] DeepSpeed ZeRO Stage: ${ZERO_STAGE}"
echo "[CONFIG] Pixel loss: w=${PIXEL_LOSS_WEIGHT} (DISABLED)"
echo "[CONFIG] SSIM loss: w=${SSIM_LOSS_WEIGHT} max_t=${SSIM_LOSS_MAX_T} window=${SSIM_WINDOW_SIZE} grad_scale=${SSIM_GRAD_SCALE}"
if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    echo "[CONFIG] Node rank: ${ACTUAL_NODE_RANK} | Master: ${ACTUAL_MASTER_ADDR}:${MASTER_PORT}"
fi
echo "============================================================================"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_TIMESTAMP=$(date +%s)
echo "[INFO] Training started at ${START_TIME}"
echo "============================================================================"

# ============================================================================
# 启动分布式训练 / Launch Distributed Training (torchrun)
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
  --finetune_from_ema True \
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
  --ssim_loss_weight "${SSIM_LOSS_WEIGHT}" \
  --ssim_loss_max_t "${SSIM_LOSS_MAX_T}" \
  --ssim_window_size "${SSIM_WINDOW_SIZE}" \
  --ssim_grad_scale "${SSIM_GRAD_SCALE}" \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --freeze_und False \
  --copy_init_moe True \
  --visual_gen True \
  --visual_und True \
  --ema "${EMA_DECAY}" \
  --zero_stage "${ZERO_STAGE}" \
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
