#!/usr/bin/env bash

# ============================================================================
# Stage1 MedQ Unified Multi-Node Training Script (H Cluster Optimized)
# Stage1 MedQ统一多节点训练脚本（H集群优化版）
# ============================================================================
# Purpose / 用途:
#   Unified training for medical image quality enhancement with multi-node support
#   支持多节点的医学图像质量增强统一训练
#
# Platform / 平台:
#   H Cluster with RDMA-enabled distributed training
#   支持RDMA的H集群分布式训练平台
#
# Datasets / 数据集:
#   12 datasets with 24 annotation files (train/test splits)
#   12个数据集，24个标注文件（训练/测试集）
#
# Tasks covered / 涵盖任务:
#   - X-ray bone shadow suppression / X光骨骼阴影抑制
#   - PET image quality enhancement (UDPET) / PET图像质量增强
#   - Fundus image restoration (Refuge, Real Fundus, EyeQ) / 眼底图像修复
#   - MRI motion artifact correction / MRI运动伪影校正
#   - MRI super-resolution (IXI T1/T2 4x) / MRI超分辨率
#   - PET low-dose denoising (AMIR) / PET低剂量去噪
#   - MRI super-resolution (AMIR) / MRI超分辨率
#   - CT low-dose denoising (AMIR) / CT低剂量去噪
#   - CT metal artifact reduction (AAPM) / CT金属伪影消除
#
# Base model / 基础模型:
#   MedQ-Uni pretrained checkpoint
#
# ============================================================================
# Usage / 使用方法:
# ============================================================================
#
# Single-node / 单节点训练:
#   bash scripts/training/train_sft_stage1_medq_unif_multinode.sh exp_name 8 23456
#
# Multi-node (H Cluster) / 多节点训练（H集群）:
#   Launch via rjob platform with -e DISTRIBUTED_JOB=true
#   使用rjob平台启动，需添加 -e DISTRIBUTED_JOB=true
#
#   rjob submit -e DISTRIBUTED_JOB=true -P 2 --gpu=8 \
#     --mount=gpfs://gpfs1/quwanying:/mnt/shared-storage-user/quwanying \
#     --charged-group=<your_group> --private-machine=group \
#     --image=<your_image> --host-network=true \
#     --custom-resources rdma/mlnx_shared=8 \
#     -- bash /mnt/shared-storage-user/.../train_sft_stage1_medq_unif_multinode.sh exp_name 8 23456
#
# Arguments / 参数说明:
#   $1: Experiment name / 实验名称 (default: stage1_medq_unif_combined_v1)
#   $2: GPUs per node / 每节点GPU数 (default: 8)
#   $3: Master port / 主节点端口 (default: 23456)
#
# ============================================================================
# Troubleshooting / 故障排查:
# ============================================================================
#
# 1. NCCL timeout / NCCL超时:
#    - Verify network connectivity / 确认网络连接正常
#    - Check NCCL_DEBUG output / 检查NCCL_DEBUG输出
#    - Increase timeout: export NCCL_TIMEOUT=3600
#
# 2. GPU OOM / GPU内存不足:
#    - Reduce token count / 减少token数：调整EXPECTED_NUM_TOKENS
#    - Enable CPU offload / 启用CPU卸载：--cpu_offload True
#
# 3. Inter-node communication failure / 节点间通信失败:
#    - Verify MASTER_ADDR is reachable / 验证MASTER_ADDR可达
#    - Check firewall rules / 检查防火墙规则
#    - Confirm RDMA configuration / 确认RDMA配置正确
#
# 4. Environment variables missing / 环境变量缺失:
#    - Ensure using: rjob -e DISTRIBUTED_JOB=true
#    - Check platform documentation for variable injection
#
# ============================================================================

# ============================================================================
# 节点环境检测 / Node Environment Detection
# ============================================================================
# 说明：检测是否运行在H集群环境，通过检查平台注入的环境变量
# Note: Detect if running on H cluster by checking platform-injected variables

if [[ -n "${NODE_COUNT}" && -n "${NODE_RANK}" && -n "${MASTER_ADDR}" ]]; then
    RUNNING_ON_H_CLUSTER=true
    echo "[INFO] Detected H Cluster environment: ${NODE_COUNT} nodes, rank ${NODE_RANK}"
else
    RUNNING_ON_H_CLUSTER=false
    echo "[INFO] Running in standalone mode"
fi

# 验证必需的环境变量 / Validate required environment variables
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

# 项目路径配置 / Project path configuration
SCRIPT_DIR="/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni"

# 模型路径配置 / Model path configuration
# MODEL_PATH="/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/unimedvl_model_checkpoint_upload"
MODEL_PATH="/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/stage1_medq_2nodes_unif_combined_v1/stage1_medq_2nodes_unif_combined_v1/0024000"


# 配置文件路径 / Config file path
CONFIG_FILE="${SCRIPT_DIR}/configs/train_stage1_medq_unif_trainonly_eyeQ.yaml"

# 训练参数配置 ） / Training parameters
TOTAL_STEPS=2001         # 总训练步数 / Total training steps
SAVE_EVERY=500           
LOG_EVERY=1              # 每步记录日志 / Log every step

# 学习率配置 / Learning rate configuration
LEARNING_RATE=1e-6       # 微调学习率 / Fine-tuning learning rate

# 批次处理配置 / Batch processing configuration
EXPECTED_NUM_TOKENS=25000       # 期望每批次token数 / Expected tokens per batch
MAX_NUM_TOKENS=30000            # 最大批次token数 / Maximum tokens per batch
MAX_NUM_TOKENS_PER_SAMPLE=25000 # 单样本最大token数 / Max tokens per sample

# 损失函数权重配置 / Loss function weights
CE_WEIGHT=0.25           # 交叉熵损失权重（文本token） / Cross-entropy loss weight
MSE_WEIGHT=1.0           # MSE损失权重（图像重建） / MSE loss weight (image reconstruction)

# EMA配置 / EMA configuration
EMA_DECAY=0.995          # 指数移动平均衰减率 / Exponential moving average decay rate

# ============================================================================
# 命令行传入参数（可选） / Command-line Arguments (Optional)
# ============================================================================
EXP_NAME="${1:-stage1_medq_2nodes_unif_combined_eyeQ_ctu_from_stage1_medq_2nodes_unif_combined_v1}"  # 实验名称 / Experiment name
NUM_GPUS="${2:-8}"                     # GPU数量 / Number of GPUs per node
MASTER_PORT="${3:-23456}"              # 主节点端口 / Master port

# 节点数量：优先使用H集群注入的NODE_COUNT
# Node count: Prioritize H cluster's NODE_COUNT
if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    NUM_NODES="${NODE_COUNT}"
else
    NUM_NODES="${4:-1}"
fi

# ============================================================================
# NCCL自动配置 / NCCL Auto-Configuration
# ============================================================================
# H集群要求：当每节点GPU数<8时，必须执行NCCL自动配置脚本
# H cluster requirement: NCCL auto-config mandatory when GPUs per node < 8

if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    # 检查GPU数量 / Check GPU count
    ACTUAL_PROC_PER_NODE="${PROC_PER_NODE:-${NUM_GPUS}}"

    if [[ ${ACTUAL_PROC_PER_NODE} -lt 8 ]]; then
        echo "[NCCL CONFIG] Running auto-config for ${ACTUAL_PROC_PER_NODE} GPUs/node"

        # 执行NCCL自动配置脚本 / Execute NCCL auto-config script
        NCCL_CONFIG_OUTPUT=$(curl -s http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export 2>&1)

        if [[ $? -eq 0 ]]; then
            eval "${NCCL_CONFIG_OUTPUT}"
        else
            echo "[ERROR] NCCL auto-configuration failed!"
            echo "${NCCL_CONFIG_OUTPUT}"
            exit 1
        fi
    fi

    # 设置NCCL调试级别 / Set NCCL debug level (WARN to reduce output)
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
    # H集群已注入CUDA_VISIBLE_DEVICES，不覆盖
    # H cluster has injected CUDA_VISIBLE_DEVICES, do not override

    # 从注入的CUDA_VISIBLE_DEVICES推断GPU数量
    # Infer GPU count from injected CUDA_VISIBLE_DEVICES
    IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
    ACTUAL_NUM_GPUS=${#GPU_ARRAY[@]}

    if [[ ${ACTUAL_NUM_GPUS} -ne ${NUM_GPUS} ]]; then
        echo "[WARNING] NUM_GPUS mismatch: expected ${NUM_GPUS}, got ${ACTUAL_NUM_GPUS}. Using ${ACTUAL_NUM_GPUS}"
        NUM_GPUS=${ACTUAL_NUM_GPUS}
    fi
else
    # 单机模式：按原逻辑生成CUDA_VISIBLE_DEVICES
    # Standalone mode: Generate CUDA_VISIBLE_DEVICES as before
    VISIBLE_GPUS=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
fi

# ============================================================================
# 分布式通信配置 / Distributed Communication Configuration
# ============================================================================

if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    # 使用H集群注入的主节点地址
    # Use H cluster injected master address
    ACTUAL_MASTER_ADDR="${MASTER_ADDR}"
    ACTUAL_NODE_RANK="${NODE_RANK}"
    ACTUAL_NNODES="${NODE_COUNT}"
else
    # 单节点模式
    # Single-node mode
    ACTUAL_MASTER_ADDR="127.0.0.1"
    ACTUAL_NODE_RANK=0
    ACTUAL_NNODES=1
fi

# ============================================================================
# 训练脚本路径 / Training Script Path
# ============================================================================
TRAIN_SCRIPT="${SCRIPT_DIR}/train/main.py"

# ============================================================================
# 训练信息输出 / Training Information Display
# ============================================================================
echo "============================================================================"
echo "[INFO] Stage1 MedQ Unified Training"
echo "[CONFIG] Exp: ${EXP_NAME} | Steps: ${TOTAL_STEPS} | LR: ${LEARNING_RATE}"
echo "[CONFIG] Nodes: ${NUM_NODES} | GPUs/node: ${NUM_GPUS} | Total GPUs: $((NUM_NODES * NUM_GPUS))"
if [[ "${RUNNING_ON_H_CLUSTER}" == true ]]; then
    echo "[CONFIG] Node rank: ${ACTUAL_NODE_RANK} | Master: ${ACTUAL_MASTER_ADDR}:${MASTER_PORT}"
fi
echo "============================================================================"

# 记录开始时间 / Record start time
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
