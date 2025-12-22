#!/usr/bin/env bash

# ============================================================================
# Stage1 MedQ Unified Training Script
# ============================================================================
# Purpose: Unified training for medical image quality enhancement
# Datasets: 12 datasets with 24 annotation files (train/test splits)
# Tasks covered:
#   - X-ray bone shadow suppression
#   - PET image quality enhancement (UDPET)
#   - Fundus image restoration (Refuge, Real Fundus, EyeQ)
#   - MRI motion artifact correction
#   - MRI super-resolution (IXI T1/T2 4x)
#   - PET low-dose denoising (AMIR)
#   - MRI super-resolution (AMIR)
#   - CT low-dose denoising (AMIR)
#   - CT metal artifact reduction (AAPM)
# Base model: MedQ-Uni pretrained checkpoint
#
# Usage:
#   bash scripts/training/train_sft_stage1_medq_unif.sh stage1_medq_unif_1ep 4 2345
#
# Arguments:
#   $1: Experiment name (default: stage1_medq_unif_v1)
#   $2: Number of GPUs (default: 4)
#   $3: Master port (default: 23456)
# ============================================================================

cd /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni

source /mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni/.venv/bin/activate

# 项目路径配置
SCRIPT_DIR="/mnt/shared-storage-user/quwanying/huoshan_wanying/MedQbench/Project/202512_MedQ-UNI/MedQ-Uni"

# 模型路径配置
MODEL_PATH="/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/unimedvl_model_checkpoint_upload"

# 配置文件路径
CONFIG_FILE="${SCRIPT_DIR}/configs/train_stage1_medq_unif_trainonly.yaml"

# 训练参数配置（正式训练时调整）
TOTAL_STEPS=7000         # 总训练步数
SAVE_EVERY=2000          # 每25步保存一次checkpoint
LOG_EVERY=1              # 每步记录日志
 
# 学习率配置
LEARNING_RATE=1e-6       # 微调学习率

# 批次处理配置
# 注意：根据GPU数量调整token数
EXPECTED_NUM_TOKENS=28000       # 期望每批次token数（4卡配置）
MAX_NUM_TOKENS=30000            # 最大批次token数（4卡配置）
MAX_NUM_TOKENS_PER_SAMPLE=28000 # 单样本最大token数（4卡配置）

# 损失函数权重配置
CE_WEIGHT=0.25           # 交叉熵损失权重（文本token）
MSE_WEIGHT=1.0           # MSE损失权重（图像重建）

# EMA配置
EMA_DECAY=0.995          # 指数移动平均衰减率

# <------------------------------------------------------

EXP_NAME="${1:-stage1_medq_unif_combined_v1}"  # 实验名称
NUM_GPUS="${2:-8}"                     # GPU数量，默认4卡
MASTER_PORT="${3:-23456}"              # 主节点端口，默认23456
NUM_NODES="${4:-1}" 
# <-------------------------------------


# ---------- 环境准备 ---------->

cd "${SCRIPT_DIR}"

# 根据GPU数量动态生成CUDA_VISIBLE_DEVICES
# 如果NUM_GPUS=4，则生成 "0,1,2,3"
# 如果NUM_GPUS=8，则生成 "0,1,2,3,4,5,6,7"
VISIBLE_GPUS=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
# <------------------------------


# ---------- 训练脚本路径 ---------->
TRAIN_SCRIPT="${SCRIPT_DIR}/train/main.py"
# <----------------------------------


# ---------- 信息输出 ---------->
echo "============================================================================"
echo "[INFO] Stage1 MedQ Unified Training - Medical Image Quality Enhancement"
echo "============================================================================"
echo "[CONFIG] Experiment name: ${EXP_NAME}"
echo "[CONFIG] Number of GPUs: ${NUM_GPUS}"
echo "[CONFIG] Model path: ${MODEL_PATH}"
echo "[CONFIG] Config file: ${CONFIG_FILE}"
echo "[CONFIG] Total steps: ${TOTAL_STEPS}"
echo "[CONFIG] Learning rate: ${LEARNING_RATE}"
echo "[CONFIG] CE weight: ${CE_WEIGHT}, MSE weight: ${MSE_WEIGHT}"
echo "[CONFIG] Datasets: 12 medical image quality enhancement tasks (24 files)"
echo "[CONFIG] Nodes:  ${NUM_NODES}"

echo "============================================================================"
# <------------------------------


# ---------- 启动分布式训练 ---------->
torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --dataset_config_file "${CONFIG_FILE}" \
  --data_seed 3432 \
  --max_checkpoints 4 \
  --checkpoint_dir "/mnt/shared-storage-user/safevl-share/quwanying/MedQbench/MedQ-UNI/model_checkpoints/training_stage1/${EXP_NAME}" \
  --model_path "${MODEL_PATH}" \
  --resume_from  "${MODEL_PATH}" \
  --resume_model_only False \
  --resume_model_optimizer True \
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

echo "[INFO] Training completed!"
# <-------------------------------------
