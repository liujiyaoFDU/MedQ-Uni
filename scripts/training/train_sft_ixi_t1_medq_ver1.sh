#!/usr/bin/env bash

# ============================================================================
# IXI T1 Medical Quality Enhancement Training Script - Version 1
# ============================================================================
# Purpose: Fine-tune model for medical image quality enhancement
# Dataset: IXI T1 (58,377 samples) - Motion/Denoise/Undersampling tasks
# Base model: MedQ-Uni pretrained checkpoint

# bash scripts/training/train_sft_ixi_t1_medq_ver1.sh ixi_t1_medq_1ep_ver1 4 2345

# ============================================================================

# ---------- 可自定义变量（脚本内修改） ---------->
# 用户环境配置
source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc

cd /inspire/hdd/global_user/hejunjun-24017/junzhin/projects/MedQ-Uni

# bash scripts/training/train_sft_ixi_t1_medq_ver1.sh ixi_t1_medq_1ep_ver1 4 2345

# 项目路径配置
SCRIPT_DIR="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/MedQ-Uni"

# 模型路径配置（需要提供预训练checkpoint路径）
MODEL_PATH="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/medical_unified_project/models_checkpoints/unimedvl/unimedvl_model_checkpoint_upload"  # TODO: 指定预训练模型路径

# 配置文件路径
CONFIG_FILE="${SCRIPT_DIR}/configs/train_ixi_t1_medq_ver1.yaml"

# # 训练参数配置（DEBUG模式）
# TOTAL_STEPS=50           # 调小至50步用于debug（原始200）
# SAVE_EVERY=25            # 每25步保存一次checkpoint
# LOG_EVERY=1              # 每步记录日志

# 训练参数配置（正式训练时调整）
TOTAL_STEPS=8001       # 正式训练建议值 20samples， 大约 3000 steps 1 epoch
SAVE_EVERY=500          # 正式训练保存间隔 (改为500步,防止crash后无checkpoint)
LOG_EVERY=1           # 正式训练日志间隔

# 学习率配置
LEARNING_RATE=1e-6      # 微调学习率

# 批次处理配置
# 注意：根据GPU数量调整token数
# - 4卡: 12000-14000 tokens
# - 8卡: 18000-20000 tokens
EXPECTED_NUM_TOKENS=18000      # 期望每批次token数（4卡配置）
MAX_NUM_TOKENS=20000           # 最大批次token数（4卡配置）
MAX_NUM_TOKENS_PER_SAMPLE=18000  # 单样本最大token数（4卡配置）

# 损失函数权重配置
CE_WEIGHT=0.25           # 交叉熵损失权重（文本token）
MSE_WEIGHT=1.0           # MSE损失权重（图像重建）

# EMA配置
EMA_DECAY=0.995          # 指数移动平均衰减率

# <------------------------------------------------------

# ---------- 命令行传入参数（可选） ---------->
EXP_NAME="${1:-ixi_t1_medq_debug_v1}"  # 实验名称，默认debug版本
NUM_GPUS="${2:-4}"                      # GPU数量，默认4卡（改为4张）
MASTER_PORT="${3:-23456}"               # 主节点端口，默认23456
# <-------------------------------------


# ---------- 环境准备 ---------->
eval "$(conda shell.bash hook)"
conda activate bagel

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
echo "[INFO] IXI T1 Medical Quality Enhancement Training"
echo "============================================================================"
echo "[CONFIG] Experiment name: ${EXP_NAME}"
echo "[CONFIG] Number of GPUs: ${NUM_GPUS}"
echo "[CONFIG] Model path: ${MODEL_PATH}"
echo "[CONFIG] Config file: ${CONFIG_FILE}"
echo "[CONFIG] Total steps: ${TOTAL_STEPS}"
echo "[CONFIG] Learning rate: ${LEARNING_RATE}"
echo "[CONFIG] CE weight: ${CE_WEIGHT}, MSE weight: ${MSE_WEIGHT}"
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
  --checkpoint_dir "${SCRIPT_DIR}/output/${EXP_NAME}" \
  --model_path "${MODEL_PATH}" \
  --resume_from "/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/MedQ-Uni/output/ixi_t1_medq_4ep_ver1" \
  --resume_model_only False \
  --resume_model_optimizer True \
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
  --num_replicate 1 \
  --num_shard "${NUM_GPUS}" \
  --sharding_strategy HYBRID_SHARD \
  --backward_prefetch BACKWARD_PRE \
  --cpu_offload False \
  --enable_tensorboard

echo "[INFO] Training completed!"
# <-------------------------------------
