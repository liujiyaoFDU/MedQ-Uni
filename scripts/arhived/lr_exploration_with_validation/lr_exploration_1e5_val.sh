#!/usr/bin/env bash
# ============================================================================
# Learning Rate Exploration with Validation: LR=1e-5
# ============================================================================
# Purpose: Explore learning rate impact with validation loss tracking
# Learning Rate: 1e-5
# Total Steps: 2000, Save Every: 500 steps
# Validation: Every 10 steps with 50 validation batches
# ============================================================================

source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc


SCRIPT_DIR="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL"
eval "$(conda shell.bash hook)"
conda activate bagel
cd "${SCRIPT_DIR}"

# ===== Model Configuration =====
model_path="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/medical_unified_project/models_checkpoints/BAGEL-7B-MoT"

# ===== Experiment Configuration =====
EXPERIMENT_NAME="lr_exploration_1e5_2000steps_val"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8

# ===== File Paths =====
TRAIN_SCRIPT="${SCRIPT_DIR}/train/main_benchmark_val.py"
CONFIG_FILE="${SCRIPT_DIR}/configs/benchmark_mot_vqa_train_split.yaml"
VAL_CONFIG_FILE="${SCRIPT_DIR}/configs/benchmark_mot_vqa_val_split.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/output/${EXPERIMENT_NAME}"
RESULTS_DIR="${OUTPUT_DIR}/logs"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

echo "============================================================================"
echo "  Learning Rate Exploration with Validation: LR=1e-5"
echo "============================================================================"
echo "[INFO] Experiment: ${EXPERIMENT_NAME}"
echo "[INFO] Learning Rate: 1e-5"
echo "[INFO] Training script: ${TRAIN_SCRIPT}"
echo "[INFO] Config file: ${CONFIG_FILE}"
echo "[INFO] Validation config: ${VAL_CONFIG_FILE}"
echo "[INFO] Model path: ${model_path}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "[INFO] Number of GPUs: ${num_gpus}"
echo "[INFO] Key settings:"
echo "       - visual_gen: True (generation branch enabled)"
echo "       - visual_und: True (understanding enabled)"
echo "       - freeze_vae: True (VAE frozen)"
echo "       - total_steps: 2000"
echo "       - save_every: 500"
echo "       - lr: 1e-5"
echo "       - validation: Every 10 steps, 50 batches"
echo "============================================================================"

# Create output directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# ===== Launch Training =====
torchrun \
  --nproc_per_node="${num_gpus}" \
  --master_addr=127.0.0.1 \
  --master_port=29603 \
  --nnodes=1 \
  "${TRAIN_SCRIPT}" \
  --dataset_config_file "${CONFIG_FILE}" \
  --data_seed 42 \
  --results_dir "${RESULTS_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --model_path "${model_path}" \
  --resume_model_only False \
  --resume_model_optimizer True \
  --finetune_from_hf True \
  --finetune_from_ema True \
  --auto_resume True \
  --wandb_name "${EXPERIMENT_NAME}" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --total_steps 2001 \
  --save_every 500 \
  --log_every 1 \
  --warmup_steps 100 \
  --lr 1e-5 \
  --num_workers 1 \
  --expected_num_tokens 8000 \
  --max_num_tokens 10000 \
  --max_num_tokens_per_sample 9000 \
  --vit_cond_dropout_prob 0 \
  --vae_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --ce_weight 1.0 \
  --mse_weight 1.0 \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --freeze_und True \
  --copy_init_moe True \
  --visual_gen True \
  --visual_und True \
  --ema 0.9999 \
  --num_replicate 1 \
  --num_shard "${num_gpus}" \
  --sharding_strategy HYBRID_SHARD \
  --backward_prefetch BACKWARD_PRE \
  --cpu_offload False \
  --max_checkpoints 3 \
  --enable_tensorboard \
  --enable_validation True \
  --val_dataset_config_file "${VAL_CONFIG_FILE}" \
  --val_every 50 \
  --val_steps 20

echo "============================================================================"
echo "[DONE] LR exploration (1e-5) with validation completed!"
echo "[INFO] Training metrics: ${RESULTS_DIR}/training_metrics.csv"
echo "[INFO] Validation metrics: ${RESULTS_DIR}/validation_metrics.csv"
echo "[INFO] TensorBoard logs: ${CHECKPOINT_DIR}/tensorboard/"
echo "============================================================================"
