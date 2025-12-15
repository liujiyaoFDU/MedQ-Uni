#!/usr/bin/env bash
# ============================================================================
# Loss Weight Sensitivity Experiment 4: Generation-Moderate
# ============================================================================
# Purpose: Test moderate emphasis on generation task
# Loss weights: ce_weight=0.5, mse_weight=1.0 (ratio 1:2)
# Expected: Higher CE loss, good MSE loss
# ============================================================================

source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc


SCRIPT_DIR="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/Uni-MedVL"
eval "$(conda shell.bash hook)"
conda activate bagel
cd "${SCRIPT_DIR}"

# ===== Model Configuration =====
model_path="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/medical_unified_project/models_checkpoints/BAGEL-7B-MoT"

# ===== Experiment Configuration =====
EXPERIMENT_NAME="sensitivity_exp4_ce0.5_mse1.0_1000steps_v2_noclip_lr1e-7"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8

# ===== Loss Weight Configuration =====
CE_WEIGHT=0.5
MSE_WEIGHT=1.0

# ===== File Paths =====
TRAIN_SCRIPT="${SCRIPT_DIR}/train/main_benchmark_v2_noclip.py"
CONFIG_FILE="${SCRIPT_DIR}/configs/benchmark_mot_vqa.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/output/${EXPERIMENT_NAME}"
RESULTS_DIR="${OUTPUT_DIR}/logs"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

echo "============================================================================"
echo "  Sensitivity Experiment 4: Generation-Moderate (ce=0.5, mse=1.0)"
echo "============================================================================"
echo "[INFO] Experiment: ${EXPERIMENT_NAME}"
echo "[INFO] Loss weights: ce_weight=${CE_WEIGHT}, mse_weight=${MSE_WEIGHT} (ratio 1:2)"
echo "[INFO] Config file: ${CONFIG_FILE}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "[INFO] Expected: Higher CE loss, good MSE loss"
echo "============================================================================"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

torchrun \
  --nproc_per_node="${num_gpus}" \
  --master_addr=127.0.0.1 \
  --master_port=29604 \
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
  --total_steps 1000 \
  --save_every 1000 \
  --log_every 1 \
  --warmup_steps 0 \
  --lr 1e-7 \
  --num_workers 1 \
  --expected_num_tokens 9000 \
  --max_num_tokens 10000 \
  --max_num_tokens_per_sample 8000 \
  --vit_cond_dropout_prob 0 \
  --vae_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  # --max_grad_norm 20.0 \
  --ce_weight ${CE_WEIGHT} \
  --mse_weight ${MSE_WEIGHT} \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --freeze_und False \
  --copy_init_moe True \
  --visual_gen True \
  --visual_und True \
  --ema 0.9999 \
  --num_replicate 1 \
  --num_shard "${num_gpus}" \
  --sharding_strategy HYBRID_SHARD \
  --backward_prefetch BACKWARD_PRE \
  --cpu_offload False \
  --max_checkpoints 1 \
  --enable_tensorboard

echo "============================================================================"
echo "[DONE] Experiment 4 completed!"
echo "[INFO] CSV file: ${RESULTS_DIR}/training_metrics.csv"
echo "============================================================================"
