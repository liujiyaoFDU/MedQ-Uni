#!/usr/bin/env bash
# ============================================================================
# Benchmark Experiment: Baseline (Single-Expert, visual_gen=False)
# ============================================================================
# Purpose: Measure performance of pure understanding model without generation branch
# Expected output: training_metrics.csv in output/benchmark_baseline_vqa_1000steps/logs/
# ============================================================================

source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc


SCRIPT_DIR="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/MedQ-Uni"
eval "$(conda shell.bash hook)"
conda activate bagel
cd "${SCRIPT_DIR}"

# ===== Model Configuration =====
model_path="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/medical_unified_project/models_checkpoints/BAGEL-7B-MoT"

# ===== Experiment Configuration =====
EXPERIMENT_NAME="benchmark_baseline_vqa_1000steps_ver2"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8

# ===== File Paths =====
TRAIN_SCRIPT="${SCRIPT_DIR}/train/main_benchmark.py"  # Using benchmark version
CONFIG_FILE="${SCRIPT_DIR}/configs/benchmark_baseline_vqa.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/output/${EXPERIMENT_NAME}"
RESULTS_DIR="${OUTPUT_DIR}/logs"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

echo "============================================================================"
echo "  Benchmark Experiment: BASELINE (Single-Expert)"
echo "============================================================================"
echo "[INFO] Experiment: ${EXPERIMENT_NAME}"
echo "[INFO] Training script: ${TRAIN_SCRIPT}"
echo "[INFO] Config file: ${CONFIG_FILE}"
echo "[INFO] Model path: ${model_path}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "[INFO] Number of GPUs: ${num_gpus}"
echo "[INFO] Key settings:"
echo "       - visual_gen: False (NO generation branch)"
echo "       - visual_und: True (understanding only)"
echo "       - total_steps: 500"
echo "       - expected_num_tokens: 12000"
echo "============================================================================"

# Create output directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# ===== Launch Training =====
torchrun \
  --nproc_per_node="${num_gpus}" \
  --master_addr=127.0.0.1 \
  --master_port=29600 \
  --nnodes=1 \
  "${TRAIN_SCRIPT}" \
  --dataset_config_file "${CONFIG_FILE}" \
  --data_seed 42 \
  --results_dir "${RESULTS_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --model_path "${model_path}" \
  --resume_from "${model_path}" \
  --resume_model_only False \
  --resume_model_optimizer True \
  --finetune_from_hf True \
  --finetune_from_ema True \
  --auto_resume True \
  --wandb_name "${EXPERIMENT_NAME}" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --total_steps 501 \
  --save_every 500\
  --log_every 1 \
  --warmup_steps 100 \
  --lr 1e-5 \
  --num_workers 1 \
  --expected_num_tokens 12000 \
  --max_num_tokens 14000 \
  --max_num_tokens_per_sample 13000 \
  --vit_cond_dropout_prob 0.05 \
  --text_cond_dropout_prob 0.3 \
  --ce_weight 1.0 \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_und False \
  --copy_init_moe True \
  --visual_gen False \
  --visual_und True \
  --ema 0.9999 \
  --num_replicate 1 \
  --num_shard "${num_gpus}" \
  --sharding_strategy HYBRID_SHARD \
  --backward_prefetch BACKWARD_PRE \
  --cpu_offload False \
  --max_checkpoints 3 \
  --enable_tensorboard

echo "============================================================================"
echo "[DONE] Baseline experiment completed!"
echo "[INFO] Results saved to: ${RESULTS_DIR}"
echo "[INFO] CSV file: ${RESULTS_DIR}/training_metrics.csv"
echo "[INFO] TensorBoard logs: ${CHECKPOINT_DIR}/tensorboard/"
echo "============================================================================"
