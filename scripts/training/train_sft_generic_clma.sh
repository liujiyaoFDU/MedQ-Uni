#!/usr/bin/env bash



# ---------- TODO: customize your own variables --------->
source path/to/user/bashrc
SCRIPT_DIR="path/project/root"
MODEL_PATH="path/to/model/checkpoint"
CONFIG_FILE="${SCRIPT_DIR}/configs/finetuned_example.yaml"
# <------------------------------------------------------

# ---------- INPUT ARGUMENTS ---------->
EXP_NAME="${1:-this_is_test}"
NUM_GPUS="${2:-8}"
MASTER_PORT="${3:-23333}"
# <-------------------------------------


eval "$(conda shell.bash hook)"
conda activate bagel


cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TRAIN_SCRIPT="${SCRIPT_DIR}/train/main.py"


echo "[INFO] Starting training with ${NUM_GPUS} GPUs"
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] Config file: ${CONFIG_FILE}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --dataset_config_file "${CONFIG_FILE}" \
  --data_seed 3432 \
  --max_checkpoints 1 \
  --checkpoint_dir "${SCRIPT_DIR}/output/${EXP_NAME}" \
  --model_path "${MODEL_PATH}" \
  --resume_from "${MODEL_PATH}" \
  --resume_model_only False \
  --resume_model_optimizer True \
  --finetune_from_hf True \
  --finetune_from_ema True \
  --auto_resume True \
  --wandb_name "${EXP_NAME}" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --total_steps 200 \
  --save_every 100 \
  --log_every 1 \
  --lr 1e-5 \
  --num_workers 1 \
  --expected_num_tokens 18000 \
  --max_num_tokens 20000 \
  --max_num_tokens_per_sample 17000 \
  --vit_cond_dropout_prob 0 \
  --vae_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --ce_weight 0.25 \
  --mse_weight 1.0 \
  --freeze_llm False \
  --freeze_vit True \
  --freeze_vae True \
  --freeze_und False \
  --copy_init_moe True \
  --visual_gen True \
  --visual_und True \
  --ema 0.995 \
  --num_replicate 1 \
  --num_shard "${NUM_GPUS}" \
  --sharding_strategy HYBRID_SHARD \
  --backward_prefetch BACKWARD_PRE \
  --cpu_offload False \
  --enable_tensorboard  
