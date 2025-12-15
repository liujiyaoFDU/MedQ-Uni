#!/usr/bin/env bash
source /inspire/hdd/global_user/hejunjun-24017/junzhin/.bashrc


SCRIPT_DIR="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/MedQ-Uni"
eval "$(conda shell.bash hook)"
conda activate bagel
cd "${SCRIPT_DIR}"

model_path="/inspire/hdd/global_user/hejunjun-24017/junzhin/projects/medical_unified_project/models_checkpoints/unimedvl_model_checkpoint_upload_gzy_student_examinations"


id="${1:-finetuned_3}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus="${NUM_GPUS:-8}"

TRAIN_SCRIPT="${SCRIPT_DIR}/train/pretrain_unified_navit_simple_ver6.py"
CONFIG_FILE="${SCRIPT_DIR}/configs/finetuned_example.yaml"

echo "[INFO] Starting training with ${num_gpus} GPUs"
echo "[INFO] Model path: ${model_path}"
echo "[INFO] Config file: ${CONFIG_FILE}"

torchrun \
  --nproc_per_node="${num_gpus}" \
  --master_addr=127.0.0.1 \
  --master_port=29503 \
  "${TRAIN_SCRIPT}" \
  --dataset_config_file "${CONFIG_FILE}" \
  --data_seed 3432 \
  --max_checkpoints 1 \
  --checkpoint_dir "${SCRIPT_DIR}/output/${id}" \
  --model_path "${model_path}" \
  --resume_from "${model_path}" \
  --resume_model_only False \
  --resume_model_optimizer True \
  --finetune_from_hf True \
  --finetune_from_ema True \
  --auto_resume True \
  --wandb_name "${id}" \
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
  --num_shard "${num_gpus}" \
  --sharding_strategy HYBRID_SHARD \
  --backward_prefetch BACKWARD_PRE \
  --cpu_offload False \
  --enable_tensorboard  
