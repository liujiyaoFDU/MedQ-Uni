# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Unified NAVIT Training Script - With Validation Support

基于 main_memory_efficient.py，添加验证集损失计算功能：
- 所有原有功能（weighted loss记录、FSDP、参数冻结等）
- 验证集数据加载（单独YAML配置）
- 定期验证损失计算（可配置频率）
- 验证指标记录（TensorBoard + CSV）

Key Features:
- 验证集支持：定期在验证集上计算损失，不参与反向传播
- FSDP兼容：验证损失跨GPU聚合
- 独立日志：validation_metrics.csv 单独记录验证损失
- 灵活配置：通过命令行参数控制验证频率和批次数
"""

import functools
import glob
import shutil
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import contextlib
from typing import Optional
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time as time_second
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn

timestamp = time_second()
local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(timestamp))

import torch
torch.backends.cudnn.benchmark = False
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae, AutoEncoder
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, fsdp_ema_setup_with_ddp, fsdp_ema_update, fsdp_wrapper_with_ddp
)

# 导入新的参数管理模块
from train.param_manager import (
    freeze_mot_branch,
    create_param_groups_v2,
    print_param_statistics,
    verify_param_groups_integrity,
    save_initial_params,
    validate_param_freezing,
    print_memory_stats,
)


def log_model_parameters(logger, model, language_model, vit_model, vae_model, training_args):
    """Log parameter statistics for each model component."""

    def count_params(model, model_name):
        """Count total, trainable, and frozen parameters."""
        if model is None:
            return 0, 0, 0

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        logger.info(f"{model_name}: Total {total_params/1e6:.1f}M | Trainable {trainable_params/1e6:.1f}M | Frozen {frozen_params/1e6:.1f}M")
        return total_params, trainable_params, frozen_params

    logger.info("=" * 60)
    logger.info("Model Parameter Statistics")
    logger.info("=" * 60)

    llm_total, llm_trainable, _ = count_params(language_model, "Language Model (LLM)")

    vit_total = vit_trainable = 0
    if training_args.visual_und and vit_model is not None:
        if hasattr(model, 'vit_model'):
            vit_total, vit_trainable, _ = count_params(model.vit_model, "Vision Transformer (ViT)")
        else:
            vit_total, vit_trainable, _ = count_params(vit_model, "Vision Transformer (ViT)")

    vae_total = vae_trainable = 0
    if training_args.visual_gen:
        if hasattr(model, 'vae_model') and model.vae_model is not None:
            vae_total, vae_trainable, _ = count_params(model.vae_model, "VAE (Internal)")
        elif vae_model is not None:
            vae_total, vae_trainable, _ = count_params(vae_model, "VAE (External)")

    total_all, trainable_all, frozen_all = count_params(model, "Complete Bagel Model")

    logger.info("-" * 60)
    logger.info(f"Summary: LLM {llm_total/1e6:.1f}M | ViT {vit_total/1e6:.1f}M | VAE {vae_total/1e6:.1f}M | Total {total_all/1e6:.1f}M")
    logger.info(f"Total Trainable Parameters: {trainable_all/1e6:.1f}M ({trainable_all/1e9:.2f}B)")
    logger.info("=" * 60)


@dataclass
class ModelArguments:
    """Model configuration arguments."""
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer_with_hidden_outputs",
        metadata={"help": "Python class name of the load layer to instantiate."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    """Data loading and processing arguments."""
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:

    # Modality switches
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # Bookkeeping & logging
    max_checkpoints: int = field(
        default=5,
        metadata={"help": "Maximum number of checkpoints to keep in the checkpoint directory."}
    )
    results_dir: str = field(
        default="output",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_name: str = field(
        default="med_bagel_project",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    enable_tensorboard: bool = field(
        default=True,
        metadata={"help": "Enable TensorBoard logging alongside W&B."}
    )

    # Reproducibility & resume
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=True,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)."}
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    resume_model_optimizer: bool = field(
        default=True,
        metadata={"help": "Load model weights and optimizer states."}
    )

    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HuggingFace model."}
    )

    # Reporting frequency
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # Optimization & scheduler
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )

    # Component-specific learning rates
    e2e_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Reserved parameter for component-specific learning rates. Currently unused."}
    )
    vae_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Independent learning rate for VAE parameters. If None, uses main lr."}
    )
    vit_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Independent learning rate for ViT parameters. If None, uses main lr."}
    )
    llm_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Independent learning rate for language model parameters. If None, uses main lr."}
    )

    # NEW: MOT分支独立学习率
    mot_und_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Independent learning rate for MOT Understanding expert. If None, uses llm_lr."}
    )
    mot_gen_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Independent learning rate for MOT Generation expert. If None, uses llm_lr."}
    )

    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )

    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )

    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )

    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )

    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )

    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )

    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )

    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )

    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )

    # Distributed training / FSDP
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )

    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )

    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )

    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )

    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # Module freezing (原有参数)
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )

    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )

    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."}
    )

    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers (legacy, for detach mechanism)."}
    )

    # NEW: MOT分支级别冻结控制
    freeze_mot_und: bool = field(
        default=False,
        metadata={"help": "Freeze MOT Understanding expert (all layers). Precise control via optimizer."}
    )
    freeze_mot_gen: bool = field(
        default=False,
        metadata={"help": "Freeze MOT Generation expert (all layers). Precise control via optimizer."}
    )

    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )

    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

    # NEW: 参数验证和调试选项
    print_param_details: bool = field(
        default=True,
        metadata={"help": "Print detailed parameter statistics before training."}
    )

    validate_freezing: bool = field(
        default=False,
        metadata={"help": "Validate frozen parameters don't change during training (adds overhead)."}
    )

    # NEW: 验证集相关参数
    enable_validation: bool = field(
        default=False,
        metadata={"help": "Enable validation set evaluation during training."}
    )
    val_dataset_config_file: str = field(
        default=None,
        metadata={"help": "YAML configuration file for validation dataset. If None, validation is disabled."}
    )
    val_every: int = field(
        default=10,
        metadata={"help": "Run validation every N training steps."}
    )
    val_steps: int = field(
        default=50,
        metadata={"help": "Number of batches to use for each validation run."}
    )


@torch.no_grad()
def validate(fsdp_model, val_loader, training_args, device, logger, actual_vae, val_steps=50):
    """
    在验证集上计算损失（不进行反向传播）

    Args:
        fsdp_model: FSDP包装的模型
        val_loader: 验证数据加载器
        training_args: 训练参数
        device: 设备
        logger: 日志器
        actual_vae: VAE模型（可能是内部或外部）
        val_steps: 验证步数

    Returns:
        dict: 验证损失字典 {'val_ce_loss', 'val_mse_loss', 'val_weighted_ce', 'val_weighted_mse', 'val_total_loss'}
    """
    # 切换到评估模式
    fsdp_model.eval()

    total_ce_loss = 0.0
    total_mse_loss = 0.0
    total_ce_samples = 0
    total_mse_samples = 0

    # 创建验证迭代器
    val_iter = iter(val_loader)

    for step in range(val_steps):
        try:
            data = next(val_iter)
        except StopIteration:
            # IterableDataset 耗尽，重新创建迭代器
            val_iter = iter(val_loader)
            data = next(val_iter)

        data = data.cuda(device).to_dict()

        # 移除不需要的字段
        data.pop('batch_data_indexes', None)
        data.pop('ce_loss_weights', None)

        # VAE编码（如果需要）
        images_to_encode = data.pop('padded_images', None)
        if images_to_encode is not None and isinstance(images_to_encode, torch.Tensor) and images_to_encode.numel() > 0:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                if hasattr(actual_vae, 'vae_encode'):
                    z_encoded = actual_vae.vae_encode(images_to_encode)
                elif hasattr(actual_vae, 'encode'):
                    z_encoded = actual_vae.encode(images_to_encode)
                else:
                    raise AttributeError(f"VAE model {type(actual_vae)} missing encode method")
            data['padded_latent'] = z_encoded

        # 前向传播
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss_dict = fsdp_model(**data)

        # 累积CE损失
        ce = loss_dict.get("ce")
        if ce is not None:
            ce_tokens = len(data.get('ce_loss_indexes', []))
            if ce_tokens > 0:
                total_ce_loss += ce.sum().item()
                total_ce_samples += ce_tokens

        # 累积MSE损失
        if training_args.visual_gen:
            mse = loss_dict.get("mse")
            if mse is not None:
                mse_tokens = len(data.get('mse_loss_indexes', []))
                if mse_tokens > 0:
                    total_mse_loss += mse.mean(dim=-1).sum().item()
                    total_mse_samples += mse_tokens

    # 恢复训练模式
    fsdp_model.train()

    # 跨GPU聚合
    total_ce_tensor = torch.tensor(total_ce_loss, device=device)
    total_mse_tensor = torch.tensor(total_mse_loss, device=device)
    total_ce_samples_tensor = torch.tensor(total_ce_samples, device=device)
    total_mse_samples_tensor = torch.tensor(total_mse_samples, device=device)

    dist.all_reduce(total_ce_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_mse_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_ce_samples_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_mse_samples_tensor, op=dist.ReduceOp.SUM)

    # 计算平均损失
    avg_ce = total_ce_tensor.item() / max(total_ce_samples_tensor.item(), 1)
    avg_mse = total_mse_tensor.item() / max(total_mse_samples_tensor.item(), 1)

    # 计算加权损失
    avg_weighted_ce = avg_ce * training_args.ce_weight
    avg_weighted_mse = avg_mse * training_args.mse_weight
    avg_total = avg_weighted_ce + avg_weighted_mse

    return {
        'val_ce_loss': avg_ce,
        'val_mse_loss': avg_mse,
        'val_weighted_ce': avg_weighted_ce,
        'val_weighted_mse': avg_weighted_mse,
        'val_total_loss': avg_total
    }


def main():
    """Main training function."""
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set default learning rates if not specified
    startup_msgs = []
    if training_args.llm_lr is None:
        training_args.llm_lr = training_args.lr
        if dist.get_rank() == 0:
            startup_msgs.append(f"LLM learning rate not set, using default: {training_args.llm_lr}")
    if training_args.vae_lr is None:
        training_args.vae_lr = training_args.lr
        if dist.get_rank() == 0:
            startup_msgs.append(f"VAE learning rate not set, using default: {training_args.vae_lr}")
    if training_args.vit_lr is None:
        training_args.vit_lr = training_args.lr
        if dist.get_rank() == 0:
            startup_msgs.append(f"ViT learning rate not set, using default: {training_args.vit_lr}")

    # NEW: Set MOT branch learning rates
    if training_args.mot_und_lr is None:
        training_args.mot_und_lr = training_args.llm_lr
        if dist.get_rank() == 0:
            startup_msgs.append(f"MOT Understanding LR not set, using llm_lr: {training_args.mot_und_lr}")
    if training_args.mot_gen_lr is None:
        training_args.mot_gen_lr = training_args.llm_lr
        if dist.get_rank() == 0:
            startup_msgs.append(f"MOT Generation LR not set, using llm_lr: {training_args.mot_gen_lr}")

    # Setup logging
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())

        if training_args.enable_tensorboard:
            tensorboard_dir = os.path.join(training_args.checkpoint_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            writer = None
    else:
        logger = create_logger(None, dist.get_rank())
        writer = None
    dist.barrier()

    if dist.get_rank() == 0:
        for m in startup_msgs:
            logger.info(m)
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # Prepare auto resume logic
    latest_checkpoint = None
    is_resuming_from_interruption = False

    if training_args.auto_resume:
        latest_checkpoint = get_latest_ckpt(training_args.checkpoint_dir)
        if latest_checkpoint is not None:
            is_resuming_from_interruption = True
            logger.info(f"🔄 Auto-resume: Found checkpoint {latest_checkpoint}")
        else:
            logger.info(f"🆕 Auto-resume: No checkpoint found in {training_args.checkpoint_dir}")

    # Determine the actual resume configuration based on the scenario
    if is_resuming_from_interruption:
        resume_from = latest_checkpoint
        resume_model_only = training_args.resume_model_only
        finetune_from_ema = False
        logger.info("📋 Resume mode: CONTINUE TRAINING (from interruption)")
        logger.info("   → Loading full training state (model + optimizer + scheduler)")
        logger.info("   → finetune_from_ema automatically disabled")

    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only

        if resume_from is not None:
            finetune_from_ema = training_args.finetune_from_ema

            if resume_model_only:
                mode = "FINETUNE FROM EMA" if finetune_from_ema else "FINETUNE FROM MODEL"
            else:
                mode = "RESUME FROM EMA" if finetune_from_ema else "RESUME FROM MODEL"
            logger.info(f"📋 Resume mode: {mode}")
            logger.info(f"   → Loading from: {resume_from}")
        else:
            finetune_from_ema = False
            logger.info("📋 Resume mode: FRESH START")

    # Validate checkpoint resume configuration
    if finetune_from_ema and (resume_from is None or not os.path.exists(resume_from)):
        error_msg = (
            "CONFIGURATION ERROR: finetune_from_ema=True requires a valid resume_from path!\n"
            f"   Current resume_from: {resume_from}\n"
            "   To fix this, you need to:\n"
            "   1. Provide a valid checkpoint path: --resume_from /path/to/checkpoint\n"
            "   2. Ensure the checkpoint folder contains ema.safetensors\n"
        )
        logger.error(error_msg)
        raise ValueError("finetune_from_ema=True requires valid resume_from path")

    if finetune_from_ema and resume_from:
        ema_file = os.path.join(resume_from, "ema.safetensors")
        if not os.path.exists(ema_file):
            error_msg = (
                f"❌ EMA FILE NOT FOUND: {ema_file}\n"
                f"   Checkpoint folder: {resume_from}\n"
                f"   Available files: {os.listdir(resume_from) if os.path.exists(resume_from) else 'N/A'}\n"
            )
            logger.error(error_msg)
            raise FileNotFoundError(f"EMA weights file not found: {ema_file}")

    # Log final resume configuration
    logger.info("=== CHECKPOINT RESUME CONFIGURATION ===")
    logger.info(f"auto_resume: {training_args.auto_resume}")
    logger.info(f"resume_from: {resume_from}")
    logger.info(f"resume_model_only: {resume_model_only}")
    logger.info(f"finetune_from_ema: {finetune_from_ema}")
    logger.info(f"resume_model_optimizer: {training_args.resume_model_optimizer}")
    if finetune_from_ema:
        logger.info("🔄 Will load EMA weights (ema.safetensors) into main model for fine-tuning")
    elif resume_from:
        logger.info("🔄 Will load main model weights (model.safetensors)")
    else:
        logger.info("🆕 No checkpoint loading - starting fresh training")
    logger.info("=" * 40)

    # Set seed
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)
    else:
        vit_model = None
        vit_config = None

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors")
            if os.path.exists(os.path.join(model_args.model_path, "ae.safetensors")) else os.path.join(model_args.model_path, "vae_model.safetensors")
        )
    else:
        vae_model = None
        vae_config = None

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
        freeze_vae=training_args.freeze_vae,
    )

    # VAE integration based on freeze_vae
    if training_args.visual_gen and not training_args.freeze_vae:
        model = Bagel(
            language_model,
            vit_model if training_args.visual_und else None,
            config,
            vae_model=vae_model
        )
        vae_model = None
        logger.info("VAE integrated into Bagel model for training")
    else:
        model = Bagel(
            language_model,
            vit_model if training_args.visual_und else None,
            config,
            vae_model=None
        )
        logger.info("VAE kept external (frozen or not needed)")

    log_model_parameters(logger, model, language_model,
                        vit_model if training_args.visual_und else None,
                        vae_model if training_args.visual_gen else None,
                        training_args)

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)

    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # ===== 应用参数冻结策略 =====
    # 1. 原有的组件级冻结
    if training_args.visual_gen:
        if hasattr(model, 'vae_model') and model.vae_model is not None:
            if training_args.freeze_vae:
                for param in model.vae_model.parameters():
                    param.requires_grad = False
                logger.info("Internal VAE parameters frozen")
            else:
                for param in model.vae_model.parameters():
                    param.requires_grad = True
                logger.info("Internal VAE parameters unfrozen for training")
        elif vae_model is not None:
            if training_args.freeze_vae:
                for param in vae_model.parameters():
                    param.requires_grad = False
                logger.info("External VAE parameters frozen")
            else:
                for param in vae_model.parameters():
                    param.requires_grad = True
                logger.info("External VAE parameters unfrozen for training")

    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
        logger.info("LLM parameters frozen (freeze_llm=True)")

    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
        logger.info("ViT parameters frozen (freeze_vit=True)")

    # 2. NEW: MOT分支级别冻结
    if training_args.freeze_mot_und:
        frozen_count = freeze_mot_branch(model, branch='und')
        logger.info(f"🔒 MOT Understanding branch frozen: {frozen_count:,} parameters")

    if training_args.freeze_mot_gen:
        frozen_count = freeze_mot_branch(model, branch='gen')
        logger.info(f"🔒 MOT Generation branch frozen: {frozen_count:,} parameters")

    logger.info("After applying freezing strategy:")
    log_model_parameters(logger, model, language_model,
                        vit_model if training_args.visual_und else None,
                        vae_model if training_args.visual_gen else None,
                        training_args)

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )

    ema_model = deepcopy(model)

    external_vae = vae_model if (training_args.visual_gen and training_args.freeze_vae) else None

    model, ema_model = FSDPCheckpoint.unified_load_checkpoint(
        resume_from, logger, model, ema_model,
        finetune_from_ema=finetune_from_ema,
        external_vae_model=external_vae
    )

    if training_args.visual_gen:
        if hasattr(model, 'vae_model') and model.vae_model is not None:
            vae_params = sum(p.numel() for p in model.vae_model.parameters())
            logger.info(f"Internal VAE: {vae_params/1e6:.2f}M parameters")
        elif external_vae is not None:
            vae_params = sum(p.numel() for p in external_vae.parameters())
            logger.info(f"External VAE: {vae_params/1e6:.2f}M parameters")
        else:
            logger.warning("Visual generation enabled but no VAE model found")

    ema_model = fsdp_ema_setup_with_ddp(ema_model, fsdp_config)

    fsdp_model = fsdp_wrapper_with_ddp(model, fsdp_config)

    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        logger.info("FSDP model structure:")
        logger.info(f"{fsdp_model}")

    # ===== 创建优化器参数组（使用改进版本）=====
    param_groups = create_param_groups_v2(fsdp_model, training_args)

    # NEW: 打印详细参数统计
    if training_args.print_param_details:
        print_param_statistics(fsdp_model, optimizer=None, rank=dist.get_rank())

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0
    )

    # NEW: 验证参数组完整性
    if training_args.print_param_details:
        verify_param_groups_integrity(fsdp_model, optimizer, rank=dist.get_rank())
        # 再次打印，这次包含优化器信息
        print_param_statistics(fsdp_model, optimizer=optimizer, rank=dist.get_rank())
        # 打印初始显存状态
        print_memory_stats(rank=dist.get_rank(), prefix="训练开始前")

    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, resume_model_optimizer=training_args.resume_model_optimizer
        )

    # NEW: 保存初始参数（用于验证冻结是否生效）
    initial_params = None
    if training_args.validate_freezing:
        initial_params = save_initial_params(fsdp_model)
        logger.info(f"Saved {len(initial_params)} frozen parameters for validation")

    # ===== 加载训练数据集 =====
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
        logger.info("=" * 50)
        logger.info("Training dataset configuration:")
        logger.info(f"{dataset_meta}")
        logger.info("=" * 50)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )

    train_dataset.set_epoch(data_args.data_seed)

    if dist.get_rank() == 0:
        logger.info("Creating training DataLoader...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=max(1, min(data_args.num_workers, 2)),
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=max(1, min(data_args.prefetch_factor, 2)),
        timeout=120,
        persistent_workers=True,
    )

    # ===== NEW: 加载验证数据集（如果启用） =====
    val_loader = None
    if training_args.enable_validation and training_args.val_dataset_config_file:
        logger.info("=" * 50)
        logger.info("🔍 Validation enabled - loading validation dataset")
        logger.info("=" * 50)

        with open(training_args.val_dataset_config_file, "r") as stream:
            val_dataset_meta = yaml.safe_load(stream)
            logger.info("Validation dataset configuration:")
            logger.info(f"{val_dataset_meta}")

        val_dataset_config = DataConfig(grouped_datasets=val_dataset_meta)
        if training_args.visual_und:
            val_dataset_config.vit_patch_size = model_args.vit_patch_size
            val_dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        if training_args.visual_gen:
            val_dataset_config.vae_image_downsample = vae_image_downsample
            val_dataset_config.max_latent_size = model_args.max_latent_size
            val_dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
            val_dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
            val_dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        val_dataset = PackedDataset(
            val_dataset_config,
            tokenizer=tokenizer,
            special_tokens=new_token_ids,
            local_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            num_workers=1,  # 验证集使用较少worker
            expected_num_tokens=training_args.expected_num_tokens,
            max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
            max_num_tokens=data_args.max_num_tokens,
            max_buffer_size=data_args.max_buffer_size,
            prefer_buffer_before=data_args.prefer_buffer_before,
            interpolate_pos=model_args.interpolate_pos,
            use_flex=training_args.use_flex,
            data_status=None,  # 验证集不需要恢复状态
        )

        val_dataset.set_epoch(data_args.data_seed + 1000)  # 使用不同seed避免与训练集重叠

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_wrapper(),
            drop_last=True,
            timeout=120,
            persistent_workers=True,
        )

        logger.info(f"✅ Validation DataLoader created - will validate every {training_args.val_every} steps")
        logger.info("=" * 50)

    if dist.get_rank() == 0:
        logger.info(f"DataLoader creation completed - Worker processes: {max(1, min(data_args.num_workers, 2))}")

    torch.distributed.barrier()
    logger.info(f"Rank {dist.get_rank()} DataLoader initialization completed, ready to start training")

    def get_unified_vae_model():
        """Get unified VAE model reference"""
        if hasattr(fsdp_model.module, 'vae_model') and fsdp_model.module.vae_model is not None:
            return fsdp_model.module
        elif vae_model is not None:
            return vae_model
        else:
            return None

    if training_args.visual_gen:
        actual_vae = get_unified_vae_model()

        if hasattr(actual_vae, 'vae_model') and actual_vae.vae_model is not None:
            if training_args.freeze_vae:
                actual_vae.vae_model.eval()
            else:
                actual_vae.vae_model.train()
        elif actual_vae is not None:
            if hasattr(actual_vae, 'to') and callable(getattr(actual_vae, 'to')):
                actual_vae.to(device)

            if training_args.freeze_vae:
                actual_vae.eval()
            else:
                actual_vae.train()

    fsdp_model.train()
    ema_model.eval()

    # BENCHMARK: Reset GPU memory stats before training starts
    torch.cuda.reset_peak_memory_stats(device)
    if dist.get_rank() == 0:
        logger.info("🔍 BENCHMARK MODE: GPU memory tracking initialized")

    start_time = time_second()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")

    actual_vae = get_unified_vae_model()
    if actual_vae is not None and hasattr(actual_vae, 'cuda'):
        actual_vae = actual_vae.cuda(device)

    for curr_step, data in enumerate(train_loader, start=train_step):
        if curr_step >= training_args.total_steps:
            break

        data = data.cuda(device).to_dict()

        data_check_passed = True
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    logger.error(f"Step {curr_step}: NaN detected in {key}")
                    data_check_passed = False

                if key == 'packed_timesteps':
                    if torch.isposinf(value).any():
                        logger.error(f"Step {curr_step}: Positive Inf detected in {key}")
                        data_check_passed = False
                else:
                    if torch.isinf(value).any():
                        logger.error(f"Step {curr_step}: Inf detected in {key}")
                        data_check_passed = False

        if not data_check_passed:
            logger.warning(f"Step {curr_step}: Data integrity check failed")

        torch.distributed.barrier()

        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)

        images_to_encode = data.pop('padded_images', None)
        if images_to_encode is not None and isinstance(images_to_encode, torch.Tensor) and images_to_encode.numel() > 0:
            assert images_to_encode.dim() == 4, f"Expect NCHW format, got {tuple(images_to_encode.shape)}"

            try:
                torch.cuda.synchronize()
                if training_args.freeze_vae:
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                            if hasattr(actual_vae, 'vae_encode'):
                                z_encoded = actual_vae.vae_encode(images_to_encode)
                            elif hasattr(actual_vae, 'encode'):
                                z_encoded = actual_vae.encode(images_to_encode)
                            else:
                                raise AttributeError(f"VAE model {type(actual_vae)} missing encode method")
                        data['padded_latent'] = z_encoded
                else:
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        if hasattr(actual_vae, 'vae_encode'):
                            z_encoded = actual_vae.vae_encode(images_to_encode)
                        elif hasattr(actual_vae, 'encode'):
                            z_encoded = actual_vae.encode(images_to_encode)
                        else:
                            raise AttributeError(f"VAE model {type(actual_vae)} missing encode method")
                    data['padded_latent'] = z_encoded
            except Exception as e:
                logger.error(f"VAE encoding failed: {e}")
                logger.error(f"actual_vae type: {type(actual_vae)}")
                logger.error(f"actual_vae available methods: {[m for m in dir(actual_vae) if not m.startswith('_')]}")
                raise

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss_dict = fsdp_model(**data)

        loss = 0

        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)

            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens

            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
            loss_dict["weighted_ce"] = (ce * training_args.ce_weight).detach()
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)
            loss_dict["weighted_ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data.get('mse_loss_indexes', [])), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss += mse * training_args.mse_weight
            loss_dict["weighted_mse"] = (mse * training_args.mse_weight).detach()
            loss_dict["total_loss"] = loss.detach()
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            loss_dict["weighted_mse"] = torch.tensor(0, device=device)
            loss_dict["total_loss"] = loss.detach()
            total_mse_tokens = torch.tensor(0, device=device)

        optimizer.zero_grad()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Step {curr_step}: Abnormal loss value: {loss}")
            torch.distributed.barrier()

        loss.backward()

        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        # NEW: 验证冻结参数（可选，会增加开销）
        if training_args.validate_freezing and initial_params is not None:
            if curr_step % (training_args.log_every * 10) == 0:  # 每N个log周期验证一次
                if not validate_param_freezing(fsdp_model, initial_params, rank=dist.get_rank()):
                    logger.warning(f"Step {curr_step}: Frozen parameter validation failed!")

        # ===== NEW: 验证集评估 =====
        if val_loader is not None and curr_step > 0 and curr_step % training_args.val_every == 0:
            logger.info(f"🔍 Running validation at step {curr_step}...")
            torch.distributed.barrier()  # 同步所有进程

            val_losses = validate(
                fsdp_model, val_loader, training_args, device,
                logger, actual_vae, training_args.val_steps
            )

            if dist.get_rank() == 0:
                # 打印验证结果
                logger.info(f"[Validation] step={curr_step}, "
                           f"val_ce={val_losses['val_ce_loss']:.4f}, "
                           f"val_mse={val_losses['val_mse_loss']:.4f}, "
                           f"val_weighted_ce={val_losses['val_weighted_ce']:.4f}, "
                           f"val_weighted_mse={val_losses['val_weighted_mse']:.4f}, "
                           f"val_total={val_losses['val_total_loss']:.4f}")

                # TensorBoard记录
                if writer is not None:
                    for key, value in val_losses.items():
                        writer.add_scalar(f'validation/{key}', value, curr_step)

                # CSV记录
                val_csv_path = os.path.join(training_args.results_dir, "validation_metrics.csv")
                write_header = not os.path.exists(val_csv_path)

                try:
                    with open(val_csv_path, 'a') as f:
                        if write_header:
                            f.write("step,val_ce_loss,val_mse_loss,val_weighted_ce,val_weighted_mse,val_total_loss\n")
                        f.write(f"{curr_step},"
                               f"{val_losses['val_ce_loss']:.6f},"
                               f"{val_losses['val_mse_loss']:.6f},"
                               f"{val_losses['val_weighted_ce']:.6f},"
                               f"{val_losses['val_weighted_mse']:.6f},"
                               f"{val_losses['val_total_loss']:.6f}\n")
                except Exception as e:
                    logger.warning(f"Failed to write validation CSV: {e}")

            torch.distributed.barrier()  # 验证完成后同步

        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize()
            end_time = time_second()
            steps_per_sec = training_args.log_every / (end_time - start_time)

            # BENCHMARK: Calculate sec/step
            sec_per_step = (end_time - start_time) / training_args.log_every

            # BENCHMARK: Get peak GPU memory and reset for next interval
            peak_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)

            message = f"(step={curr_step:07d}) "
            log = {}

            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        value = value.mean()

                    if value.device != device:
                        value = value.to(device)

                    avg_loss = torch.tensor(value.item(), device=device)
                else:
                    avg_loss = torch.tensor(float(value), device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                log[key] = avg_loss

            # BENCHMARK: Add performance metrics to message
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Sec/Step: {sec_per_step:.4f}, "
            message += f"Peak Memory [Allocated: {peak_memory_allocated:.2f} GB | Reserved: {peak_memory_reserved:.2f} GB], "
            logger.info(message)

            for i, group in enumerate(optimizer.param_groups):
                group_name = group.get('name', f'group_{i}')
                log[f'lr_{group_name}'] = group['lr']
            log['lr'] = optimizer.param_groups[0]['lr']
            log['total_mse_tokens'] = total_mse_tokens.item()
            log['total_ce_tokens'] = total_ce_tokens.item()
            log['total_norm'] = total_norm.item()
            log['total_samples'] = total_samples.item()
            log['training_mode'] = 'with_validation'

            # BENCHMARK: Add metrics to log dict
            log['sec_per_step'] = sec_per_step
            log['peak_memory_allocated_gb'] = peak_memory_allocated
            log['peak_memory_reserved_gb'] = peak_memory_reserved

            if dist.get_rank() == 0 and writer is not None:
                for key, value in log.items():
                    if isinstance(value, (int, float, torch.Tensor)):
                        scalar_value = value.item() if isinstance(value, torch.Tensor) else value
                        writer.add_scalar(f'train/{key}', scalar_value, curr_step)

            # BENCHMARK: Write to CSV file (only rank 0)
            if dist.get_rank() == 0:
                csv_path = os.path.join(training_args.results_dir, "training_metrics.csv")
                write_header = not os.path.exists(csv_path)

                try:
                    with open(csv_path, 'a') as f:
                        if write_header:
                            f.write("step,ce_loss,mse_loss,weighted_ce_loss,weighted_mse_loss,total_loss,sec_per_step,peak_memory_allocated_gb,peak_memory_reserved_gb,lr\n")

                        ce_loss_val = log.get('ce', 0.0)
                        mse_loss_val = log.get('mse', 0.0)
                        weighted_ce_val = log.get('weighted_ce', 0.0)
                        weighted_mse_val = log.get('weighted_mse', 0.0)
                        total_loss_val = log.get('total_loss', 0.0)
                        lr_val = log['lr']
                        f.write(f"{curr_step},{ce_loss_val:.6f},{mse_loss_val:.6f},"
                                f"{weighted_ce_val:.6f},{weighted_mse_val:.6f},{total_loss_val:.6f},"
                                f"{sec_per_step:.4f},{peak_memory_allocated:.2f},{peak_memory_reserved:.2f},"
                                f"{lr_val:.8f}\n")
                except Exception as e:
                    logger.warning(f"Failed to write CSV: {e}")

            start_time = time_second()

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            logger.info("Saving checkpoint...")
            torch.cuda.empty_cache()

            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)

            actual_save_path = FSDPCheckpoint.unified_save_checkpoint(
                save_path=os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}"),
                train_steps=curr_step,
                model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_status=gather_list,
                logger=logger,
                fsdp_config=fsdp_config,
                tokenizer=tokenizer,
                vae_model=vae_model if training_args.visual_gen else None,
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
            )

            if dist.get_rank() == 0:
                all_dirs = glob.glob(os.path.join(training_args.checkpoint_dir, "*"))
                ckpt_dirs = []

                for d in all_dirs:
                    if os.path.isdir(d):
                        dirname = os.path.basename(d)
                        try:
                            int(dirname)
                            ckpt_dirs.append(d)
                        except ValueError:
                            continue

                ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(os.path.basename(x)))

                while len(ckpt_dirs) > training_args.max_checkpoints:
                    oldest = ckpt_dirs.pop(0)
                    shutil.rmtree(oldest)
                    logger.info(f"Removed old checkpoint: {oldest}")

    if dist.get_rank() == 0:
        logger.info("=" * 50)
        logger.info(f"Training completed: {curr_step} steps")
        logger.info("=" * 50)
        if writer is not None:
            writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
