# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Unified NAVIT Training Script — DeepSpeed ZeRO-2 Version

This script implements unified multimodal training for vision-language models
with DeepSpeed ZeRO-2 distributed training support.

Differences from the FSDP version (main_sr_pixel_loss.py):
- Uses DeepSpeed ZeRO-2 instead of FSDP HYBRID_SHARD
- EMA model is a plain deepcopy (no wrapping needed for ZeRO-2)
- Backward/step handled by ds_engine.backward() / ds_engine.step()
- Gradient clipping handled by DeepSpeed config
- Checkpoint format identical (safetensors) for inference compatibility
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
import deepspeed

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
from train.deepspeed_utils import (
    DeepSpeedCheckpoint,
    generate_deepspeed_config,
    deepspeed_ema_update,
    apply_deepspeed_activation_checkpointing,
    grad_checkpoint_check_fn,
)


def load_tensorboard_run_name_from_checkpoint(checkpoint_path, logger):
    """
    Load TensorBoard run name from checkpoint metadata.
    Returns None if not found (legacy checkpoint without metadata).
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None

    training_args_file = os.path.join(checkpoint_path, "training_args.json")
    if os.path.exists(training_args_file):
        try:
            import json
            with open(training_args_file, 'r') as f:
                data = json.load(f)
                tensorboard_metadata = data.get('tensorboard_metadata', {})
                run_name = tensorboard_metadata.get('run_name')
                if run_name:
                    logger.info(f"Loaded TensorBoard run_name from checkpoint: {run_name}")
                    return run_name
        except Exception as e:
            logger.warning(f"Failed to load TensorBoard metadata from checkpoint: {e}")

    return None


def log_model_parameters(logger, model, language_model, vit_model, vae_model, training_args):
    """Log parameter statistics for each model component."""

    def count_params(model, model_name):
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
        metadata={"help": "When resume_model_only=True, load the EMA weights instead of raw weights."}
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

    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW beta1 coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW beta2 coefficient."}
    )

    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW epsilon for numerical stability."}
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

    pixel_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Optional pixel-space fidelity loss weight (paired restoration only)."}
    )

    pixel_loss_type: str = field(
        default="l1",
        metadata={"help": "Pixel loss type: l1 or l2/mse."}
    )

    pixel_loss_max_t: float = field(
        default=0.3,
        metadata={"help": "Apply pixel loss only when diffusion timestep t <= this value (high SNR)."}
    )

    pixel_loss_paired_only: bool = field(
        default=True,
        metadata={"help": "Deprecated (ignored): kept for backward compatibility."}
    )

    pixel_loss_chunk_size: int = field(
        default=2,
        metadata={"help": "Base chunk size for VAE decode (1-4 recommended)."}
    )

    pixel_loss_adaptive_chunk: bool = field(
        default=True,
        metadata={"help": "Enable adaptive chunk size based on image resolution."}
    )

    pixel_loss_use_v0: bool = field(
        default=False,
        metadata={"help": "Use original (v0) pixel loss implementation without chunking."}
    )

    # SSIM Loss
    ssim_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for SSIM (Structural Similarity) loss."}
    )

    ssim_loss_max_t: float = field(
        default=0.1,
        metadata={"help": "Apply SSIM loss only when diffusion timestep t <= this value."}
    )

    ssim_window_size: int = field(
        default=11,
        metadata={"help": "Window size for SSIM computation (default: 11)."}
    )

    ssim_grad_scale: float = field(
        default=0.1,
        metadata={"help": "Gradient scaling for SSIM through VAE decoder (0.01-0.1 recommended)."}
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

    # DeepSpeed configuration (replaces FSDP args)
    zero_stage: int = field(
        default=2,
        metadata={"help": "DeepSpeed ZeRO stage (2 or 3). Stage 2 recommended for EMA compatibility."}
    )
    ds_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed JSON config file. If None, config is auto-generated."}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for DeepSpeed (set by launcher, do not set manually)."}
    )

    # Module freezing
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
        metadata={"help": "Freeze the visual understanding connector layers."}
    )

    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )

    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )


def main():
    """Main training function with DeepSpeed ZeRO-2."""
    assert torch.cuda.is_available()

    # Initialize distributed with DeepSpeed
    deepspeed.init_distributed(dist_backend="nccl")
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

    # Setup logging
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()

    # Optional: per-rank hang watchdog
    hang_watchdog_secs_env = os.environ.get("TRAIN_HANG_WATCHDOG_SECS", "").strip()
    try:
        hang_watchdog_secs = float(hang_watchdog_secs_env) if hang_watchdog_secs_env else 0.0
    except Exception:
        hang_watchdog_secs = 0.0
    hang_watchdog_enabled = hang_watchdog_secs > 0
    if hang_watchdog_enabled:
        import faulthandler
        import threading

        faulthandler.enable(all_threads=True)

        def _start_step_watchdog(step: int, stage_ref: list):
            timer = threading.Timer(
                hang_watchdog_secs,
                lambda: (
                    logger.error(
                        f"[Hang Watchdog] step={step} rank={int(dist.get_rank())} stage={stage_ref[0]} "
                        f"exceeded {hang_watchdog_secs:.0f}s; dumping traceback"
                    ),
                    faulthandler.dump_traceback(all_threads=True),
                ),
            )
            timer.daemon = True
            timer.start()
            return timer

    # Optional debug: confirm code file paths
    if dist.get_rank() == 0:
        pixel_loss_debug = os.environ.get("PIXEL_LOSS_DEBUG", "").lower() in {"1", "true", "yes", "y"}
        if pixel_loss_debug:
            try:
                import modeling.bagel.bagel as bagel_mod
                logger.info(f"[Pixel Loss Debug] code_paths train={__file__} bagel={bagel_mod.__file__}")
            except Exception as e:
                logger.warning(f"[Pixel Loss Debug] failed to resolve code_paths: {e}")

    if dist.get_rank() == 0:
        for m in startup_msgs:
            logger.info(m)
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # ========================================================================
    # Resume logic (identical to FSDP version)
    # ========================================================================
    is_resuming_from_interruption = False
    resume_from = training_args.resume_from
    resume_model_only = training_args.resume_model_only
    finetune_from_ema = False

    if training_args.auto_resume:
        latest_checkpoint = get_latest_ckpt(training_args.checkpoint_dir)
        if latest_checkpoint is not None:
            is_resuming_from_interruption = True
            resume_from = latest_checkpoint
            resume_model_only = False
            finetune_from_ema = False
            logger.info(f"Auto-resume: Found checkpoint in checkpoint_dir: {latest_checkpoint}")
            logger.info("Resume mode: CONTINUE TRAINING (from interruption)")
        elif training_args.resume_from is not None:
            finetune_from_ema = training_args.finetune_from_ema
            if resume_model_only:
                mode = "FINETUNE FROM EMA" if finetune_from_ema else "FINETUNE FROM MODEL"
            else:
                mode = "RESUME FROM EMA" if finetune_from_ema else "RESUME FROM MODEL"
            logger.info(f"Auto-resume: No checkpoint in {training_args.checkpoint_dir}")
            logger.info(f"Resume mode: {mode} (using resume_from as fallback)")
            logger.info(f"   Loading from: {resume_from}")
        else:
            logger.info(f"Auto-resume: No checkpoint found in {training_args.checkpoint_dir}")
            logger.info("Resume mode: FRESH START")
    elif resume_from is not None:
        finetune_from_ema = training_args.finetune_from_ema
        if resume_model_only:
            mode = "FINETUNE FROM EMA" if finetune_from_ema else "FINETUNE FROM MODEL"
        else:
            mode = "RESUME FROM EMA" if finetune_from_ema else "RESUME FROM MODEL"
        logger.info(f"Resume mode: {mode}")
        logger.info(f"   Loading from: {resume_from}")
    else:
        logger.info("Resume mode: FRESH START")

    # Validate checkpoint resume configuration
    if finetune_from_ema and (resume_from is None or not os.path.exists(resume_from)):
        error_msg = (
            "CONFIGURATION ERROR: finetune_from_ema=True requires a valid resume_from path!\n"
            f"   Current resume_from: {resume_from}\n"
            "   To fix: provide --resume_from /path/to/checkpoint"
        )
        logger.error(error_msg)
        raise ValueError("finetune_from_ema=True requires valid resume_from path")

    if finetune_from_ema and resume_from:
        ema_file = os.path.join(resume_from, "ema.safetensors")
        if not os.path.exists(ema_file):
            error_msg = (
                f"EMA FILE NOT FOUND: {ema_file}\n"
                f"   Checkpoint folder: {resume_from}\n"
                f"   Available files: {os.listdir(resume_from) if os.path.exists(resume_from) else 'N/A'}"
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
    logger.info("=" * 40)

    # Set seed
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # ========================================================================
    # TensorBoard initialization
    # ========================================================================
    tensorboard_dir = None
    if dist.get_rank() == 0 and training_args.enable_tensorboard:
        tensorboard_run_name = None

        if is_resuming_from_interruption:
            tensorboard_run_name = load_tensorboard_run_name_from_checkpoint(resume_from, logger)
            if tensorboard_run_name is None:
                tensorboard_dir = os.path.join(training_args.checkpoint_dir, "tensorboard")
                logger.info("Legacy checkpoint detected, using root TensorBoard directory")
            else:
                tensorboard_dir = os.path.join(training_args.checkpoint_dir, "tensorboard", tensorboard_run_name)
                logger.info(f"Resuming TensorBoard run: {tensorboard_run_name}")
        elif resume_from is not None:
            ckpt_step = os.path.basename(resume_from)
            tensorboard_run_name = f"{local_time}_from_{ckpt_step}"
            tensorboard_dir = os.path.join(training_args.checkpoint_dir, "tensorboard", tensorboard_run_name)
            logger.info(f"Starting new experiment from checkpoint, TensorBoard run: {tensorboard_run_name}")
        else:
            tensorboard_run_name = local_time
            tensorboard_dir = os.path.join(training_args.checkpoint_dir, "tensorboard", tensorboard_run_name)
            logger.info(f"Starting fresh training, TensorBoard run: {tensorboard_run_name}")

        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs: {tensorboard_dir}")
    else:
        writer = None
        tensorboard_run_name = None

    # ========================================================================
    # Model setup (identical to FSDP version)
    # ========================================================================
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
    if training_args.visual_gen and (not training_args.freeze_vae or training_args.pixel_loss_weight > 0 or training_args.ssim_loss_weight > 0):
        model = Bagel(
            language_model,
            vit_model if training_args.visual_und else None,
            config,
            vae_model=vae_model
        )
        vae_model = None
        if training_args.freeze_vae:
            logger.info("VAE integrated into Bagel model (frozen) for pixel loss computation")
        else:
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

    # ========================================================================
    # Freezing strategy (identical to FSDP version)
    # ========================================================================
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
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    logger.info("After applying freezing strategy:")
    log_model_parameters(logger, model, language_model,
                        vit_model if training_args.visual_und else None,
                        vae_model if training_args.visual_gen else None,
                        training_args)

    # ========================================================================
    # EMA model — simple deepcopy (no FSDP/DDP wrapping needed for ZeRO-2)
    # ========================================================================
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False

    # Load model/EMA weights BEFORE DeepSpeed initialization
    external_vae = vae_model if (training_args.visual_gen and training_args.freeze_vae) else None
    model, ema_model = DeepSpeedCheckpoint.load_model_checkpoint(
        resume_from, logger, model, ema_model,
        finetune_from_ema=finetune_from_ema,
        external_vae_model=external_vae,
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

    # Move EMA model to device
    ema_model = ema_model.to(device).to(torch.bfloat16)

    # ========================================================================
    # Optimizer with per-component learning rates
    # ========================================================================
    def create_param_groups(model, training_args):
        assigned_params = set()
        param_groups = []

        if hasattr(model, "vae_model") and model.vae_model is not None:
            vae_params = [p for p in model.vae_model.parameters() if p.requires_grad]
            if vae_params:
                param_groups.append({
                    'params': vae_params,
                    'lr': getattr(training_args, 'vae_lr', training_args.lr),
                    'name': 'vae_model'
                })
                assigned_params.update(id(p) for p in vae_params)

        if hasattr(model, "vit_model") and model.vit_model is not None:
            vit_params = [p for p in model.vit_model.parameters() if p.requires_grad]
            if vit_params:
                param_groups.append({
                    'params': vit_params,
                    'lr': getattr(training_args, 'vit_lr', training_args.lr),
                    'name': 'vit_model'
                })
                assigned_params.update(id(p) for p in vit_params)

        if hasattr(model, "language_model") and model.language_model is not None:
            llm_params = [p for p in model.language_model.parameters() if p.requires_grad and id(p) not in assigned_params]
            if llm_params:
                param_groups.append({
                    'params': llm_params,
                    'lr': getattr(training_args, 'llm_lr', training_args.lr),
                    'name': 'language_model'
                })
                assigned_params.update(id(p) for p in llm_params)

        others_params = [
            p for p in model.parameters() if p.requires_grad and id(p) not in assigned_params
        ]
        if others_params:
            param_groups.append({
                'params': others_params,
                'lr': getattr(training_args, 'lr', 1e-5),
                'name': 'others'
            })

        return param_groups

    param_groups = create_param_groups(model, training_args)

    if dist.get_rank() == 0:
        logger.info("=" * 60)
        logger.info("Optimizer parameter group configuration")
        logger.info("=" * 60)
        total_params = 0
        for i, group in enumerate(param_groups):
            group_name = group.get('name', f'group_{i}')
            group_lr = group.get('lr', 'default')
            param_count = sum(p.numel() for p in group['params'])
            total_params += param_count
            logger.info(f"{group_name}: {param_count:,} parameters ({param_count/1e6:.1f}M), learning_rate={group_lr}")
        logger.info(f"Total trainable parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        logger.info("=" * 60)

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0
    )

    # Scheduler
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
        raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")

    # ========================================================================
    # DeepSpeed initialization
    # ========================================================================
    if training_args.ds_config_file and os.path.exists(training_args.ds_config_file):
        import json
        with open(training_args.ds_config_file, 'r') as f:
            ds_config = json.load(f)
        logger.info(f"Loaded DeepSpeed config from {training_args.ds_config_file}")
    else:
        ds_config = generate_deepspeed_config(training_args)
        logger.info(f"Auto-generated DeepSpeed config: ZeRO stage {training_args.zero_stage}")

    ds_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        lr_scheduler=scheduler,
    )
    logger.info("DeepSpeed engine initialized")

    # Activation checkpointing
    apply_deepspeed_activation_checkpointing(ds_engine.module, grad_checkpoint_check_fn)

    if dist.get_rank() == 0:
        logger.info("DeepSpeed model structure (top-level):")
        for name, child in ds_engine.module.named_children():
            logger.info(f"  {name}: {type(child).__name__}")

    # ========================================================================
    # Load optimizer/scheduler state (after DeepSpeed init)
    # ========================================================================
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        ds_engine, scheduler, train_step, data_status = DeepSpeedCheckpoint.load_train_state(
            resume_from, ds_engine, scheduler, logger,
            resume_model_optimizer=training_args.resume_model_optimizer,
        )

    # ========================================================================
    # Dataset and DataLoader (identical to FSDP version)
    # ========================================================================
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
        logger.info("=" * 50)
        logger.info("Dataset configuration information:")
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
        logger.info("Creating DataLoader...")

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

    if dist.get_rank() == 0:
        logger.info(f"DataLoader creation completed - Worker processes: {max(1, min(data_args.num_workers, 2))}")

    torch.distributed.barrier()
    logger.info(f"Rank {dist.get_rank()} DataLoader initialization completed, ready to start training")

    # ========================================================================
    # VAE setup
    # ========================================================================
    def get_unified_vae_model():
        """Get unified VAE model reference (from DeepSpeed engine or external)."""
        if hasattr(ds_engine.module, 'vae_model') and ds_engine.module.vae_model is not None:
            return ds_engine.module.vae_model
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

    ds_engine.train()
    ema_model.eval()

    start_time = time_second()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")

    actual_vae = get_unified_vae_model()
    if actual_vae is not None and hasattr(actual_vae, 'cuda'):
        actual_vae = actual_vae.cuda(device)

    # ========================================================================
    # Training loop
    # ========================================================================
    for curr_step, data in enumerate(train_loader, start=train_step):
        if curr_step >= training_args.total_steps:
            break

        stage_ref = ["data_to_cuda"]
        watchdog_timer = None
        if hang_watchdog_enabled:
            watchdog_timer = _start_step_watchdog(curr_step, stage_ref)

        data = data.cuda(device).to_dict()

        # Data integrity check
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

        step_barrier = os.environ.get("TRAIN_STEP_BARRIER", "").lower() in {"1", "true", "yes", "y"}
        if step_barrier:
            torch.distributed.barrier()

        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)

        # ---- VAE encode ----
        images_to_encode = data.get('padded_images', None)
        if images_to_encode is not None and isinstance(images_to_encode, torch.Tensor) and images_to_encode.numel() > 0:
            assert images_to_encode.dim() == 4, f"Expect NCHW format, got {tuple(images_to_encode.shape)}"

            try:
                stage_ref[0] = "vae_encode"
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

        # Pixel loss debug
        pixel_loss_debug = os.environ.get("PIXEL_LOSS_DEBUG", "").lower() in {"1", "true", "yes", "y"}
        if pixel_loss_debug and training_args.pixel_loss_weight > 0 and dist.get_rank() == 0 and curr_step % 10 == 0:
            logger.info(f"[Data Check Step {curr_step}] 'padded_images' in data: {'padded_images' in data}")

        if training_args.pixel_loss_weight > 0 or training_args.ssim_loss_weight > 0:
            data['pixel_loss_weight'] = training_args.pixel_loss_weight
            data['pixel_loss_type'] = training_args.pixel_loss_type
            data['pixel_loss_max_t'] = training_args.pixel_loss_max_t
            data['pixel_loss_paired_only'] = training_args.pixel_loss_paired_only
            data['pixel_loss_chunk_size'] = training_args.pixel_loss_chunk_size
            data['pixel_loss_adaptive_chunk'] = training_args.pixel_loss_adaptive_chunk
            data['pixel_loss_use_v0'] = training_args.pixel_loss_use_v0
            data['ssim_loss_weight'] = training_args.ssim_loss_weight
            data['ssim_loss_max_t'] = training_args.ssim_loss_max_t
            data['ssim_window_size'] = training_args.ssim_window_size
            data['ssim_grad_scale'] = training_args.ssim_grad_scale
        else:
            data.pop('padded_images', None)

        # ---- Forward pass ----
        stage_ref[0] = "forward"
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            loss_dict = ds_engine(**data)

        loss = 0

        # ---- CE loss ----
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
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        # ---- MSE + Pixel + SSIM losses ----
        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data.get('mse_loss_indexes', [])), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss += mse * training_args.mse_weight

            pixel = loss_dict.get("pixel")
            if pixel is not None:
                pixel_loss_debug = os.environ.get("PIXEL_LOSS_DEBUG", "").lower() in {"1", "true", "yes", "y"}
                if pixel_loss_debug and dist.is_initialized():
                    try:
                        pixel_val = pixel.detach().float().view(1)
                    except Exception:
                        pixel_val = torch.tensor([float("nan")], device=device)

                    abnormal_local = (
                        (not torch.isfinite(pixel_val).all().item())
                        or float(pixel_val.item()) > 1.01
                    )
                    if abnormal_local:
                        logger.warning(
                            f"[Pixel Loss Train Debug] abnormal_local=True rank={int(dist.get_rank())} "
                            f"step={curr_step} pixel={float(pixel_val.item()):.6g}"
                        )

                loss_dict["pixel"] = pixel.detach()
                loss += pixel * training_args.pixel_loss_weight
            else:
                loss_dict["pixel"] = torch.tensor(0, device=device)

            # SSIM Loss
            ssim = loss_dict.get("ssim")
            if ssim is not None:
                ssim_loss_debug = os.environ.get("SSIM_LOSS_DEBUG", "").lower() in {"1", "true", "yes", "y"}
                if ssim_loss_debug and dist.is_initialized():
                    try:
                        ssim_val = ssim.detach().float().view(1)
                    except Exception:
                        ssim_val = torch.tensor([float("nan")], device=device)

                    abnormal_local = (
                        (not torch.isfinite(ssim_val).all().item())
                        or float(ssim_val.item()) > 1.01
                    )
                    if abnormal_local:
                        logger.warning(
                            f"[SSIM Loss Train Debug] abnormal_local=True rank={int(dist.get_rank())} "
                            f"step={curr_step} ssim={float(ssim_val.item()):.6g}"
                        )

                loss_dict["ssim"] = ssim.detach()
                loss += ssim * training_args.ssim_loss_weight
            else:
                loss_dict["ssim"] = torch.tensor(0, device=device)
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            loss_dict["pixel"] = torch.tensor(0, device=device)
            loss_dict["ssim"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        # ---- Backward + step (DeepSpeed handles gradient clipping) ----
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Step {curr_step}: Abnormal loss value: {loss}")
            torch.distributed.barrier()

        stage_ref[0] = "backward"
        ds_engine.backward(loss)

        stage_ref[0] = "optimizer_step"
        # ds_engine.step() internally calls optimizer.step() + scheduler.step()
        # (scheduler was passed via lr_scheduler= to deepspeed.initialize)
        ds_engine.step()
        torch.cuda.empty_cache()

        # EMA update
        deepspeed_ema_update(ema_model, ds_engine, decay=training_args.ema)

        # ---- Logging ----
        if curr_step % training_args.log_every == 0:
            stage_ref[0] = "logging"
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize()
            end_time = time_second()
            steps_per_sec = training_args.log_every / (end_time - start_time)
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

                # Catch abnormal pixel loss before all_reduce
                if key == "pixel":
                    pixel_val = float(avg_loss.item())
                    is_abnormal = (not torch.isfinite(avg_loss).all().item()) or (pixel_val > 1.0)
                    if is_abnormal:
                        logger.warning(
                            f"[Pixel Loss Pre-AllReduce] rank={dist.get_rank()} step={curr_step:07d} "
                            f"pixel={pixel_val:.6g} is abnormal, clamping to 0"
                        )
                        avg_loss = torch.tensor(0.0, device=device)

                # Catch abnormal SSIM loss before all_reduce
                if key == "ssim":
                    ssim_val = float(avg_loss.item())
                    is_abnormal = (not torch.isfinite(avg_loss).all().item()) or (ssim_val > 1.0)
                    if is_abnormal:
                        logger.warning(
                            f"[SSIM Loss Pre-AllReduce] rank={dist.get_rank()} step={curr_step:07d} "
                            f"ssim={ssim_val:.6g} is abnormal, clamping to 0"
                        )
                        avg_loss = torch.tensor(0.0, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                log[key] = avg_loss

            # Weighted loss terms
            weighted_terms = {}
            if "ce" in log:
                weighted_terms["ce_weighted"] = log["ce"] * training_args.ce_weight
            if "mse" in log:
                weighted_terms["mse_weighted"] = log["mse"] * training_args.mse_weight
            if "pixel" in log:
                weighted_terms["pixel_weighted"] = log["pixel"] * training_args.pixel_loss_weight
            if "ssim" in log:
                weighted_terms["ssim_weighted"] = log["ssim"] * training_args.ssim_loss_weight

            for key, avg_loss in weighted_terms.items():
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                log[key] = avg_loss

            # Total loss (averaged across ranks)
            total_loss_avg = loss.detach()
            if total_loss_avg.device != device:
                total_loss_avg = total_loss_avg.to(device)
            dist.all_reduce(total_loss_avg, op=dist.ReduceOp.SUM)
            total_loss_avg = total_loss_avg.item() / dist.get_world_size()
            message += f"Train Loss total: {total_loss_avg:.4f}, "
            log["loss_total"] = total_loss_avg

            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)

            for i, group in enumerate(optimizer.param_groups):
                group_name = group.get('name', f'group_{i}')
                log[f'lr_{group_name}'] = group['lr']
            log['lr'] = optimizer.param_groups[0]['lr']
            log['total_mse_tokens'] = total_mse_tokens.item()
            log['total_ce_tokens'] = total_ce_tokens.item()
            # DeepSpeed handles gradient clipping; log the norm if available
            log['total_norm'] = 0.0  # DeepSpeed clips internally
            log['total_samples'] = total_samples.item()
            log['training_mode'] = 'deepspeed_zero2'

            if dist.get_rank() == 0 and writer is not None:
                for key, value in log.items():
                    if isinstance(value, (int, float, torch.Tensor)):
                        scalar_value = value.item() if isinstance(value, torch.Tensor) else value
                        writer.add_scalar(f'train/{key}', scalar_value, curr_step)
            start_time = time_second()

        # ---- Data status tracking ----
        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        # ---- Checkpoint save ----
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            stage_ref[0] = "checkpoint_save"
            logger.info("Saving checkpoint...")
            torch.cuda.empty_cache()

            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)

            actual_save_path = DeepSpeedCheckpoint.save_checkpoint(
                save_path=os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}"),
                train_steps=curr_step,
                ds_engine=ds_engine,
                ema_model=ema_model,
                scheduler=scheduler,
                data_status=gather_list,
                logger=logger,
                tokenizer=tokenizer,
                vae_model=vae_model if training_args.visual_gen else None,
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                tensorboard_run_name=tensorboard_run_name if dist.get_rank() == 0 else None,
                tensorboard_log_dir=tensorboard_dir if (dist.get_rank() == 0 and writer is not None) else None,
            )

            # Checkpoint rotation (keep latest N)
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

        if watchdog_timer is not None:
            watchdog_timer.cancel()

    if dist.get_rank() == 0:
        logger.info("=" * 50)
        logger.info(f"Training completed: {curr_step} steps")
        logger.info("=" * 50)
        if writer is not None:
            writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
