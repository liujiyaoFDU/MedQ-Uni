# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSpeed utilities for distributed training.

This module provides DeepSpeed-based alternatives to the FSDP utilities in fsdp_utils.py.
It reuses checkpoint save/load helpers (safetensors format) from fsdp_utils.py
for inference compatibility.

Key differences from FSDP:
- Uses DeepSpeed ZeRO-2 instead of FSDP HYBRID_SHARD
- EMA model is a plain deepcopy (no FSDP/DDP wrapping needed with ZeRO-2)
- Optimizer/scheduler are created externally (not via DeepSpeed JSON config)
  to support per-component learning rates
- Checkpoint format is identical (safetensors) for inference compatibility
"""

import logging
import os
import json
from datetime import datetime

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

# Reuse checkpoint helpers from fsdp_utils (no FSDP dependency)
from train.fsdp_utils import (
    dataclass_to_dict,
    get_clean_state_dict,
    save_component_weights,
    load_component_weights,
    validate_checkpoint_integrity,
    migrate_old_checkpoint,
    detect_model_wrapping,
    grad_checkpoint_check_fn,
    FSDPCheckpoint,
)
from train.train_utils import get_latest_ckpt, create_logger

_logger = logging.getLogger(__name__)


def generate_deepspeed_config(training_args):
    """
    Build DeepSpeed config dict programmatically from TrainingArguments.

    The config does NOT include an "optimizer" key because an external
    optimizer with per-component learning rates (vae_lr, vit_lr, llm_lr)
    is passed directly to deepspeed.initialize(optimizer=...).

    Args:
        training_args: TrainingArguments dataclass instance.

    Returns:
        dict: DeepSpeed configuration dictionary.
    """
    zero_stage = getattr(training_args, 'zero_stage', 2)

    ds_config = {
        # Batch size: DataLoader uses batch_size=1 with token packing.
        # Set explicit micro_batch and grad_accum, DeepSpeed computes train_batch_size automatically.
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "steps_per_print": getattr(training_args, 'log_every', 10),

        # ZeRO optimization
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
            "allgather_partitions": True,
            "allgather_bucket_size": int(5e8),
            "reduce_bucket_size": int(5e8),
        },

        # Mixed precision: bfloat16
        "bf16": {
            "enabled": True,
        },

        # Allow torch.amp.autocast to propagate into the engine.
        # Without this, DeepSpeed disables external autocast inside engine.forward(),
        # causing dtype mismatches (e.g. float32 timestep embeddings vs bf16 weights).
        "torch_autocast": {
            "enabled": True,
            "dtype": "bfloat16",
        },

        # Gradient clipping (replaces fsdp_model.clip_grad_norm_)
        "gradient_clipping": getattr(training_args, 'max_grad_norm', 1.0),

        # No "optimizer" key here: an external optimizer with per-component
        # learning rates (vae_lr, vit_lr, llm_lr) is passed to
        # deepspeed.initialize(optimizer=...) and takes precedence.

        # Wall clock breakdown for profiling (disabled by default)
        "wall_clock_breakdown": False,
    }

    # ZeRO-3 specific options
    if zero_stage == 3:
        ds_config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": int(5e7),
            "stage3_param_persistence_threshold": int(1e5),
            "stage3_gather_16bit_weights_on_model_save": True,
        })

    return ds_config


class DeepSpeedCheckpoint:
    """Checkpoint management for DeepSpeed training.

    Saves checkpoints in the same safetensors format as FSDPCheckpoint
    for full inference compatibility with the existing pipeline.
    """

    @staticmethod
    def save_checkpoint(
        save_path,
        train_steps,
        ds_engine,
        ema_model,
        scheduler,
        data_status,
        logger,
        tokenizer=None,
        vae_model=None,
        model_args=None,
        data_args=None,
        training_args=None,
        **kwargs,
    ):
        """Save checkpoint in safetensors format compatible with existing inference.

        With ZeRO-2, ds_engine.module.state_dict() returns the full model state
        dict (parameters are not partitioned). We save exactly the same file
        structure as FSDPCheckpoint for inference compatibility.

        Args:
            save_path: Directory path (will be created).
            train_steps: Current training step (used in directory name).
            ds_engine: DeepSpeed engine.
            ema_model: EMA model (plain nn.Module, not wrapped).
            scheduler: LR scheduler instance.
            data_status: Data loader state for resuming.
            logger: Logger instance.
            tokenizer: Optional tokenizer for inference files.
            vae_model: Optional external VAE model.
            model_args, data_args, training_args: Config dataclasses.
            **kwargs: Additional metadata (tensorboard_run_name, etc.).

        Returns:
            str: Actual save path.
        """
        # Path handling: support both directory and complete path formats
        normalized_path = os.path.normpath(os.path.abspath(save_path))
        step_pattern = f"{train_steps:07d}"
        path_basename = os.path.basename(normalized_path)

        if path_basename == step_pattern:
            actual_save_path = normalized_path
        elif path_basename.isdigit() and len(path_basename) == 7:
            logger.warning(f"Path step mismatch: '{path_basename}' vs {train_steps}, correcting...")
            actual_save_path = os.path.join(os.path.dirname(normalized_path), step_pattern)
        else:
            actual_save_path = os.path.join(normalized_path, step_pattern)

        os.makedirs(actual_save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {actual_save_path}")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # ---- Model weights (rank 0 only for ZeRO-2) ----
        if rank == 0:
            torch.cuda.empty_cache()

            # Main model
            state_dict = ds_engine.module.state_dict()
            clean_dict, metadata = get_clean_state_dict(state_dict)
            save_file(clean_dict, os.path.join(actual_save_path, "model.safetensors"))
            logger.info(f"Saved main model: {len(clean_dict)} keys")

            # Component-wise saves
            save_component_weights(state_dict, actual_save_path, "vit_model", logger)
            vae_saved = save_component_weights(state_dict, actual_save_path, "vae_model", logger)
            save_component_weights(state_dict, actual_save_path, "language_model", logger)

            # External VAE fallback
            if not vae_saved and vae_model is not None:
                try:
                    external_vae_sd = vae_model.state_dict()
                    if external_vae_sd:
                        clean_vae, _ = get_clean_state_dict(external_vae_sd)
                        save_file(clean_vae, os.path.join(actual_save_path, "vae_model.safetensors"))
                        logger.info(f"Saved external VAE: {len(clean_vae)} keys")
                        del external_vae_sd, clean_vae
                except Exception as e:
                    logger.error(f"Failed to save external VAE: {e}")

            del state_dict, clean_dict
            torch.cuda.empty_cache()

            # EMA model
            if ema_model is not None:
                ema_sd = ema_model.state_dict()
                clean_ema, _ = get_clean_state_dict(ema_sd)
                save_file(clean_ema, os.path.join(actual_save_path, "ema.safetensors"))
                logger.info(f"Saved EMA model: {len(clean_ema)} keys")
                del ema_sd, clean_ema
                torch.cuda.empty_cache()

        # ---- Optimizer state (sharded per rank) ----
        try:
            opt_state = ds_engine.optimizer.state_dict()
            opt_path = os.path.join(
                actual_save_path, f"optimizer.{rank:05d}-of-{world_size:05d}.pt"
            )
            torch.save(opt_state, opt_path)
            del opt_state
        except Exception as e:
            logger.warning(f"Failed to save optimizer state on rank {rank}: {e}")

        # ---- Scheduler, data_status, inference files (rank 0 only) ----
        if rank == 0:
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(actual_save_path, "scheduler.pt"))

            if data_status is not None:
                torch.save(data_status, os.path.join(actual_save_path, "data_status.pt"))

            # Inference files (reuse the static method from FSDPCheckpoint)
            FSDPCheckpoint._save_inference_files(
                actual_save_path,
                ds_engine.module,  # unwrapped model
                tokenizer,
                vae_model,
                model_args,
                data_args,
                training_args,
                logger,
                train_steps=train_steps,
                **{k: v for k, v in kwargs.items()
                   if k in ('tensorboard_run_name', 'tensorboard_log_dir')},
            )

        dist.barrier()
        logger.info(f"Checkpoint saved successfully to {actual_save_path}")

        # Validate
        if rank == 0:
            expected = []
            if hasattr(ds_engine.module, 'vae_model') or vae_model is not None:
                expected.append('vae_model')
            if hasattr(ds_engine.module, 'vit_model'):
                expected.append('vit_model')
            if hasattr(ds_engine.module, 'language_model'):
                expected.append('language_model')
            validate_checkpoint_integrity(actual_save_path, expected, logger)

        return actual_save_path

    @staticmethod
    def load_model_checkpoint(resume_from, logger, model, ema_model=None,
                              finetune_from_ema=False, external_vae_model=None):
        """Load model/EMA weights before deepspeed.initialize().

        This reuses the same loading logic as FSDPCheckpoint.try_load_ckpt()
        because model loading happens before any wrapping (FSDP or DeepSpeed).

        Args:
            resume_from: Checkpoint directory path (or None).
            logger: Logger instance.
            model: Model to load weights into.
            ema_model: Optional EMA model.
            finetune_from_ema: If True, load EMA weights into main model.
            external_vae_model: Optional external VAE model.

        Returns:
            tuple: (model, ema_model)
        """
        # Delegate to FSDPCheckpoint.try_load_ckpt which does plain
        # state_dict loading (no FSDP-specific operations)
        return FSDPCheckpoint.try_load_ckpt(
            resume_from, logger, model, ema_model,
            finetune_from_ema=finetune_from_ema,
            external_vae_model=external_vae_model,
        )

    @staticmethod
    def load_train_state(resume_from, ds_engine, scheduler, logger,
                         resume_model_optimizer=True):
        """Load optimizer, scheduler, step count, and data status after deepspeed.initialize().

        Args:
            resume_from: Checkpoint directory path.
            ds_engine: DeepSpeed engine (optimizer is accessed via ds_engine.optimizer).
            scheduler: LR scheduler instance.
            logger: Logger instance.
            resume_model_optimizer: Whether to load optimizer state.

        Returns:
            tuple: (ds_engine, scheduler, train_steps, data_status)
        """
        if resume_from is None or not os.path.exists(resume_from):
            return ds_engine, scheduler, 0, None

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # ---- Optimizer ----
        optimizer_loaded = False
        if resume_model_optimizer:
            opt_file = os.path.join(
                resume_from, f"optimizer.{rank:05d}-of-{world_size:05d}.pt"
            )
            if os.path.exists(opt_file):
                try:
                    opt_state = torch.load(opt_file, map_location="cpu", weights_only=True)

                    # Validate parameter group compatibility
                    saved_groups = opt_state.get('param_groups', [])
                    current_groups = ds_engine.optimizer.param_groups

                    if len(saved_groups) != len(current_groups):
                        _logger.warning(
                            f"Optimizer group count mismatch: saved={len(saved_groups)}, "
                            f"current={len(current_groups)}. Using fresh optimizer."
                        )
                    else:
                        compatible = True
                        for i, (sg, cg) in enumerate(zip(saved_groups, current_groups)):
                            if len(sg.get('params', [])) != len(cg.get('params', [])):
                                _logger.warning(
                                    f"Group {i} param count mismatch: "
                                    f"saved={len(sg.get('params', []))}, "
                                    f"current={len(cg.get('params', []))}"
                                )
                                compatible = False
                                break

                        if compatible:
                            ds_engine.optimizer.load_state_dict(opt_state)
                            optimizer_loaded = True
                            _logger.info(f"Optimizer state loaded from {opt_file}")
                        else:
                            _logger.warning("Optimizer incompatible. Using fresh optimizer.")

                    del opt_state
                except Exception as e:
                    _logger.warning(f"Failed to load optimizer: {e}. Using fresh optimizer.")
            else:
                _logger.info(f"Optimizer file not found: {opt_file}")

        # ---- Scheduler ----
        if optimizer_loaded:
            sched_file = os.path.join(resume_from, "scheduler.pt")
            if os.path.exists(sched_file):
                try:
                    sched_state = torch.load(sched_file, map_location="cpu", weights_only=True)
                    scheduler.load_state_dict(sched_state)
                    _logger.info("Scheduler state loaded")
                    del sched_state
                except Exception as e:
                    _logger.warning(f"Failed to load scheduler: {e}")
        else:
            _logger.info("Optimizer not loaded, skipping scheduler loading.")

        # ---- Train step ----
        checkpoint_name = os.path.basename(os.path.normpath(resume_from))
        try:
            train_steps = int(checkpoint_name) + 1
        except ValueError:
            train_state_path = os.path.join(resume_from, "train_state.json")
            if os.path.exists(train_state_path):
                with open(train_state_path, 'r') as f:
                    train_state = json.load(f)
                    train_steps = train_state.get('train_steps', 0)
                _logger.info(f"Loaded train_steps from train_state.json: {train_steps}")
            else:
                train_steps = 0
                _logger.warning(
                    f"Checkpoint '{checkpoint_name}' is not numeric and "
                    "train_state.json not found, starting from step 0"
                )

        # ---- Data status ----
        data_status = None
        data_status_path = os.path.join(resume_from, "data_status.pt")
        if os.path.exists(data_status_path):
            data_status_all = torch.load(data_status_path, map_location="cpu", weights_only=True)
            if rank < len(data_status_all):
                data_status = data_status_all[rank]
            else:
                data_status = None

        return ds_engine, scheduler, train_steps, data_status


@torch.no_grad()
def deepspeed_ema_update(ema_model, ds_engine, decay=0.9999):
    """Update EMA model parameters from DeepSpeed engine.

    With ZeRO-2, all model parameters are materialized on each GPU,
    so we can directly iterate over parameters without gather operations.
    This is much simpler than the FSDP version which needs to traverse
    internal FSDP handles and separately handle DDP-wrapped VAE.

    Args:
        ema_model: Plain nn.Module (not wrapped) with requires_grad=False.
        ds_engine: DeepSpeed engine containing the training model.
        decay: EMA decay factor (default 0.9999).
    """
    ema_params = []
    new_params = []

    for ema_p, model_p in zip(ema_model.parameters(), ds_engine.module.parameters()):
        if model_p.requires_grad:
            ema_params.append(ema_p.data)
            new_params.append(model_p.data.to(dtype=ema_p.dtype))

    if ema_params:
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, new_params, alpha=1 - decay)

    total_params = sum(p.numel() for p in ema_params) if ema_params else 0
    _logger.debug(f"EMA updated: {total_params} parameters, decay={decay}")


def apply_deepspeed_activation_checkpointing(model, check_fn=None):
    """Apply activation checkpointing to eligible layers.

    Uses PyTorch's native checkpoint wrapper (same as the FSDP version)
    which is compatible with DeepSpeed.

    Args:
        model: The unwrapped model (ds_engine.module).
        check_fn: Predicate function to identify eligible modules.
                   Defaults to grad_checkpoint_check_fn from fsdp_utils.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    import functools

    if check_fn is None:
        check_fn = grad_checkpoint_check_fn

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=check_fn,
    )
    _logger.info("Activation checkpointing applied")
