# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Unimedvl Team
# SPDX-License-Identifier: Apache-2.0

"""
Loss Functions for Bagel Model

This module contains modular implementations of loss functions used in the Bagel model:
- Cross Entropy Loss: Language modeling loss
- MSE Loss: Rectified flow velocity matching loss
- Pixel Loss: Pixel-space fidelity loss for image restoration tasks
"""

import logging
import os
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)


def compute_ce_loss(
    last_hidden_state: torch.Tensor,
    ce_loss_indexes: Optional[torch.BoolTensor],
    packed_label_ids: Optional[torch.LongTensor],
    lm_head: nn.Module,
) -> Optional[torch.Tensor]:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        last_hidden_state: Model output hidden states, shape (sequence_length, hidden_size)
        ce_loss_indexes: Boolean mask or indices indicating which positions to compute CE loss
        packed_label_ids: Target token IDs for language modeling
        lm_head: Language model head (linear layer) to project hidden states to vocabulary

    Returns:
        CE loss tensor with shape (num_ce_tokens,), or None if ce_loss_indexes is None
    """
    ce = None
    if ce_loss_indexes is not None:
        packed_ce_preds = lm_head(last_hidden_state[ce_loss_indexes])
        ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
    return ce


def compute_mse_loss(
    last_hidden_state: torch.Tensor,
    mse_loss_indexes: torch.BoolTensor,
    llm2vae: nn.Module,
    packed_latent_clean: torch.Tensor,
    noise: torch.Tensor,
    packed_timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MSE loss for rectified flow velocity matching.

    This implements the velocity matching objective for diffusion models:
    - Clean latent: z0
    - Noise: epsilon
    - Noisy latent: zt = (1-t)*z0 + t*epsilon
    - Target velocity: v = epsilon - z0
    - Predicted velocity: v_pred (from model)
    - Loss: MSE(v_pred, v) for t > 0

    Args:
        last_hidden_state: Model output hidden states
        mse_loss_indexes: Mask indicating which positions need MSE supervision
        llm2vae: Linear projection from LLM hidden state to VAE latent space
        packed_latent_clean: Clean latent z0, shape (num_tokens, latent_dim)
        noise: Noise epsilon, shape (num_tokens, latent_dim)
        packed_timesteps: Timestep t for each token, shape (num_tokens,)

    Returns:
        MSE loss tensor, shape (num_supervised_tokens, latent_dim)
    """
    # Aliases for clarity
    z0_tokens_clean = packed_latent_clean
    noise_tokens = noise
    t_tokens_all = packed_timesteps

    # Predict velocity
    velocity_pred = llm2vae(last_hidden_state[mse_loss_indexes])

    # Target velocity
    velocity_target = noise_tokens - z0_tokens_clean

    # Supervise only tokens with t > 0
    supervise_mask = t_tokens_all > 0

    # Compute squared error
    mse = (velocity_pred - velocity_target[supervise_mask]) ** 2

    return mse


def compute_pixel_loss(
    # 核心模型输出
    last_hidden_state: torch.Tensor,
    mse_loss_indexes: torch.BoolTensor,
    llm2vae: nn.Module,
    # 潜变量和噪声
    packed_latent_clean: torch.Tensor,
    noise: torch.Tensor,
    packed_timesteps: torch.Tensor,
    # 图像和形状信息
    padded_images: Optional[torch.Tensor],
    patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]],
    packed_vae_token_indexes: Optional[torch.LongTensor],
    # VAE 解码器
    vae_decode_fn: Callable[[torch.Tensor], torch.Tensor],
    # 模型配置
    latent_patch_size: int,
    latent_channel: int,
    latent_downsample: int,
    # 损失配置
    pixel_loss_weight: float,
    pixel_loss_type: str,
    pixel_loss_max_t: float,
    # 状态标志
    is_training: bool,
) -> Optional[torch.Tensor]:
    """
    Compute pixel-space fidelity loss for image restoration tasks (SR, denoising, etc.).

    This loss operates in pixel space by:
    1. Predicting clean latent z0 from noisy latent zt using velocity matching
    2. Decoding predicted z0 to pixel space using VAE decoder
    3. Computing L1/L2 loss between predicted and ground truth pixels
    4. Applying timestep-based weighting (only for low-noise steps, t <= pixel_loss_max_t)

    Args:
        last_hidden_state: Model output hidden states
        mse_loss_indexes: Mask for MSE supervision positions
        llm2vae: Projection from LLM hidden to VAE latent
        packed_latent_clean: Clean latent z0
        noise: Noise epsilon
        packed_timesteps: Timestep for each token
        padded_images: Ground truth images (padded to max size), shape (B, C, H, W)
        patchified_vae_latent_shapes: List of (h, w) for each image's latent grid
        packed_vae_token_indexes: Indices of VAE tokens in packed sequence
        vae_decode_fn: VAE decoder function (latent -> pixel)
        latent_patch_size: Patch size in latent space
        latent_channel: Number of latent channels
        latent_downsample: Downsampling factor (VAE encoder)
        pixel_loss_weight: Loss weight (must be > 0)
        pixel_loss_type: "l1" or "l2"/"mse"
        pixel_loss_max_t: Max timestep to apply pixel loss (high SNR only)
        is_training: Training mode flag

    Returns:
        Pixel loss scalar tensor, or None if conditions not met or no valid images
    """
    pixel = None  # 初始化像素损失，若条件不满足则保持为空

    # Aliases for clarity
    z0_tokens_clean = packed_latent_clean
    noise_tokens = noise
    t_tokens_all = packed_timesteps

    # ========== Pixel Loss Entry Diagnostics ==========
    pixel_loss_debug = os.environ.get("PIXEL_LOSS_DEBUG", "").lower() in {"1", "true", "yes", "y"}
    pixel_loss_debug_verbose = os.environ.get("PIXEL_LOSS_DEBUG_VERBOSE", "").lower() in {"1", "true", "yes", "y"}
    if pixel_loss_debug and is_training and pixel_loss_weight > 0:
        if dist.get_rank() == 0:
            logger.info("="*60)
            logger.info("[Pixel Loss Entry] Checking entry conditions")
            logger.info(f"  pixel_loss_weight={pixel_loss_weight}")
            logger.info(f"  pixel_loss_max_t={pixel_loss_max_t}")
            logger.info(f"  padded_images is None: {padded_images is None}")
            logger.info(f"  vae_decode_fn is None: {vae_decode_fn is None}")
            logger.info(f"  patchified_vae_latent_shapes is None: {patchified_vae_latent_shapes is None}")
            logger.info(f"  packed_vae_token_indexes is None: {packed_vae_token_indexes is None}")
            logger.info(f"  packed_timesteps is None: {packed_timesteps is None}")
            if patchified_vae_latent_shapes is not None:
                logger.info(f"  len(patchified_vae_latent_shapes)={len(patchified_vae_latent_shapes)}")
            logger.info("="*60)
    # ===================================================

    if (
        is_training
        and pixel_loss_weight > 0
        and pixel_loss_max_t > 0
        and padded_images is not None
        and vae_decode_fn is not None
        and patchified_vae_latent_shapes is not None
        and packed_vae_token_indexes is not None
        and packed_timesteps is not None
        and len(patchified_vae_latent_shapes) > 0
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if pixel_loss_debug:
            logger.info(f"++ [Pixel Loss Entry] rank={rank} conditions met")

        # Predict velocity
        velocity_pred = llm2vae(last_hidden_state[mse_loss_indexes])
        velocity_target = noise_tokens - z0_tokens_clean
        supervise_mask = t_tokens_all > 0

        # Ensure prediction count matches supervised tokens
        if velocity_pred.shape[0] == int(supervise_mask.sum().item()):
            # Build per-token prediction of clean latent z0: ẑ0 = zt - t * v̂
            t_tokens_supervised = t_tokens_all[supervise_mask].to(z0_tokens_clean.dtype)
            z0_tokens_pred = z0_tokens_clean[supervise_mask] + t_tokens_supervised[:, None] * (velocity_target[supervise_mask] - velocity_pred)

            z0_tokens_hybrid = z0_tokens_clean.clone()
            z0_tokens_hybrid[supervise_mask] = z0_tokens_pred

            # Compute per-image timestep
            tokens_per_image = [h * w for (h, w) in patchified_vae_latent_shapes]
            total_tokens_expected = sum(tokens_per_image)

            if total_tokens_expected == z0_tokens_hybrid.shape[0]:
                device = z0_tokens_hybrid.device
                tokens_per_image_t = torch.tensor(tokens_per_image, device=device, dtype=torch.long)
                image_start_ptrs = torch.cat(
                    [torch.zeros(1, device=device, dtype=torch.long), tokens_per_image_t.cumsum(0)[:-1]],
                    dim=0,
                )

                # First timestep per image
                t_img = t_tokens_all[image_start_ptrs].to(z0_tokens_clean.dtype)

                # Optional: verify mapping
                if pixel_loss_debug and pixel_loss_debug_verbose:
                    max_verbose_env = os.environ.get("PIXEL_LOSS_DEBUG_VERBOSE_MAX", "5")
                    try:
                        max_verbose = int(max_verbose_env)
                    except Exception:
                        max_verbose = 5

                    # Track verbose report count (use a simple module-level counter)
                    if not hasattr(compute_pixel_loss, "_verbose_reports"):
                        compute_pixel_loss._verbose_reports = 0

                    verbose_reports = compute_pixel_loss._verbose_reports
                    if verbose_reports < max_verbose:
                        compute_pixel_loss._verbose_reports += 1

                        vae_seq_pos = packed_vae_token_indexes
                        is_sorted = bool(torch.all(vae_seq_pos[1:] >= vae_seq_pos[:-1]).item()) if vae_seq_pos.numel() > 1 else True
                        mse_kind = "none"
                        mse_count = -1
                        mse_min = -1
                        mse_max = -1

                        if mse_loss_indexes is not None:
                            if mse_loss_indexes.dtype == torch.bool:
                                mse_kind = "mask"
                                mse_count = int(mse_loss_indexes.sum().item())
                                mse_seq_pos = torch.nonzero(mse_loss_indexes, as_tuple=False).squeeze(-1)
                            else:
                                mse_kind = "indices"
                                mse_seq_pos = mse_loss_indexes.to(device=device, dtype=torch.long).view(-1)
                                mse_count = int(mse_seq_pos.numel())

                            if mse_seq_pos.numel() > 0:
                                mse_min = int(mse_seq_pos.min().item())
                                mse_max = int(mse_seq_pos.max().item())

                        logger.info(
                            f"[Pixel Loss Debug] mapping_check rank={int(dist.get_rank())} "
                            f"total_vae_tokens={int(vae_seq_pos.numel())} expected={int(total_tokens_expected)} "
                            f"vae_seq_pos(min/max)={int(vae_seq_pos.min().item())}/{int(vae_seq_pos.max().item())} "
                            f"sorted={is_sorted} "
                            f"supervised_tokens={int(supervise_mask.sum().item())} "
                            f"mse_loss_indexes(kind={mse_kind},count={mse_count},min/max={mse_min}/{mse_max})"
                        )

                        # Verify mse_loss_indexes matches expected
                        if mse_loss_indexes is not None and mse_kind != "none":
                            expected_mse_seq_pos = vae_seq_pos[supervise_mask]
                            same_count = int(mse_seq_pos.numel()) == int(expected_mse_seq_pos.numel())
                            same_set = False
                            if same_count:
                                mse_sorted = torch.sort(mse_seq_pos).values
                                expected_sorted = torch.sort(expected_mse_seq_pos).values
                                same_set = bool(torch.equal(mse_sorted, expected_sorted))

                            if (not same_set) and dist.get_rank() == 0:
                                logger.warning(
                                    "[Pixel Loss Debug] mse_loss_indexes does not match expected supervised VAE token positions; "
                                    "this suggests sequence packing / timestep-mask misalignment."
                                )

                # Target images: t>0 (conditioning images use t=0 via -inf sentinel)
                image_is_target = t_img > 0

                # Weight only low-noise steps: linear ramp w(t) = max(0, (t_max - t)/t_max)
                w_img = torch.zeros_like(t_img, dtype=packed_latent_clean.dtype)
                w_img[image_is_target] = torch.clamp(
                    (float(pixel_loss_max_t) - t_img[image_is_target]) / float(pixel_loss_max_t),
                    min=0.0,
                    max=1.0,
                )

                selected = w_img > 0

                # Debug summary
                if pixel_loss_debug:
                    if dist.get_rank() == 0:
                        logger.info(
                            f"[Pixel Loss Debug] images={len(t_img)} target={int(image_is_target.sum().item())} "
                            f"selected(w>0)={int(selected.sum().item())} "
                            f"t_img(min/mean/max)={t_img.min().item():.4f}/{t_img.mean().item():.4f}/{t_img.max().item():.4f}"
                        )
                        if bool(selected.any().item()):
                            logger.info(
                                f"[Pixel Loss Debug] w_img(min/mean/max)={w_img[selected].min().item():.4f}/"
                                f"{w_img[selected].mean().item():.4f}/{w_img[selected].max().item():.4f} "
                                f"pixel_loss_max_t={float(pixel_loss_max_t):.4f}"
                            )

                if bool(selected.any().item()):
                    p = latent_patch_size
                    c = latent_channel
                    vae_downsample = latent_downsample // p

                    # Collect predicted latents + GT images for selected images
                    pred_latents = []
                    gt_images = []
                    weights = []

                    if pixel_loss_debug and dist.get_rank() == 0:
                        selected_idx = torch.nonzero(selected).squeeze(-1).tolist()
                        logger.info(
                            f"[Pixel Loss Debug] selected_idx={selected_idx} "
                            f"tokens_per_image={tokens_per_image} "
                            f"w_img_selected={[w_img[i].item() for i in selected_idx]}"
                        )

                    token_ptr = 0
                    for img_idx, (h, w) in enumerate(patchified_vae_latent_shapes):
                        num_img_tokens = h * w
                        if bool(selected[img_idx].item()):
                            tok = z0_tokens_hybrid[token_ptr : token_ptr + num_img_tokens]
                            tok = tok.view(h, w, p, p, c)
                            latent = torch.einsum("hwpqc->chpwq", tok).reshape(c, h * p, w * p)
                            pred_latents.append(latent)

                            H_img = h * latent_downsample
                            W_img = w * latent_downsample
                            gt_images.append(padded_images[img_idx, :, :H_img, :W_img])
                            weights.append(w_img[img_idx])

                            if pixel_loss_debug and dist.get_rank() == 0:
                                seq_slice = packed_vae_token_indexes[token_ptr : token_ptr + num_img_tokens]
                                seq_min = int(seq_slice.min().item()) if seq_slice.numel() > 0 else -1
                                seq_max = int(seq_slice.max().item()) if seq_slice.numel() > 0 else -1
                                logger.info(
                                    f"[Pixel Loss Debug] img_idx={img_idx} token_ptr={token_ptr} "
                                    f"tok_shape={tok.shape} latent_shape={latent.shape} "
                                    f"gt_shape={(H_img, W_img)} "
                                    f"weight={w_img[img_idx].item():.4f} "
                                    f"vae_seq_pos(min/max)={seq_min}/{seq_max}"
                                )
                        token_ptr += num_img_tokens

                    if len(pred_latents) > 0:
                        # Pad to max size for batched decode
                        max_h_lat = max(z.shape[1] for z in pred_latents)
                        max_w_lat = max(z.shape[2] for z in pred_latents)
                        max_h_img = max_h_lat * vae_downsample
                        max_w_img = max_w_lat * vae_downsample

                        latent_batch = z0_tokens_hybrid.new_zeros((len(pred_latents), c, max_h_lat, max_w_lat))
                        gt_batch = padded_images.new_zeros((len(gt_images), padded_images.shape[1], max_h_img, max_w_img))
                        mask = padded_images.new_zeros((len(gt_images), 1, max_h_img, max_w_img))

                        if pixel_loss_debug and dist.get_rank() == 0:
                            logger.info(
                                f"[Pixel Loss Debug] batch_shapes latent_batch={latent_batch.shape} "
                                f"gt_batch={gt_batch.shape} mask={mask.shape} "
                                f"max_h_lat={max_h_lat} max_w_lat={max_w_lat} "
                                f"max_h_img={max_h_img} max_w_img={max_w_img}"
                            )

                        for i, (z, x_gt, w_i) in enumerate(zip(pred_latents, gt_images, weights)):
                            H_lat, W_lat = z.shape[1], z.shape[2]
                            H_img, W_img = x_gt.shape[1], x_gt.shape[2]
                            latent_batch[i, :, :H_lat, :W_lat] = z
                            gt_batch[i, :, :H_img, :W_img] = x_gt
                            mask[i, :, :H_img, :W_img] = w_i.to(mask.dtype)

                        # VAE decode
                        x_pred = vae_decode_fn(latent_batch)
                        x_pred = x_pred[:, :, :max_h_img, :max_w_img]
                        gt_batch = gt_batch.to(x_pred.dtype)

                        # Compute loss in [0,1] space
                        x_pred_01 = (x_pred * 0.5 + 0.5).clamp(0, 1)
                        gt_batch_01 = (gt_batch * 0.5 + 0.5).clamp(0, 1)

                        if pixel_loss_type.lower() in {"l2", "mse"}:
                            diff = (x_pred_01 - gt_batch_01) ** 2
                        else:
                            diff = (x_pred_01 - gt_batch_01).abs()

                        denom = mask.sum() * x_pred_01.shape[1]

                        # Protect against small denom
                        MIN_DENOM = 1e-3
                        rank = dist.get_rank() if dist.is_initialized() else 0
                        if float(denom.item()) > MIN_DENOM:
                            if pixel_loss_debug:
                                logger.info(f"++ [Pixel Loss Normal] rank={rank} denom={float(denom.item()):.6g}")
                            pixel = (diff * mask).sum() / denom
                        else:
                            logger.warning(f"++ [Pixel Loss Small] rank={rank} denom={float(denom.item()):.6g} < {MIN_DENOM}, setting pixel=0")
                            if pixel_loss_debug:
                                logger.warning(
                                    f"[Pixel Loss] rank={int(dist.get_rank())} "
                                    f"denom too small: {float(denom.item()):.6g} (< {MIN_DENOM}), "
                                    f"setting pixel=0 to prevent explosion"
                                )
                            pixel = torch.tensor(0.0, device=diff.device, dtype=diff.dtype)

                        if pixel_loss_debug:
                            logger.info(f"++ [Pixel Loss Exit] rank={rank} pixel={float(pixel.item()):.6g} shape={pixel.shape} dtype={pixel.dtype}")

                        # Abnormal value diagnostics
                        if pixel_loss_debug:
                            rank = dist.get_rank() if dist.is_initialized() else 0
                            logger.warning(f"++ [Pixel Loss Abnormal] rank={rank} checking pixel={float(pixel.detach().float().item()):.6g} shape={pixel.shape} dtype={pixel.dtype}")
                            pixel_scalar = float(pixel.detach().float().item())
                            if (not torch.isfinite(pixel.detach()).all().item()) or pixel_scalar > 1.01:
                                max_reports_env = os.environ.get("PIXEL_LOSS_DEBUG_ABNORMAL_MAX", "10")
                                logger.debug(f"++ [Pixel Loss Debug] rank={rank} max_reports_env={max_reports_env}")
                                try:
                                    max_reports = int(max_reports_env)
                                except Exception:
                                    max_reports = 10

                                if not hasattr(compute_pixel_loss, "_abnormal_reports"):
                                    compute_pixel_loss._abnormal_reports = 0

                                reported = compute_pixel_loss._abnormal_reports
                                if reported >= max_reports:
                                    pass
                                else:
                                    compute_pixel_loss._abnormal_reports = reported + 1

                                def _stat(t: torch.Tensor):
                                    t_f = t.detach().float()
                                    return (
                                        f"shape={tuple(t.shape)} dtype={t.dtype} "
                                        f"finite={bool(torch.isfinite(t_f).all().item())} "
                                        f"min={float(t_f.min().item()):.6g} "
                                        f"mean={float(t_f.mean().item()):.6g} "
                                        f"max={float(t_f.max().item()):.6g}"
                                    )

                                if reported < max_reports:
                                    logger.warning("=" * 80)
                                    logger.warning("[Pixel Loss Abnormal] pixel value out of expected range")
                                    logger.warning(f"  rank={int(dist.get_rank())}")
                                    logger.warning(f"  pixel={pixel_scalar:.6g} pixel_loss_type={pixel_loss_type} pixel_loss_max_t={float(pixel_loss_max_t):.6g}")
                                    logger.warning(f"  denom={float(denom.detach().float().item()):.6g} mask_sum={float(mask.detach().float().sum().item()):.6g} C={int(x_pred_01.shape[1])}")
                                    logger.warning(
                                        f"  token_mapping total_tokens_expected={int(total_tokens_expected)} "
                                        f"packed_vae_token_indexes={int(packed_vae_token_indexes.numel())} "
                                        f"tokens_per_image_sum={int(sum(tokens_per_image))} images={len(tokens_per_image)}"
                                    )
                                    logger.warning(
                                        f"  masks supervised_tokens={int(supervise_mask.sum().item())} "
                                        f"mse_loss_indexes_sum={int(mse_loss_indexes.sum().item()) if mse_loss_indexes is not None else -1}"
                                    )
                                    logger.warning(f"  x_pred: {_stat(x_pred)}")
                                    logger.warning(f"  gt_batch: {_stat(gt_batch)}")
                                    logger.warning(f"  x_pred_01: {_stat(x_pred_01)}")
                                    logger.warning(f"  gt_batch_01: {_stat(gt_batch_01)}")
                                    logger.warning(f"  diff: {_stat(diff)}")
                                    logger.warning(f"  mask: {_stat(mask)}")
                                    logger.warning(
                                        f"  t_img(min/mean/max)={float(t_img.min().item()):.6g}/"
                                        f"{float(t_img.mean().item()):.6g}/{float(t_img.max().item()):.6g} "
                                        f"selected={int(selected.sum().item())}/{len(selected)}"
                                    )
                                    logger.warning("=" * 80)
                else:
                    # No images selected
                    if pixel_loss_debug and dist.get_rank() == 0:
                        logger.info("++ [Pixel Loss Skip] selected==0, setting pixel=0")
                    pixel = torch.tensor(0.0, device=z0_tokens_hybrid.device, dtype=z0_tokens_hybrid.dtype)

    # Final safety check
    if pixel is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if pixel_loss_debug:
            logger.info(f"++ [Pixel Loss Final Check] rank={rank} pixel={float(pixel.detach().item()):.6g} shape={pixel.shape} dtype={pixel.dtype}")
        pixel_val = float(pixel.detach().item())
        is_abnormal = (not torch.isfinite(pixel).all().item()) or (pixel_val > 1.0)
        if is_abnormal:
            logger.warning(
                f"++ [Pixel Loss Final Check Abnormal] rank={rank} "
                f"pixel={pixel_val:.6g} out of valid range [0,1], clamping to 0"
            )
            pixel = torch.tensor(0.0, device=pixel.device, dtype=pixel.dtype)

    if pixel_loss_debug:
        if pixel is not None:
            logger.info(f"++ [Pixel Loss Shape] pixel.shape={pixel.shape}")

    return pixel
