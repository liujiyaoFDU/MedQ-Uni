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


def calculate_chunk_size(
    num_images: int,
    max_h_img: int,
    max_w_img: int,
    base_chunk_size: int = 2,
    adaptive: bool = True
) -> int:
    """
    根据图像分辨率动态计算 chunk_size，用于分块 VAE decode

    Args:
        num_images: 批次中图像总数
        max_h_img: 最大图像高度（像素）
        max_w_img: 最大图像宽度（像素）
        base_chunk_size: 基础 chunk size（当 adaptive=False 时使用）
        adaptive: 是否启用自适应调整

    Returns:
        chunk_size: 最终使用的 chunk size

    Examples:
        >>> calculate_chunk_size(4, 1024, 1024, base_chunk_size=2, adaptive=True)
        1  # 大图像，chunk=1
        >>> calculate_chunk_size(4, 512, 512, base_chunk_size=2, adaptive=True)
        2  # 中等图像，chunk=2
        >>> calculate_chunk_size(8, 256, 256, base_chunk_size=2, adaptive=True)
        4  # 小图像，chunk=4
    """
    if not adaptive:
        return min(base_chunk_size, num_images)

    megapixels = (max_h_img * max_w_img) / 1e6

    if megapixels >= 1.0:    # >= 1024×1024
        chunk_size = 1
    elif megapixels >= 0.25: # >= 512×512
        chunk_size = 2
    else:                    # < 512×512
        chunk_size = 4

    return min(chunk_size, num_images)


def compute_pixel_loss_v0(
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
    [V0] Original implementation - Compute pixel-space fidelity loss for image restoration tasks (SR, denoising, etc.).

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
    # === 新增参数：分块 VAE decode ===
    pixel_loss_chunk_size: int = 2,
    pixel_loss_adaptive_chunk: bool = True,
    pixel_loss_use_v0: bool = False,
) -> Optional[torch.Tensor]:
    """
    Compute pixel-space fidelity loss with chunked VAE decode to reduce memory usage.

    This version implements memory-efficient chunked VAE decoding to prevent OOM errors
    when processing large batches of high-resolution images (e.g., 1024x1024).

    New features compared to v0:
    - Chunked VAE decode: Splits large batches into smaller chunks
    - Adaptive chunk sizing: Automatically adjusts chunk size based on image resolution
    - Streaming loss accumulation: Only stores loss scalars instead of all decoded images
    - Fallback to v0: Can switch back to original implementation via parameter or env var

    Args:
        ... (same as compute_pixel_loss_v0) ...
        pixel_loss_chunk_size: Base chunk size for VAE decode (1-4 recommended).
                                Lower = more memory saving, higher = faster.
        pixel_loss_adaptive_chunk: Enable adaptive chunk size based on image resolution.
                                    Recommended for mixed-resolution datasets.
        pixel_loss_use_v0: Force using original v0 implementation. Useful for debugging
                           or comparison. Can also be controlled via PIXEL_LOSS_USE_V0 env var.

    Returns:
        Pixel loss scalar tensor, or None if conditions not met or no valid images

    Memory savings:
        - 1024x1024 images: ~50-75% reduction (from OOM to ~18GB)
        - 512x512 images: ~36% reduction
        - 256x256 images: ~25% reduction

    Performance impact:
        - 1024x1024: ~10-15% slower
        - 512x512: ~5-8% slower
        - 256x256: <5% slower
    """
    # 可选：环境变量强制使用 v0
    import os
    use_v0 = pixel_loss_use_v0 or os.environ.get("PIXEL_LOSS_USE_V0", "0") == "1"

    if use_v0:
        # Fallback to original implementation
        return compute_pixel_loss_v0(
            last_hidden_state=last_hidden_state,
            mse_loss_indexes=mse_loss_indexes,
            llm2vae=llm2vae,
            packed_latent_clean=packed_latent_clean,
            noise=noise,
            packed_timesteps=packed_timesteps,
            padded_images=padded_images,
            patchified_vae_latent_shapes=patchified_vae_latent_shapes,
            packed_vae_token_indexes=packed_vae_token_indexes,
            vae_decode_fn=vae_decode_fn,
            latent_patch_size=latent_patch_size,
            latent_channel=latent_channel,
            latent_downsample=latent_downsample,
            pixel_loss_weight=pixel_loss_weight,
            pixel_loss_type=pixel_loss_type,
            pixel_loss_max_t=pixel_loss_max_t,
            is_training=is_training,
        )

    # ========== 新版分块实现开始 ==========
    pixel = None  # 初始化像素损失

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
            logger.info("[Pixel Loss Entry - Chunked Version] Checking entry conditions")
            logger.info(f"  pixel_loss_weight={pixel_loss_weight}")
            logger.info(f"  pixel_loss_max_t={pixel_loss_max_t}")
            logger.info(f"  pixel_loss_chunk_size={pixel_loss_chunk_size}")
            logger.info(f"  pixel_loss_adaptive_chunk={pixel_loss_adaptive_chunk}")
            logger.info(f"  padded_images is None: {padded_images is None}")
            logger.info(f"  vae_decode_fn is None: {vae_decode_fn is None}")
            if patchified_vae_latent_shapes is not None:
                logger.info(f"  len(patchified_vae_latent_shapes)={len(patchified_vae_latent_shapes)}")
            logger.info("="*60)

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
            logger.info(f"++ [Pixel Loss Entry - Chunked] rank={rank} conditions met")

        # Predict velocity
        velocity_pred = llm2vae(last_hidden_state[mse_loss_indexes])
        velocity_target = noise_tokens - z0_tokens_clean
        supervise_mask = t_tokens_all > 0

        if velocity_pred.shape[0] == int(supervise_mask.sum().item()):
            # Build per-token prediction of clean latent z0
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

                t_img = t_tokens_all[image_start_ptrs].to(z0_tokens_clean.dtype)

                # Target images: t>0
                image_is_target = t_img > 0

                # Weight only low-noise steps
                w_img = torch.zeros_like(t_img, dtype=packed_latent_clean.dtype)
                w_img[image_is_target] = torch.clamp(
                    (float(pixel_loss_max_t) - t_img[image_is_target]) / float(pixel_loss_max_t),
                    min=0.0,
                    max=1.0,
                )

                selected = w_img > 0

                if pixel_loss_debug and dist.get_rank() == 0:
                    logger.info(
                        f"[Pixel Loss Debug - Chunked] images={len(t_img)} target={int(image_is_target.sum().item())} "
                        f"selected(w>0)={int(selected.sum().item())}"
                    )

                if bool(selected.any().item()):
                    p = latent_patch_size
                    c = latent_channel
                    vae_downsample = latent_downsample // p

                    # Collect predicted latents + GT images
                    pred_latents = []
                    gt_images = []
                    weights = []

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
                        token_ptr += num_img_tokens

                    if len(pred_latents) > 0:
                        # Pad to max size
                        max_h_lat = max(z.shape[1] for z in pred_latents)
                        max_w_lat = max(z.shape[2] for z in pred_latents)
                        max_h_img = max_h_lat * vae_downsample
                        max_w_img = max_w_lat * vae_downsample

                        latent_batch = z0_tokens_hybrid.new_zeros((len(pred_latents), c, max_h_lat, max_w_lat))
                        gt_batch = padded_images.new_zeros((len(gt_images), padded_images.shape[1], max_h_img, max_w_img))
                        mask = padded_images.new_zeros((len(gt_images), 1, max_h_img, max_w_img))

                        for i, (z, x_gt, w_i) in enumerate(zip(pred_latents, gt_images, weights)):
                            H_lat, W_lat = z.shape[1], z.shape[2]
                            H_img, W_img = x_gt.shape[1], x_gt.shape[2]
                            latent_batch[i, :, :H_lat, :W_lat] = z
                            gt_batch[i, :, :H_img, :W_img] = x_gt
                            mask[i, :, :H_img, :W_img] = w_i.to(mask.dtype)

                        # ========== 关键改动：分块 VAE decode ==========
                        chunk_size = calculate_chunk_size(
                            num_images=len(pred_latents),
                            max_h_img=max_h_img,
                            max_w_img=max_w_img,
                            base_chunk_size=pixel_loss_chunk_size,
                            adaptive=pixel_loss_adaptive_chunk
                        )

                        num_chunks = (len(pred_latents) + chunk_size - 1) // chunk_size

                        if pixel_loss_debug and dist.get_rank() == 0:
                            logger.info(
                                f"[Pixel Loss Chunked] num_images={len(pred_latents)} "
                                f"resolution={max_h_img}×{max_w_img} "
                                f"megapixels={(max_h_img * max_w_img) / 1e6:.2f} "
                                f"chunk_size={chunk_size} num_chunks={num_chunks}"
                            )

                        # 累积损失分子/分母（流式计算，不存储解码图像）
                        loss_numerator = torch.tensor(0.0, device=latent_batch.device, dtype=latent_batch.dtype)
                        loss_denominator = torch.tensor(0.0, device=latent_batch.device, dtype=latent_batch.dtype)

                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = min(start_idx + chunk_size, len(pred_latents))

                            # 切分当前 chunk
                            latent_chunk = latent_batch[start_idx:end_idx]
                            gt_chunk = gt_batch[start_idx:end_idx]
                            mask_chunk = mask[start_idx:end_idx]

                            # VAE decode 当前 chunk
                            x_pred_chunk = vae_decode_fn(latent_chunk)
                            x_pred_chunk = x_pred_chunk[:, :, :max_h_img, :max_w_img]
                            gt_chunk = gt_chunk.to(x_pred_chunk.dtype)

                            # 计算 [0,1] 空间的损失
                            x_pred_chunk_01 = (x_pred_chunk * 0.5 + 0.5).clamp(0, 1)
                            gt_chunk_01 = (gt_chunk * 0.5 + 0.5).clamp(0, 1)

                            if pixel_loss_type.lower() in {"l2", "mse"}:
                                diff_chunk = (x_pred_chunk_01 - gt_chunk_01) ** 2
                            else:
                                diff_chunk = (x_pred_chunk_01 - gt_chunk_01).abs()

                            # 累积损失
                            loss_numerator += (diff_chunk * mask_chunk).sum()
                            loss_denominator += mask_chunk.sum() * x_pred_chunk_01.shape[1]

                            # 释放中间张量
                            del x_pred_chunk, x_pred_chunk_01, gt_chunk_01, diff_chunk

                            # 每 2 个 chunk 或最后一个 chunk 清理显存
                            if (chunk_idx + 1) % 2 == 0 or chunk_idx == num_chunks - 1:
                                torch.cuda.empty_cache()

                        # 计算最终损失
                        MIN_DENOM = 1e-3
                        if float(loss_denominator.item()) > MIN_DENOM:
                            if pixel_loss_debug:
                                logger.info(f"++ [Pixel Loss Normal - Chunked] rank={rank} denom={float(loss_denominator.item()):.6g}")
                            pixel = loss_numerator / loss_denominator
                        else:
                            logger.warning(f"++ [Pixel Loss Small - Chunked] rank={rank} denom={float(loss_denominator.item()):.6g} < {MIN_DENOM}, setting pixel=0")
                            pixel = torch.tensor(0.0, device=loss_numerator.device, dtype=loss_numerator.dtype)

                        if pixel_loss_debug:
                            logger.info(f"++ [Pixel Loss Exit - Chunked] rank={rank} pixel={float(pixel.item()):.6g}")
                else:
                    # No images selected
                    if pixel_loss_debug and dist.get_rank() == 0:
                        logger.info("++ [Pixel Loss Skip - Chunked] selected==0, setting pixel=0")
                    pixel = torch.tensor(0.0, device=z0_tokens_hybrid.device, dtype=z0_tokens_hybrid.dtype)

    # Final safety check
    if pixel is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        pixel_val = float(pixel.detach().item())
        is_abnormal = (not torch.isfinite(pixel).all().item()) or (pixel_val > 1.0)
        if is_abnormal:
            logger.warning(
                f"++ [Pixel Loss Final Check Abnormal - Chunked] rank={rank} "
                f"pixel={pixel_val:.6g} out of valid range [0,1], clamping to 0"
            )
            pixel = torch.tensor(0.0, device=pixel.device, dtype=pixel.dtype)

    if pixel_loss_debug and pixel is not None:
        logger.info(f"++ [Pixel Loss Shape - Chunked] pixel.shape={pixel.shape}")

    return pixel


# =============================================================================
# SSIM Loss Implementation
# =============================================================================

def _gaussian_window(window_size: int, sigma: float, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a 2D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the window (typically 11)
        sigma: Standard deviation of the Gaussian (typically 1.5)
        channel: Number of channels to replicate the window for
        device: Target device
        dtype: Target dtype

    Returns:
        Gaussian window tensor of shape (channel, 1, window_size, window_size)
    """
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # Create 2D window from 1D
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)  # (window_size, window_size)
    window_2d = window_2d / window_2d.sum()

    # Expand for all channels: (channel, 1, window_size, window_size)
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)

    return window.contiguous()


def _ssim_core(
    x: torch.Tensor,
    y: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """
    Core SSIM computation between two images.

    Args:
        x: Predicted image, shape (B, C, H, W), range [0, data_range]
        y: Ground truth image, shape (B, C, H, W), range [0, data_range]
        window: Gaussian window for convolution
        window_size: Size of the Gaussian window
        channel: Number of image channels
        data_range: Dynamic range of the images (1.0 for [0,1] normalized)
        K1: Stability constant for luminance (default 0.01)
        K2: Stability constant for contrast (default 0.03)

    Returns:
        SSIM map of shape (B, C, H', W') where H', W' are reduced by padding
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    padding = window_size // 2

    # Compute means using grouped convolution
    mu_x = F.conv2d(x, window, padding=padding, groups=channel)
    mu_y = F.conv2d(y, window, padding=padding, groups=channel)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Compute variances and covariance
    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channel) - mu_xy

    # Clamp variances to avoid negative values due to numerical precision
    sigma_x_sq = torch.clamp(sigma_x_sq, min=0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0)

    # SSIM formula
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map


def compute_ssim_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    window_size: int = 11,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """
    Compute SSIM loss for image quality assessment.

    SSIM (Structural Similarity Index) measures the perceptual difference between
    two images by considering luminance, contrast, and structure.

    This returns 1 - SSIM so that minimizing the loss maximizes SSIM.

    Args:
        x_pred: Predicted image, shape (B, C, H, W), range [0, 1]
        x_gt: Ground truth image, shape (B, C, H, W), range [0, 1]
        mask: Optional weight mask, shape (B, 1, H, W). If provided, computes
              weighted average SSIM loss.
        window_size: Size of the Gaussian window (default: 11, must be odd)
        data_range: Dynamic range of the images (default: 1.0 for normalized images)
        K1: Stability constant for luminance comparison (default: 0.01)
        K2: Stability constant for contrast comparison (default: 0.03)

    Returns:
        Scalar SSIM loss tensor (1 - mean SSIM)

    Example:
        >>> x_pred = torch.rand(2, 3, 256, 256)
        >>> x_gt = torch.rand(2, 3, 256, 256)
        >>> loss = compute_ssim_loss(x_pred, x_gt)
        >>> loss.backward()
    """
    if x_pred.dim() != 4 or x_gt.dim() != 4:
        raise ValueError(f"Expected 4D tensors (B, C, H, W), got {x_pred.dim()}D and {x_gt.dim()}D")

    if x_pred.shape != x_gt.shape:
        raise ValueError(f"Shape mismatch: x_pred {x_pred.shape} vs x_gt {x_gt.shape}")

    B, C, H, W = x_pred.shape

    # Ensure window size is valid
    if window_size % 2 == 0:
        window_size += 1
    if H < window_size or W < window_size:
        # Fall back to smaller window for small images
        window_size = min(H, W)
        if window_size % 2 == 0:
            window_size -= 1
        window_size = max(3, window_size)

    # Create Gaussian window
    sigma = 1.5
    window = _gaussian_window(window_size, sigma, C, x_pred.device, x_pred.dtype)

    # Compute SSIM map
    ssim_map = _ssim_core(
        x_pred, x_gt, window, window_size, C,
        data_range=data_range, K1=K1, K2=K2
    )

    # Apply mask if provided
    if mask is not None:
        # Ensure mask has same spatial dimensions as ssim_map
        if mask.shape[2:] != ssim_map.shape[2:]:
            # Resize mask to match ssim_map
            mask = F.interpolate(mask, size=ssim_map.shape[2:], mode='bilinear', align_corners=False)

        # Expand mask to match channels
        if mask.shape[1] == 1 and ssim_map.shape[1] > 1:
            mask = mask.expand(-1, ssim_map.shape[1], -1, -1)

        # Weighted mean
        weighted_ssim = (ssim_map * mask).sum()
        total_weight = mask.sum() * C + 1e-8
        mean_ssim = weighted_ssim / total_weight
    else:
        mean_ssim = ssim_map.mean()

    # Return 1 - SSIM as loss (lower is better)
    ssim_loss = 1.0 - mean_ssim

    # Clamp to valid range [0, 1]
    ssim_loss = torch.clamp(ssim_loss, min=0.0, max=1.0)

    return ssim_loss


def _scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale backward gradient by `scale` without changing forward value.

    Uses straight-through trick: x.detach() + scale * (x - x.detach())
    Forward: returns x. Backward: grad *= scale.

    This is critical for SSIM loss through the VAE decoder: SSIM's rational
    function (numerator / denominator with small C2=0.0009) produces gradients
    200-2000x larger than simple L1/L2 pixel losses. Without scaling, these
    oversized gradients corrupt LLM/diffusion weights during training.
    """
    return x.detach() + scale * (x - x.detach())


def compute_perceptual_loss(
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
    ssim_loss_weight: float,
    ssim_loss_max_t: float,
    # 状态标志
    is_training: bool,
    # 可选参数
    ssim_window_size: int = 11,
    # === 梯度缩放参数 ===
    ssim_grad_scale: float = 0.1,
    # === 分块 VAE decode 参数 ===
    pixel_loss_chunk_size: int = 2,
    pixel_loss_adaptive_chunk: bool = True,
) -> Optional[torch.Tensor]:
    """
    Compute SSIM-based perceptual loss with chunked VAE decode.

    This function follows the same structure as compute_pixel_loss() but uses
    SSIM instead of L1/L2 for image comparison.

    Args:
        ... (same as compute_pixel_loss) ...
        ssim_loss_weight: Weight for SSIM loss (must be > 0 to enable)
        ssim_loss_max_t: Apply SSIM only when timestep t <= this value
        ssim_window_size: Window size for SSIM computation (default: 11)

    Returns:
        SSIM loss scalar tensor, or None if conditions not met
    """
    ssim_loss = None

    # Aliases for clarity
    z0_tokens_clean = packed_latent_clean
    noise_tokens = noise
    t_tokens_all = packed_timesteps

    ssim_loss_debug = os.environ.get("SSIM_LOSS_DEBUG", "").lower() in {"1", "true", "yes", "y"}

    if (
        is_training
        and ssim_loss_weight > 0
        and ssim_loss_max_t > 0
        and padded_images is not None
        and vae_decode_fn is not None
        and patchified_vae_latent_shapes is not None
        and packed_vae_token_indexes is not None
        and packed_timesteps is not None
        and len(patchified_vae_latent_shapes) > 0
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if ssim_loss_debug:
            logger.info(f"++ [SSIM Loss Entry] rank={rank} conditions met, ssim_grad_scale={ssim_grad_scale}")

        # Predict velocity (same as pixel loss)
        velocity_pred = llm2vae(last_hidden_state[mse_loss_indexes])
        velocity_target = noise_tokens - z0_tokens_clean
        supervise_mask = t_tokens_all > 0

        if velocity_pred.shape[0] == int(supervise_mask.sum().item()):
            # Build per-token prediction of clean latent z0
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

                t_img = t_tokens_all[image_start_ptrs].to(z0_tokens_clean.dtype)
                image_is_target = t_img > 0

                # Weight only low-noise steps
                w_img = torch.zeros_like(t_img, dtype=packed_latent_clean.dtype)
                w_img[image_is_target] = torch.clamp(
                    (float(ssim_loss_max_t) - t_img[image_is_target]) / float(ssim_loss_max_t),
                    min=0.0,
                    max=1.0,
                )

                selected = w_img > 0

                if ssim_loss_debug and dist.get_rank() == 0:
                    logger.info(
                        f"[SSIM Loss Debug] images={len(t_img)} target={int(image_is_target.sum().item())} "
                        f"selected(w>0)={int(selected.sum().item())}"
                    )

                if bool(selected.any().item()):
                    p = latent_patch_size
                    c = latent_channel
                    vae_downsample = latent_downsample // p

                    # Collect predicted latents + GT images
                    pred_latents = []
                    gt_images = []
                    weights = []

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
                        token_ptr += num_img_tokens

                    if len(pred_latents) > 0:
                        # Pad to max size
                        max_h_lat = max(z.shape[1] for z in pred_latents)
                        max_w_lat = max(z.shape[2] for z in pred_latents)
                        max_h_img = max_h_lat * vae_downsample
                        max_w_img = max_w_lat * vae_downsample

                        latent_batch = z0_tokens_hybrid.new_zeros((len(pred_latents), c, max_h_lat, max_w_lat))
                        gt_batch = padded_images.new_zeros((len(gt_images), padded_images.shape[1], max_h_img, max_w_img))
                        mask = padded_images.new_zeros((len(gt_images), 1, max_h_img, max_w_img))

                        for i, (z, x_gt, w_i) in enumerate(zip(pred_latents, gt_images, weights)):
                            H_lat, W_lat = z.shape[1], z.shape[2]
                            H_img, W_img = x_gt.shape[1], x_gt.shape[2]
                            latent_batch[i, :, :H_lat, :W_lat] = z
                            gt_batch[i, :, :H_img, :W_img] = x_gt
                            mask[i, :, :H_img, :W_img] = w_i.to(mask.dtype)

                        # Chunked VAE decode
                        chunk_size = calculate_chunk_size(
                            num_images=len(pred_latents),
                            max_h_img=max_h_img,
                            max_w_img=max_w_img,
                            base_chunk_size=pixel_loss_chunk_size,
                            adaptive=pixel_loss_adaptive_chunk
                        )

                        num_chunks = (len(pred_latents) + chunk_size - 1) // chunk_size

                        if ssim_loss_debug and dist.get_rank() == 0:
                            logger.info(
                                f"[SSIM Loss Chunked] num_images={len(pred_latents)} "
                                f"resolution={max_h_img}×{max_w_img} "
                                f"chunk_size={chunk_size} num_chunks={num_chunks}"
                            )

                        # Accumulate SSIM loss
                        ssim_numerator = torch.tensor(0.0, device=latent_batch.device, dtype=latent_batch.dtype)
                        ssim_denominator = torch.tensor(0.0, device=latent_batch.device, dtype=latent_batch.dtype)

                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = min(start_idx + chunk_size, len(pred_latents))

                            latent_chunk = latent_batch[start_idx:end_idx]
                            gt_chunk = gt_batch[start_idx:end_idx]
                            mask_chunk = mask[start_idx:end_idx]

                            # VAE decode
                            x_pred_chunk = vae_decode_fn(latent_chunk)
                            x_pred_chunk = x_pred_chunk[:, :, :max_h_img, :max_w_img]
                            # Scale gradients to prevent SSIM gradient explosion through VAE
                            if ssim_grad_scale < 1.0:
                                x_pred_chunk = _scale_gradient(x_pred_chunk, ssim_grad_scale)
                            gt_chunk = gt_chunk.to(x_pred_chunk.dtype)

                            # Normalize to [0, 1]
                            x_pred_chunk_01 = (x_pred_chunk * 0.5 + 0.5).clamp(0, 1)
                            gt_chunk_01 = (gt_chunk * 0.5 + 0.5).clamp(0, 1)

                            # Compute SSIM loss for this chunk
                            chunk_ssim_loss = compute_ssim_loss(
                                x_pred_chunk_01,
                                gt_chunk_01,
                                mask=mask_chunk,
                                window_size=ssim_window_size,
                            )

                            # Accumulate weighted by chunk size
                            chunk_weight = mask_chunk.sum()
                            ssim_numerator += chunk_ssim_loss * chunk_weight
                            ssim_denominator += chunk_weight

                            # Cleanup
                            del x_pred_chunk, x_pred_chunk_01, gt_chunk_01

                            if (chunk_idx + 1) % 2 == 0 or chunk_idx == num_chunks - 1:
                                torch.cuda.empty_cache()

                        # Final SSIM loss
                        MIN_DENOM = 1e-3
                        if float(ssim_denominator.item()) > MIN_DENOM:
                            ssim_loss = ssim_numerator / ssim_denominator
                        else:
                            if ssim_loss_debug:
                                logger.warning(f"++ [SSIM Loss Small Denom] rank={rank} denom={float(ssim_denominator.item()):.6g}")
                            ssim_loss = torch.tensor(0.0, device=ssim_numerator.device, dtype=ssim_numerator.dtype)

                        if ssim_loss_debug:
                            logger.info(f"++ [SSIM Loss Exit] rank={rank} ssim_loss={float(ssim_loss.item()):.6g}")
                else:
                    if ssim_loss_debug and dist.get_rank() == 0:
                        logger.info("++ [SSIM Loss Skip] selected==0, setting ssim_loss=0")
                    ssim_loss = torch.tensor(0.0, device=z0_tokens_hybrid.device, dtype=z0_tokens_hybrid.dtype)

    # Final safety check
    if ssim_loss is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        ssim_val = float(ssim_loss.detach().item())
        is_abnormal = (not torch.isfinite(ssim_loss).all().item()) or (ssim_val > 1.0)
        if is_abnormal:
            logger.warning(
                f"++ [SSIM Loss Final Check Abnormal] rank={rank} "
                f"ssim_loss={ssim_val:.6g} out of valid range [0,1], clamping to 0"
            )
            ssim_loss = torch.tensor(0.0, device=ssim_loss.device, dtype=ssim_loss.dtype)

    return ssim_loss
