# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 Unimedvl Team
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Unimedvl Team.
# Modifications: Added intermediate layer extraction capabilities.

import copy
import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding

from tqdm import tqdm


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        freeze_vae=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift

        self.freeze_vae = freeze_vae


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig, vae_model=None):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        self.vae_model = vae_model
        if vae_model is not None:
            for param in self.vae_model.parameters():
                param.requires_grad = not config.freeze_vae

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)

            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)


            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

  

        self.config = config
        self._init_weights()


    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def vae_encode(self, images):
        if self.vae_model is not None:
            if hasattr(self.vae_model, 'module'):
                z_encoded = self.vae_model.module.encode(images)
            else:
                z_encoded = self.vae_model.encode(images)

            return z_encoded
        return None

    def vae_decode(self, latents):
        if self.vae_model is not None:
            if hasattr(self.vae_model, 'module'):
                return self.vae_model.module.decode(latents)
            else:
                return self.vae_model.decode(latents)
        return None

    def vae_encode_with_moments(self, images):
        if self.vae_model is not None:
            if hasattr(self.vae_model, 'module'):
                if hasattr(self.vae_model.module, 'encode_with_moments'):
                    z, mu, logvar = self.vae_model.module.encode_with_moments(images)
                else:
                    raise AttributeError("VAE module missing encode_with_moments method")
            else:
                if hasattr(self.vae_model, 'encode_with_moments'):
                    z, mu, logvar = self.vae_model.encode_with_moments(images)
                else:
                    raise AttributeError("VAE model missing encode_with_moments method")

            return z, mu, logvar
        else:
            raise RuntimeError("VAE model not initialized")

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_images: Optional[torch.Tensor] = None,
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        pixel_loss_weight: float = 0.0,
        pixel_loss_type: str = "l1",
        pixel_loss_max_t: float = 0.0,
        pixel_loss_paired_only: bool = True,
        extract_diffusion_features: bool = False,
        align_only: bool = False,
    ) -> torch.Tensor:

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und and vit_token_seqlens is not None:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen and padded_latent is not None:
            p = self.latent_patch_size
            total_tokens = sum(h * w for h, w in patchified_vae_latent_shapes)
            packed_latent_clean = padded_latent[0].new_zeros(
                (total_tokens, p * p * self.latent_channel),
                dtype=padded_latent[0].dtype,
                device=padded_latent[0].device
            )

            start_idx = 0
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                processed_latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                end_idx = start_idx + h * w
                packed_latent_clean[start_idx:end_idx] = processed_latent
                start_idx = end_idx

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            if self.training:
                # Training mode: use forward_train() parameters
                packed_und_token_indexes = packed_text_indexes
                if packed_vit_token_indexes is not None:
                    packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)

                extra_inputs.update(
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_vae_token_indexes,
                )
            else:
                # Inference mode: use forward_inference() parameters
                # Choose mode based on whether we have VAE tokens
                if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
                    extra_inputs.update(
                        mode="gen",
                        packed_vae_token_indexes=packed_vae_token_indexes,
                        packed_text_indexes=packed_text_indexes,
                    )
                else:
                    extra_inputs.update(
                        mode="und",
                    )

        if extract_diffusion_features or align_only:
            # Completely separate training and inference calls
            if self.training:
                # Training mode: use forward_train() parameters
                model_outputs = self.language_model(
                    packed_sequence=packed_sequence,
                    sample_lens=sample_lens,
                    packed_position_ids=packed_position_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    align_only=align_only,
                    **extra_inputs,
                )
            else:
                # Inference mode: use forward_inference() parameters
                # Note: forward_inference does NOT accept attention_mask, output_hidden_states, align_only
                # Convert sample_lens (List[int]) to tensor for forward_inference
                query_lens_tensor = torch.tensor(sample_lens, dtype=torch.int32, device=packed_sequence.device)
                model_outputs = self.language_model(
                    packed_query_sequence=packed_sequence,
                    query_lens=query_lens_tensor,
                    packed_query_position_ids=packed_position_ids,
                    packed_query_indexes=torch.arange(packed_sequence.shape[0], device=packed_sequence.device),
                    update_past_key_values=False,  # Validation doesn't need KV cache
                    **extra_inputs,
                )

            # Extract the hidden states tensor from the returned object
            if hasattr(model_outputs, 'last_hidden_state'):
                last_hidden_state = model_outputs.last_hidden_state
            elif hasattr(model_outputs, 'packed_query_sequence'):
                last_hidden_state = model_outputs.packed_query_sequence
            else:
                raise ValueError(f"model_outputs ({type(model_outputs)}) does not have 'last_hidden_state' or 'packed_query_sequence' attribute")

            diffusion_features = None
            if hasattr(model_outputs, 'diffusion_features') and model_outputs.diffusion_features is not None:
                if packed_vae_token_indexes is not None and len(packed_vae_token_indexes) > 0:
                    diffusion_features = model_outputs.diffusion_features[packed_vae_token_indexes]
                else:
                    diffusion_features = None
        else:
            # Completely separate training and inference calls
            if self.training:
                # Training mode: use forward_train() parameters
                last_hidden_state = self.language_model(
                    packed_sequence=packed_sequence,
                    sample_lens=sample_lens,
                    packed_position_ids=packed_position_ids,
                    attention_mask=attention_mask,
                    **extra_inputs,
                )
            else:
                # Inference mode: use forward_inference() parameters
                # Note: forward_inference does NOT accept attention_mask
                # Convert sample_lens (List[int]) to tensor for forward_inference
                query_lens_tensor = torch.tensor(sample_lens, dtype=torch.int32, device=packed_sequence.device)
                model_output = self.language_model(
                    packed_query_sequence=packed_sequence,
                    query_lens=query_lens_tensor,
                    packed_query_position_ids=packed_position_ids,
                    packed_query_indexes=torch.arange(packed_sequence.shape[0], device=packed_sequence.device),
                    update_past_key_values=False,  # Validation doesn't need KV cache
                    **extra_inputs,
                )
                # forward_inference returns BaseNavitOutputWithPast with packed_query_sequence attribute
                if hasattr(model_output, 'packed_query_sequence'):
                    last_hidden_state = model_output.packed_query_sequence
                elif hasattr(model_output, 'last_hidden_state'):
                    last_hidden_state = model_output.last_hidden_state
                else:
                    last_hidden_state = model_output  # Fallback for direct tensor return
            diffusion_features = None

        if align_only and diffusion_features is not None:
            result = {'diffusion_features': diffusion_features}

            if ce_loss_indexes is not None:
                packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
                result['ce'] = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
            else:
                result['ce'] = None

            result['mse'] = None
            return result

        mse = None
        pixel = None
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

            # Optional pixel-space fidelity loss (for paired restoration tasks such as SR/denoise).
            # This uses the VAE decoder as a differentiable projector: ẑ0 -> x̂0, then apply L1/L2 in pixel space.

            # ========== Pixel Loss Entry Diagnostics ==========
            pixel_loss_debug = os.environ.get("PIXEL_LOSS_DEBUG", "").lower() in {"1", "true", "yes", "y"}
            if pixel_loss_debug and self.training and pixel_loss_weight > 0:
                import logging
                import torch.distributed as dist
                logger = logging.getLogger(__name__)
                if dist.get_rank() == 0:
                    logger.info("="*60)
                    logger.info("[Pixel Loss Entry] Checking entry conditions")
                    logger.info(f"  pixel_loss_weight={pixel_loss_weight}")
                    logger.info(f"  pixel_loss_max_t={pixel_loss_max_t}")
                    logger.info(f"  padded_images is None: {padded_images is None}")
                    logger.info(f"  self.vae_model is None: {self.vae_model is None}")
                    logger.info(f"  patchified_vae_latent_shapes is None: {patchified_vae_latent_shapes is None}")
                    logger.info(f"  packed_vae_token_indexes is None: {packed_vae_token_indexes is None}")
                    logger.info(f"  packed_timesteps is None: {packed_timesteps is None}")
                    if patchified_vae_latent_shapes is not None:
                        logger.info(f"  len(patchified_vae_latent_shapes)={len(patchified_vae_latent_shapes)}")
                    logger.info("="*60)
            # ===================================================

            if (
                self.training
                and pixel_loss_weight > 0
                and pixel_loss_max_t > 0
                and padded_images is not None
                and self.vae_model is not None
                and patchified_vae_latent_shapes is not None
                and packed_vae_token_indexes is not None
                and packed_timesteps is not None
                and len(patchified_vae_latent_shapes) > 0
            ):
                # Ensure (has_mse tokens) align with mse_loss_indexes predictions.
                if packed_mse_preds.shape[0] == int(has_mse.sum().item()):
                    # Build per-token prediction of clean latent z0: ẑ0 = zt - t * v̂, with zt = (1-t)z0 + tε.
                    # Here v̂ ≈ (ε - z0) is the rectified-flow "velocity" target.
                    t_tokens = packed_timesteps[has_mse].to(packed_latent_clean.dtype)
                    zhat0_tokens = packed_latent_clean[has_mse] + t_tokens[:, None] * (target[has_mse] - packed_mse_preds)

                    packed_latent_hat0 = packed_latent_clean.clone()
                    packed_latent_hat0[has_mse] = zhat0_tokens

	                    # Compute per-image timestep (constant within an image).
                    tokens_per_image = [h * w for (h, w) in patchified_vae_latent_shapes]
                    total_tokens_expected = sum(tokens_per_image)
                    if total_tokens_expected == packed_latent_hat0.shape[0]:
                        device = packed_latent_hat0.device
                        tokens_per_image_t = torch.tensor(tokens_per_image, device=device, dtype=torch.long)
                        image_start_ptrs = torch.cat(
                            [torch.zeros(1, device=device, dtype=torch.long), tokens_per_image_t.cumsum(0)[:-1]],
                            dim=0,
                        )

                        # First timestep per image.
                        t_img = packed_timesteps[image_start_ptrs].to(packed_latent_clean.dtype)

                        # Target images are those with t>0 (conditioning images use t=0 via -inf sentinel).
                        # NOTE: paired-sample detection is intentionally omitted here; enable pixel loss only
                        # when the training data is guaranteed to be paired.
                        image_is_target = t_img > 0

                        # Weight only low-noise steps (high SNR): linear ramp w(t) = max(0, (t_max - t)/t_max).
                        w_img = torch.zeros_like(t_img, dtype=packed_latent_clean.dtype)
                        w_img[image_is_target] = torch.clamp(
                            (float(pixel_loss_max_t) - t_img[image_is_target]) / float(pixel_loss_max_t),
                            min=0.0,
                            max=1.0,
                        )

                        selected = w_img > 0

                        # Debug summary (rank 0 only).
                        if pixel_loss_debug:
                            import logging
                            import torch.distributed as dist
                            logger = logging.getLogger(__name__)
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
                            p = self.latent_patch_size
                            c = self.latent_channel
                            vae_downsample = self.latent_downsample // p

                            # Collect predicted latents + GT images for selected images (variable sizes).
                            pred_latents = []
                            gt_images = []
                            weights = []

                            token_ptr = 0
                            for img_idx, (h, w) in enumerate(patchified_vae_latent_shapes):
                                num_img_tokens = h * w
                                if bool(selected[img_idx].item()):
                                    tok = packed_latent_hat0[token_ptr : token_ptr + num_img_tokens]
                                    tok = tok.view(h, w, p, p, c)
                                    latent = torch.einsum("hwpqc->chpwq", tok).reshape(c, h * p, w * p)
                                    pred_latents.append(latent)

                                    H_img = h * self.latent_downsample
                                    W_img = w * self.latent_downsample
                                    gt_images.append(padded_images[img_idx, :, :H_img, :W_img])
                                    weights.append(w_img[img_idx])
                                token_ptr += num_img_tokens

                            if len(pred_latents) > 0:
                                # Pad latents/images to the max size for a single batched decode.
                                max_h_lat = max(z.shape[1] for z in pred_latents)
                                max_w_lat = max(z.shape[2] for z in pred_latents)
                                max_h_img = max_h_lat * vae_downsample
                                max_w_img = max_w_lat * vae_downsample

                                latent_batch = packed_latent_hat0.new_zeros((len(pred_latents), c, max_h_lat, max_w_lat))
                                gt_batch = padded_images.new_zeros((len(gt_images), padded_images.shape[1], max_h_img, max_w_img))
                                mask = padded_images.new_zeros((len(gt_images), 1, max_h_img, max_w_img))

                                for i, (z, x_gt, w_i) in enumerate(zip(pred_latents, gt_images, weights)):
                                    H_lat, W_lat = z.shape[1], z.shape[2]
                                    H_img, W_img = x_gt.shape[1], x_gt.shape[2]
                                    latent_batch[i, :, :H_lat, :W_lat] = z
                                    gt_batch[i, :, :H_img, :W_img] = x_gt
                                    mask[i, :, :H_img, :W_img] = w_i.to(mask.dtype)

                                x_pred = self.vae_decode(latent_batch)
                                x_pred = x_pred[:, :, :max_h_img, :max_w_img]
                                gt_batch = gt_batch.to(x_pred.dtype)

                                # Compute loss in [0,1] pixel space (consistent with inference saving path),
                                # which avoids extreme out-of-range decoder outputs exploding L2.
                                x_pred_01 = (x_pred * 0.5 + 0.5).clamp(0, 1)
                                gt_batch_01 = (gt_batch * 0.5 + 0.5).clamp(0, 1)

                                if pixel_loss_type.lower() in {"l2", "mse"}:
                                    diff = (x_pred_01 - gt_batch_01) ** 2
                                else:
                                    diff = (x_pred_01 - gt_batch_01).abs()

                                denom = mask.sum() * x_pred_01.shape[1]
                                if float(denom.item()) > 0:
                                    pixel = (diff * mask).sum() / denom


        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        result = dict(mse=mse, ce=ce, pixel=pixel)
        if diffusion_features is not None:
            result['diffusion_features'] = diffusion_features

        return result


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        embed_device = next(self.language_model.model.embed_tokens.parameters()).device

        if packed_text_ids.device != embed_device:
            packed_text_ids = packed_text_ids.to(embed_device)

        if packed_text_position_ids.device != embed_device:
            packed_text_position_ids = packed_text_position_ids.to(embed_device)
        if text_token_lens.device != embed_device:
            text_token_lens = text_token_lens.to(embed_device)
        if packed_text_indexes.device != embed_device:
            packed_text_indexes = packed_text_indexes.to(embed_device)
        if packed_key_value_indexes.device != embed_device:
            packed_key_value_indexes = packed_key_value_indexes.to(embed_device)
        if key_values_lens.device != embed_device:
            key_values_lens = key_values_lens.to(embed_device)

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        embed_device = next(self.language_model.model.embed_tokens.parameters()).device

        tensors_to_check = {
            'packed_text_ids': packed_text_ids,
            'packed_text_indexes': packed_text_indexes,
            'packed_vit_tokens': packed_vit_tokens,
            'packed_vit_token_indexes': packed_vit_token_indexes,
            'packed_vit_position_ids': packed_vit_position_ids,
            'vit_token_seqlens': vit_token_seqlens,
            'packed_position_ids': packed_position_ids,
            'packed_seqlens': packed_seqlens,
            'packed_indexes': packed_indexes,
            'packed_key_value_indexes': packed_key_value_indexes,
            'key_values_lens': key_values_lens,
        }

        corrected_tensors = {}
        device_corrections = 0

        for name, tensor in tensors_to_check.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != embed_device:
                corrected_tensors[name] = tensor.to(embed_device)
                device_corrections += 1
            else:
                corrected_tensors[name] = tensor

        if device_corrections > 0:
            packed_text_ids = corrected_tensors['packed_text_ids']
            packed_text_indexes = corrected_tensors['packed_text_indexes']
            packed_vit_tokens = corrected_tensors['packed_vit_tokens']
            packed_vit_token_indexes = corrected_tensors['packed_vit_token_indexes']
            packed_vit_position_ids = corrected_tensors['packed_vit_position_ids']
            vit_token_seqlens = corrected_tensors['vit_token_seqlens']
            packed_position_ids = corrected_tensors['packed_position_ids']
            packed_seqlens = corrected_tensors['packed_seqlens']
            packed_indexes = corrected_tensors['packed_indexes']
            packed_key_value_indexes = corrected_tensors['packed_key_value_indexes']
            key_values_lens = corrected_tensors['key_values_lens']

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()

        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)

            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h

            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)

            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]

        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        embed_device = next(self.language_model.model.embed_tokens.parameters()).device

        tensors_to_check = {
            'padded_images': padded_images,
            'packed_vae_position_ids': packed_vae_position_ids,
            'packed_timesteps': packed_timesteps,
            'packed_vae_token_indexes': packed_vae_token_indexes,
            'packed_text_ids': packed_text_ids,
            'packed_text_indexes': packed_text_indexes,
            'packed_position_ids': packed_position_ids,
            'packed_seqlens': packed_seqlens,
            'packed_indexes': packed_indexes,
            'key_values_lens': key_values_lens,
            'packed_key_value_indexes': packed_key_value_indexes,
        }

        corrected_tensors = {}
        device_corrections = 0

        for name, tensor in tensors_to_check.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != embed_device:
                corrected_tensors[name] = tensor.to(embed_device)
                device_corrections += 1
            else:
                corrected_tensors[name] = tensor

        if device_corrections > 0:
            padded_images = corrected_tensors['padded_images']
            packed_vae_position_ids = corrected_tensors['packed_vae_position_ids']
            packed_timesteps = corrected_tensors['packed_timesteps']
            packed_vae_token_indexes = corrected_tensors['packed_vae_token_indexes']
            packed_text_ids = corrected_tensors['packed_text_ids']
            packed_text_indexes = corrected_tensors['packed_text_indexes']
            packed_position_ids = corrected_tensors['packed_position_ids']
            packed_seqlens = corrected_tensors['packed_seqlens']
            packed_indexes = corrected_tensors['packed_indexes']
            key_values_lens = corrected_tensors['key_values_lens']
            packed_key_value_indexes = corrected_tensors['packed_key_value_indexes']

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        total_tokens = sum(h * w for h, w in patchified_vae_latent_shapes)
        packed_latent = padded_latent[0].new_zeros(
            (total_tokens, p * p * self.latent_channel),
            dtype=padded_latent[0].dtype,
            device=padded_latent[0].device
        )

        start_idx = 0
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            processed_latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            end_idx = start_idx + h * w
            packed_latent[start_idx:end_idx] = processed_latent
            start_idx = end_idx
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    
    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        embed_device = next(self.language_model.model.embed_tokens.parameters()).device

        tensors_to_check = {
            'x_t': x_t,
            'timestep': timestep,
            'packed_vae_token_indexes': packed_vae_token_indexes,
            'packed_vae_position_ids': packed_vae_position_ids,
            'packed_text_ids': packed_text_ids,
            'packed_text_indexes': packed_text_indexes,
            'packed_indexes': packed_indexes,
            'packed_position_ids': packed_position_ids,
            'packed_seqlens': packed_seqlens,
            'key_values_lens': key_values_lens,
            'packed_key_value_indexes': packed_key_value_indexes,
        }

        if cfg_text_packed_position_ids is not None:
            tensors_to_check['cfg_text_packed_position_ids'] = cfg_text_packed_position_ids
        if cfg_text_packed_query_indexes is not None:
            tensors_to_check['cfg_text_packed_query_indexes'] = cfg_text_packed_query_indexes
        if cfg_text_key_values_lens is not None:
            tensors_to_check['cfg_text_key_values_lens'] = cfg_text_key_values_lens
        if cfg_text_packed_key_value_indexes is not None:
            tensors_to_check['cfg_text_packed_key_value_indexes'] = cfg_text_packed_key_value_indexes
        if cfg_img_packed_position_ids is not None:
            tensors_to_check['cfg_img_packed_position_ids'] = cfg_img_packed_position_ids
        if cfg_img_packed_query_indexes is not None:
            tensors_to_check['cfg_img_packed_query_indexes'] = cfg_img_packed_query_indexes
        if cfg_img_key_values_lens is not None:
            tensors_to_check['cfg_img_key_values_lens'] = cfg_img_key_values_lens
        if cfg_img_packed_key_value_indexes is not None:
            tensors_to_check['cfg_img_packed_key_value_indexes'] = cfg_img_packed_key_value_indexes

        corrected_tensors = {}
        device_corrections = 0

        for name, tensor in tensors_to_check.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != embed_device:
                corrected_tensors[name] = tensor.to(embed_device)
                device_corrections += 1
            else:
                corrected_tensors[name] = tensor

        if device_corrections > 0:
            x_t = corrected_tensors['x_t']
            timestep = corrected_tensors['timestep']
            packed_vae_token_indexes = corrected_tensors['packed_vae_token_indexes']
            packed_vae_position_ids = corrected_tensors['packed_vae_position_ids']
            packed_text_ids = corrected_tensors['packed_text_ids']
            packed_text_indexes = corrected_tensors['packed_text_indexes']
            packed_indexes = corrected_tensors['packed_indexes']
            packed_position_ids = corrected_tensors['packed_position_ids']
            packed_seqlens = corrected_tensors['packed_seqlens']
            key_values_lens = corrected_tensors['key_values_lens']
            packed_key_value_indexes = corrected_tensors['packed_key_value_indexes']

            if 'cfg_text_packed_position_ids' in corrected_tensors:
                cfg_text_packed_position_ids = corrected_tensors['cfg_text_packed_position_ids']
            if 'cfg_text_packed_query_indexes' in corrected_tensors:
                cfg_text_packed_query_indexes = corrected_tensors['cfg_text_packed_query_indexes']
            if 'cfg_text_key_values_lens' in corrected_tensors:
                cfg_text_key_values_lens = corrected_tensors['cfg_text_key_values_lens']
            if 'cfg_text_packed_key_value_indexes' in corrected_tensors:
                cfg_text_packed_key_value_indexes = corrected_tensors['cfg_text_packed_key_value_indexes']
            if 'cfg_img_packed_position_ids' in corrected_tensors:
                cfg_img_packed_position_ids = corrected_tensors['cfg_img_packed_position_ids']
            if 'cfg_img_packed_query_indexes' in corrected_tensors:
                cfg_img_packed_query_indexes = corrected_tensors['cfg_img_packed_query_indexes']
            if 'cfg_img_key_values_lens' in corrected_tensors:
                cfg_img_key_values_lens = corrected_tensors['cfg_img_key_values_lens']
            if 'cfg_img_packed_key_value_indexes' in corrected_tensors:
                cfg_img_packed_key_value_indexes = corrected_tensors['cfg_img_packed_key_value_indexes']

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))

        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1

        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)

        packed_timestep_embeds = self.time_embedder(timestep)

        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed

        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)

        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}

        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )

        v_t = self.llm2vae(output.packed_query_sequence)

        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )

            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)

            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )

            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)

            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)

                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)

                v_t_text = v_t_text_ * scale

                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")

                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)

                v_t = v_t_ * scale
        else:
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        embed_device = next(self.language_model.model.embed_tokens.parameters()).device

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long, device=embed_device),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long, device=embed_device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=embed_device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=embed_device),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        embed_device = next(self.language_model.model.embed_tokens.parameters()).device

        if packed_key_value_indexes.device != embed_device:
            packed_key_value_indexes = packed_key_value_indexes.to(embed_device)
        if key_values_lens.device != embed_device:
            key_values_lens = key_values_lens.to(embed_device)
        if packed_start_tokens.device != embed_device:
            packed_start_tokens = packed_start_tokens.to(embed_device)
        if packed_query_position_ids.device != embed_device:
            packed_query_position_ids = packed_query_position_ids.to(embed_device)

        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens),
                device=key_values_lens.device,
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id:
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output
