# SSIM Loss v4: Inference dtype mismatch fix (Float vs BFloat16)

**Date**: 2026-02-07
**Status**: Fixed

## Problem

After SSIM loss integration, inference on SSIM-trained checkpoints (v3) crashes with:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

**Error location**: `siglip_navit.py:190` in `SiglipVisionEmbeddings.forward`:
```python
patch_embeds = self.patch_embedding(packed_pixel_values)
# packed_pixel_values = Float32 (from image transforms)
# self.patch_embedding.weight = BFloat16 (from checkpoint)
```

**Affected checkpoints**:
- `stage1_medq_2nodes_unif_eyeQ1_sr_ssim_loss_v3/0002000` (partial failure)
- `stage1_medq_2nodes_unif_eyeQ1_sr_ssim_loss_v3/0010000` (complete failure)

## Root Cause Analysis

**Why training works**: Training wraps the forward pass with `torch.amp.autocast("cuda", dtype=torch.bfloat16)` (`main_sr_pixel_loss.py:1251`), which auto-casts Float32 inputs to BFloat16 for Linear layers.

**Why inference fails**: `forward_cache_update_vit` in `bagel.py` is called under `@torch.no_grad` but NOT under autocast. The call chain:

```
infer_understanding_text (MedQ-Uni_run_batch_test2.py:380)
  -> inferencer.update_context_image (inferencer.py:185)
    -> model.forward_cache_update_vit (bagel.py:714)
      -> self.vit_model (siglip_navit.py:397)
        -> self.embeddings (siglip_navit.py:352)
          -> self.patch_embedding(packed_pixel_values)  <-- CRASH HERE
```

Image transforms output Float32 tensors. Model weights are BFloat16 (saved from mixed-precision training). Without autocast, `F.linear()` requires matching dtypes.

**Existing checks were insufficient**:
- `bagel.py:687-705`: Device check only (`.device`), no `.dtype` check
- `bagel.py:723-724`: Dtype check exists but AFTER the ViT call (too late)

## Fix

Two dtype casts added (4 lines total):

### 1. Primary fix: `modeling/bagel/siglip_navit.py` (SiglipVisionEmbeddings.forward)

Defensive fix at point of failure, protects ALL callers:

```python
def forward(self, packed_pixel_values, packed_flattened_position_ids):
    if packed_pixel_values.dtype != self.patch_embedding.weight.dtype:       # <-- NEW
        packed_pixel_values = packed_pixel_values.to(self.patch_embedding.weight.dtype)  # <-- NEW
    patch_embeds = self.patch_embedding(packed_pixel_values)
    ...
```

### 2. Inference safeguard: `modeling/bagel/bagel.py` (forward_cache_update_vit)

Early dtype cast in the inference-specific path:

```python
packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
if packed_vit_tokens.dtype != packed_text_embedding.dtype:             # <-- NEW
    packed_vit_tokens = packed_vit_tokens.to(packed_text_embedding.dtype)  # <-- NEW
packed_sequence = packed_text_embedding.new_zeros(...)
```

## Files Modified

| File | Change |
|------|--------|
| `modeling/bagel/siglip_navit.py` | +2 lines (dtype check + cast in `SiglipVisionEmbeddings.forward`) |
| `modeling/bagel/bagel.py` | +2 lines (dtype check + cast in `forward_cache_update_vit`) |

## Verification

- Python syntax check: PASS
- Module import (`from modeling.bagel.bagel import Bagel`): PASS
- Inference test on v3 checkpoint: PASS (user confirmed)

## Remaining: Generation Quality

After dtype fix, step-10000 checkpoint can now be evaluated. Quality assessment (PSNR/SSIM) is pending re-evaluation. Step-2000 results showed:

| Dataset | PSNR | SSIM | Status |
|---------|------|------|--------|
| AMIR MRI SR | 18.37 | 0.642 | Acceptable |
| IXI T1 multi-task | 12.11 | 0.474 | Low |
| Eye fundus restoration | 7.57 | 0.066 | Critical |

Quality investigation is a separate issue from this dtype fix.
