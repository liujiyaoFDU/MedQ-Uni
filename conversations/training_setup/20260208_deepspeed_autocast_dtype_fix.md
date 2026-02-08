# DeepSpeed bf16 + torch.amp.autocast Dtype Mismatch Fix

**Date**: 2026-02-08
**Status**: Fixed, training running

## Problem

DeepSpeed ZeRO-2 training crashes immediately at the first forward pass:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

Crash location: `TimestepEmbedder.mlp` (first `nn.Linear` layer)
- Input: float32 (from `timestep_embedding()` which uses `.float()` for numerical precision)
- Weight: bf16 (DeepSpeed bf16 mode converts all model parameters to bf16)

## Root Cause

DeepSpeed WARNING revealed the issue:

```
[WARNING] [torch_autocast.py:122:autocast_if_enabled]
torch.autocast is enabled outside DeepSpeed but disabled within the DeepSpeed engine.
If you are using DeepSpeed's built-in mixed precision, the engine will follow the
settings in bf16/fp16 section. To use torch's native autocast instead, configure
the `torch_autocast` section in the DeepSpeed config.
```

The problem chain:

1. Training script (`main_sr_pixel_loss_deepspeed.py:1186`) wraps forward pass with `torch.amp.autocast("cuda", dtype=torch.bfloat16)`
2. DeepSpeed engine's `forward()` **disables** this external autocast context
3. Model parameters are already bf16 (DeepSpeed `bf16.enabled=True` converts them)
4. `TimestepEmbedder.timestep_embedding()` creates float32 tensors (explicit `.float()` call for numerical precision of sinusoidal embeddings)
5. Without autocast, `nn.Linear` does NOT auto-cast float32 input to match bf16 weights
6. `F.linear(float32_input, bf16_weight)` -> crash

**Why FSDP doesn't have this problem**: FSDP does not disable `torch.amp.autocast`. The autocast context propagates through the entire forward pass, and `nn.Linear` automatically casts float32 inputs to bf16.

## Fix

**File changed**: `train/deepspeed_utils.py` (DeepSpeed config only, no model code changes)

Added `torch_autocast` section to the DeepSpeed config in `generate_deepspeed_config()`:

```python
ds_config = {
    ...
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
    ...
}
```

This tells DeepSpeed to preserve the external `torch.amp.autocast` context instead of disabling it, making behavior consistent with FSDP training.

## What This Means

| Component | Before Fix | After Fix |
|-----------|-----------|-----------|
| DeepSpeed `bf16` | Converts params to bf16 | Same |
| External `torch.amp.autocast` | Disabled by DeepSpeed engine | Preserved |
| `nn.Linear` with float32 input | Crash (dtype mismatch) | Auto-cast to bf16 |
| Model code (`modeling/`) | Unchanged | Unchanged |

## Files Modified

- `train/deepspeed_utils.py`: Added `torch_autocast` config section (~5 lines)

## Verification

Training starts successfully and passes the first forward pass without dtype errors. Monitoring checkpoint saving behavior next.
