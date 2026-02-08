# VAE Decoder Gradient Checkpointing 技术文档

**日期**: 2025-12-31
**问题**: 训练中 VAE decode 阶段 OOM（即使使用 chunked decode）
**解决方案**: Gradient Checkpointing for VAE Decoder

---

## 1. 问题背景

### 1.1 现象

训练脚本 `2025_12_21_gpu_inf_s1_pixel_l2_loss_v2.sh` 在启用 pixel loss 后，即使实现了 chunked VAE decode（`chunk_size=1`），仍然在 VAE decode 阶段 OOM。

```
[rank6]: torch.cuda.OutOfMemoryError: CUDA out of memory.
[rank6]:   File "modeling/bagel/losses.py", line 806, in compute_pixel_loss
[rank6]:     x_pred_chunk = vae_decode_fn(latent_chunk)
[rank6]:   File "modeling/autoencoder.py", line 252, in forward
[rank6]:     h = self.up[i_level].upsample(h)
```

### 1.2 根本原因

**Chunking 只解决了"批量图片数量"问题，没有解决"单张图片 VAE decode 激活内存"问题。**

原因分析：
1. Pixel loss 需要梯度从 `x_pred` 流回 `llm2vae` 模块
2. 虽然 VAE 是 frozen 的，但反向传播的链式法则仍需保留 VAE decoder 的中间激活
3. VAE decoder 包含多次 upsample（分辨率翻倍），中间激活的显存消耗巨大

```
梯度流向：
llm_output → llm2vae → z_pred → VAE.decode() → x_pred → pixel_loss
                ↑
        需要更新的参数
```

### 1.3 内存估算

VAE 配置: `ch=128, ch_mult=[1, 2, 4, 4]`

对于 64×64 latent → 512×512 像素图像：
- 中间层最大分辨率: 512×512
- 中间层最大通道数: 128×4 = 512
- **单层激活**: 512×512×512 × 4 bytes (fp32) ≈ **512 MB**
- **多层 + 梯度**: 几 GB 显存

---

## 2. 解决方案：Gradient Checkpointing

### 2.1 原理

Gradient Checkpointing 是一种**用时间换空间**的技术：
- **前向传播**：不保存中间激活，只保存输入
- **反向传播**：重新计算中间激活，然后计算梯度

```python
# 原始（保存所有激活）
h1 = layer1(h0)  # 保存 h1
h2 = layer2(h1)  # 保存 h2
h3 = layer3(h2)  # 保存 h3
# backward: 使用保存的 h1, h2, h3 计算梯度

# Checkpointing（不保存中间激活）
h3 = checkpoint(layer1 + layer2 + layer3, h0)
# backward: 重新计算 h1, h2，然后计算梯度
```

### 2.2 显存节省

| 场景 | 原始显存 | Checkpoint 后 | 节省比例 |
|------|---------|--------------|---------|
| 512×512 图像 | ~3 GB | ~1 GB | ~67% |
| 1024×1024 图像 | ~10 GB | ~3 GB | ~70% |

### 2.3 性能开销

- **额外计算**: 高分辨率层需要重新计算一次
- **速度影响**: VAE decode 慢 ~20-30%
- **训练精度**: 完全不受影响（数学等价）

---

## 3. 实现细节

### 3.1 修改文件

**文件 1**: `modeling/autoencoder.py`

```python
# 添加 import
from torch.utils.checkpoint import checkpoint as grad_checkpoint

class Decoder(nn.Module):
    # 新增辅助方法
    def _forward_level(self, h: Tensor, i_level: int) -> Tensor:
        """处理单个分辨率级别的前向传播"""
        for i_block in range(self.num_res_blocks + 1):
            h = self.up[i_level].block[i_block](h)
            if len(self.up[i_level].attn) > 0:
                h = self.up[i_level].attn[i_block](h)
        if i_level != 0:
            h = self.up[i_level].upsample(h)
        return h

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            # 对高分辨率层 (i_level <= 1) 使用 gradient checkpointing
            if i_level <= 1 and h.requires_grad:
                h = grad_checkpoint(
                    self._forward_level,
                    h,
                    i_level,
                    use_reentrant=False
                )
            else:
                h = self._forward_level(h, i_level)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
```

**文件 2**: `scripts/2025_12_21_gpu_inf_s1_pixel_l2_loss_v2.sh`

```bash
# 固定 chunk_size=1，关闭自适应
PIXEL_LOSS_CHUNK_SIZE="${PIXEL_LOSS_CHUNK_SIZE:-1}"
PIXEL_LOSS_ADAPTIVE_CHUNK="${PIXEL_LOSS_ADAPTIVE_CHUNK:-false}"
```

### 3.2 关键设计决策

1. **只对高分辨率层 checkpoint** (`i_level <= 1`)
   - i_level=0: 最高分辨率 (512×512)，激活最大
   - i_level=1: 次高分辨率 (256×256)
   - 低分辨率层 (i_level >= 2) 激活较小，不值得 checkpoint

2. **使用 `use_reentrant=False`**
   - PyTorch 2.0+ 推荐的方式
   - 更安全，支持 `torch.autograd.grad()` 等高级操作

3. **条件判断 `h.requires_grad`**
   - 推理时不需要 checkpoint（没有反向传播）
   - 避免不必要的性能损失

---

## 4. 验证方法

### 4.1 显存监控

```bash
# 训练时监控显存
watch -n 1 nvidia-smi

# 或使用 nvitop
nvitop
```

### 4.2 梯度验证

确保 `llm2vae` 的梯度正常：

```python
# 在训练代码中添加检查
if step % 100 == 0:
    for name, param in model.named_parameters():
        if 'llm2vae' in name and param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

### 4.3 Pixel Loss 数值对比

修改前后 pixel loss 的数值应该完全一致（除了浮点误差）。

---

## 5. 总结

| 修改项 | 内容 |
|--------|------|
| `modeling/autoencoder.py` | 添加 gradient checkpointing |
| `scripts/...v2.sh` | 固定 `chunk_size=1`, `adaptive=false` |
| 显存节省 | ~50-70% |
| 速度开销 | ~20-30% 慢 |
| 训练精度 | 无影响 |

---

## 6. 相关文档

- [PyTorch Checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html)
- [分块 VAE Decode 技术文档](./20251230_2044_分块VAE_Decode解决OOM问题_技术文档.md)
