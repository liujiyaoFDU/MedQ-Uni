# SR/Restoration：在现有 Diffusion/Flow Loss 上加 Pixel L1/L2 以提升 PSNR/SSIM（实现记录）

日期：2025-12-25

## 1. 背景与问题

现有模型在生成/编辑/理解任务上表现不错，但在部分医学图像超分（SR）/恢复任务上，**PSNR、SSIM 指标不佳**。根因通常是：

- 训练目标以 diffusion/flow 的“噪声/速度回归”为主，偏向学习分布合理性与可生成性；
- PSNR/SSIM 更偏向“逐像素 fidelity（保真）”，需要额外的像素域约束。

目标：**在不破坏现有能力（尤其 T2I/编辑）的前提下**，在现有训练框架里加入一个“简单、可控、可开关”的像素层损失。

---

## 2. 你们当前训练的 diffusion loss 是什么（代码层面）

在 `modeling/bagel/bagel.py` 中，visual generation 的训练本质是 latent space 的 rectified-flow / velocity 回归：

- 干净 latent：`z0`（来自 VAE encode）
- 噪声：`ε ~ N(0,I)`
- timestep：`t∈(0,1)`（代码中 `sigmoid(N(0,1))`，并带 `timestep_shift`）
- 前向混合：`z_t = (1-t)·z0 + t·ε = z0 + t·(ε-z0)`
- velocity target：`v = ε - z0`
- 模型预测：`v̂_θ`
- diffusion/flow loss：`L_diff = ||v̂_θ - v||²`

这不是传统 DDPM 的纯 `ε-pred`，但**同样可以从模型输出反推对干净样本的估计**，从而接像素域 loss。

---

## 3. 方案：把像素 L1/L2 加在 `x̂0` 上（而不是加在噪声上）

关键做法：

1) 由模型输出反推干净 latent 估计：

- `ẑ0 = z_t - t·v̂_θ`

2) 用 VAE decoder 把 latent 投回像素空间：

- `x̂0 = D(ẑ0)`

3) 在像素空间对齐 GT（用于 PSNR/SSIM 的 fidelity）：

- `L_pix = ||x̂0 - x_GT||_1` 或 `||x̂0 - x_GT||²`

总损失（训练端加权）：

- `L_total = L_diff + λ_pix·L_pix`

### 为什么不会“奇怪”

梯度链路是：

`∂L_pix/∂v̂_θ = -t · (J_D)^T · ∂L_pix/∂x̂0`

直观含义：像素误差通过 decoder 反传回 latent，再乘一个 `t` 门控因子（越接近干净、像素监督越稳定）。

---

## 4. 关键安全措施：确保“对现有模型影响最小”

实现时做了两层门控，避免像素回归影响纯生成任务：

1) **timestep 门控（高 SNR 才启用）**  
仅在 `t <= pixel_loss_max_t` 时启用，并采用线性衰减权重：  
`w(t) = clamp((t_max - t)/t_max, 0, 1)`

2) **paired-only 门控（只对 restoration 样本启用）**  
仅对“同一个 sample 内同时存在 conditioning 图（t≈0）和 target 图（t>0）”的样本启用像素 loss，避免对纯 T2I 样本施加像素回归。

---

## 5. 代码改动清单（本次提交）

### 5.1 模型侧：计算 pixel loss（默认关闭）

- `modeling/bagel/bagel.py:153`
  - `Bagel.forward()` 新增参数：`padded_images`, `pixel_loss_weight`, `pixel_loss_type`, `pixel_loss_max_t`, `pixel_loss_paired_only`
  - 在已有的 velocity MSE（`mse`）之外，新增可选 `pixel` loss：
    - 反推 `ẑ0` → VAE decode → 对齐 GT 像素（L1/L2）
    - 对 variable size 图像做 padding+mask，避免 padding 区域影响 loss
  - `forward()` 返回 dict 新增字段：`pixel`

### 5.2 训练入口：CLI 参数 + loss 汇总（默认关闭）

- `train/main.py:375`
  - TrainingArguments 新增：
    - `pixel_loss_weight`（默认 0.0）
    - `pixel_loss_type`（默认 l1）
    - `pixel_loss_max_t`（默认 0.3）
    - `pixel_loss_paired_only`（默认 True）
  - 训练循环：
    - 仅当 `pixel_loss_weight>0` 时保留 `padded_images` 并把 pixel loss 参数传给模型
    - 将 `loss_dict["pixel"]` 计入总 loss：`loss += pixel * pixel_loss_weight`

### 5.3 文档：把“为何能加 L1/L2”讲清楚

- `idea_creation/improve_structure_performance/diffusion_pixel_latent_loss_medical.md:1`
  - 增补 §8：针对你们现有代码的 rectified-flow 目标，解释如何加像素 loss、梯度怎么走、为何要做 timestep/paired 门控

### 5.4 训练脚本（EyeQ1）：保持原配置，只切入口并默认开启像素 loss

- 新增：`scripts/training/train_sft_stage1_medq_unif_multinode_eyeQ1_sr_pixel_loss.sh`
  - 从 `scripts/training/train_sft_stage1_medq_unif_multinode_eyeQ1.sh` 复制
  - 修改：
    - `TRAIN_SCRIPT` 改为 `train/main_sr_pixel_loss.py`
    - torchrun 增加参数：
      - `--pixel_loss_weight 0.05`
      - `--pixel_loss_type l2`
      - `--pixel_loss_max_t 0.3`
    - 其余训练配置保持一致

> 说明：你们现在的 `train/main_sr_pixel_loss.py` 是一个“完整拷贝版 main”，同样是 `torchrun + nccl` 入口，因此可以单独使用。

---

## 6. 建议的第一组实验参数（保守）

如果目标是拉 PSNR/SSIM（偏像素一致性），建议从：

- `--pixel_loss_weight 0.05`
- `--pixel_loss_type l2`（更贴 PSNR）
- `--pixel_loss_max_t 0.3`（只约束低噪声阶段）
- `--pixel_loss_paired_only True`（避免影响纯生成）

再根据结果微调：
- PSNR/SSIM 仍低：可尝试把 `pixel_loss_weight` 提到 `0.1`，或把 `pixel_loss_max_t` 增到 `0.4`
- 生成多样性下降/过平滑：降低 `pixel_loss_weight` 或减小 `pixel_loss_max_t`

