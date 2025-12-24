# Diffusion 医学图像 Restoration：Pixel/Latent 多层级 Loss 统一理解（基于本目录 10 篇论文）

> 目标：把“扩散 loss + 额外约束（L1/L2/感知/对抗/解剖先验等）”在**医学图像 restoration**场景下讲清楚：  
> 1) 每篇论文的任务与 loss 组成；2) pixel-level vs latent-level 的差异与取舍；3) latent loss 与 pixel/feature loss 的分配方式（机制层面）；4) 能力边界（尤其医学：避免 hallucination）。

---

## 0. 论文速览（本目录 `papers/*.pdf`）

| arXiv | 标题（缩写） | 主要任务 | 扩散发生在哪个空间 | 训练/优化中的 Loss 关键词（只列类型，不列数值） |
|---|---|---|---|---|
| 2112.10752 | LDM | 高分辨率生成；两阶段 AE+Diffusion | **Latent**（AE latent） | AE: `L_rec + L_adv + L_reg(KL/VQ)`；Diff: `||ε-εθ||²`；可条件建模 `p(z|y)` |
| 2501.01423 | VA‑VAE | 解决“重建 vs 生成”优化困境（更好 tokenizer） | **Latent**（token/latent） | 在 tokenizer 训练里加 `VF alignment loss`（特征对齐 + 自适应权重） |
| 2411.04873 | LPL | 给 LDM 加“decoder 特征感知”目标 | **Latent + Decoder‑feature** | `L_tot = L_diff + w·L_LPL`；LPL 只在高 SNR 时启用；多层特征+归一化+mask |
| 2308.12465 | InverseSR | 3D Brain MRI 超分（逆问题） | **Latent 优化**（不重新训练扩散） | 反演优化：`λ_perc·L_perc + λ_mae·L1`（在 `f∘D(·)` 的像素域上算） |
| 2308.13712 | RDDM | Residual + Noise 双扩散（生成/恢复统一） | **Pixel** | `L_res + L_ε`（可二选一或同时）；残差扩散+噪声扩散双调度 |
| 2504.00264 | DiffDenoise | 自监督医学去噪（条件扩散+稳定采样+蒸馏） | **Pixel** | 扩散训练用 `L1(ε)`；SRDS 对称采样；KD 再用 `L1` 蒸馏加速 |
| 2504.11286 | LRformer | 轻量医学 restoration（语义可靠先验） | 非扩散（频域注意力） | 训练主损失 `L1`；另有可靠性先验融合（非 loss 权重） |
| 2410.10328 | AFP loss | MR→CT 翻译（强调局部解剖结构） | 非扩散 | `L_AFP = mean_i ||ϕ_i(x̂)-ϕ_i(x)||_1`（特征域）+ 可与 `L1` 组合/微调 |
| 2411.13548 | MGHF | SR 的高频感知 loss 框架 | 非扩散（作为可插拔损失） | INN 特征的多项：`MGHF-n + CSC + LIP`，Gram/MSE/相关/patchNCE 等加权和 |
| 2507.20590 | HYPIR | “扩散先验初始化 + GAN 微调”加速恢复 | **Latent 流水线**（不再做多步扩散采样） | `L_adv + λ_rec·Recon(MSE+LPIPS)`；并先微调 encoder；扩散 prior 通过初始化注入 |

> 关键观察：  
> - **“diffusion loss + L1/L2”并不是唯一范式**：也可以用“扩散 prior（权重初始化/score prior）+ GAN/重建 loss”把扩散当作强先验而不是训练目标。  
> - **pixel / latent / feature** 是三层空间：像素域最强约束但最贵；latent 最省但受 tokenizer 影响；feature（decoder/seg）介于两者，常用于“结构/感知”补偿。

---

## 1. 统一符号：Restoration 与 Diffusion 的“两个层面”

### 1.1 Restoration（逆问题）抽象

- 干净图像：`x`（医学：CT/MR 或其 slice/volume）
- 退化观测：`y = f(x) + n`（下采样/模糊/噪声/缺失 slice 等）
- 目标：恢复 `x̂ ≈ x`，同时避免“看起来更真但不忠实”的 hallucination

### 1.2 Diffusion（生成/条件生成）的训练目标（核心）

离散 DDPM/LDM 常见 forward：

`z_t = √(ᾱ_t) z_0 + √(1-ᾱ_t) ε,  ε ~ N(0, I)`  

模型常预测 `εθ(z_t, t, c)`（`c` 是条件，如低质输入、文本、结构先验等），标准的简化目标：

`L_diff(θ) = E_{z0,ε,t} [ || ε - εθ(z_t, t, c) ||_p ]`  

- `p=2` 是经典（MSE）；`p=1` 在一些 restoration/robust 场景会出现（例如 DiffDenoise 的噪声回归用 L1）。
- 也可等价写为对 `x0/z0` 的重建（用 `εθ` 推回 `ẑ0`），并伴随 `λ(t)` 权重（例如 `λ_t = 1/σ_t²`）。

> **重要点**：扩散训练的“监督信号”不是最终 `x` 的像素误差，而是“跨噪声级别的去噪方向/score”。  
> 你把扩散当作“学习先验/条件分布”的方式，而不是单纯回归网络。

---

## 2. Pixel-level vs Latent-level：到底差在哪？

### 2.1 Pixel-level（像素域扩散/损失）

典型特征：
- 网络直接在 `x`（或残差）上做 diffusion，loss 直接对像素算（RDDM、DiffDenoise）
- 优点：不受 tokenizer 信息瓶颈影响；对医学细小结构更“直接”
- 缺点：计算/显存成本高（尤其 3D）；多尺度高频建模更难，训练更慢

### 2.2 Latent-level（AE latent 上扩散/损失）

典型特征：
- 先训练 autoencoder：`z=E(x)`，再在 `z` 上训练 diffusion（LDM）
- 优点：效率高；容易扩展到更大分辨率/更强 backbone
- 缺点：**tokenizer 的“压缩/重建特性”变成上限**：  
  - 细小病灶/薄层结构可能在 `E` 的下采样/通道瓶颈中被“抹掉”或被“重编码为别的东西”  
  - 这就是 VA‑VAE 讨论的“重建 vs 生成”困境：latent 维度越高重建越好，但 diffusion 训练越难

### 2.3 Feature-level（decoder/seg 特征域）是“中间层补偿”

两类常见做法：
- **decoder 特征感知**：LPL 用 AE decoder 的多层特征做 perceptual-like loss，让 diffusion 学到“对 decoder 更友好”的 latent 结构
- **任务特征感知（医学）**：AFP loss 用分割网络的特征当作“解剖结构对齐”的监督信号

> 直觉：pixel loss 约束强但可能过平滑；latent diffusion 学先验但可能不忠实；feature loss 用“结构语义”把两者粘起来。

---

## 3. “Diffusion loss + L1/L2/感知/对抗”等怎么一起用？（训练层面的统一框架）

把所有论文里出现的“多损失”抽象成：

`L_total = L_diff  +  λ_pix·L_pix  +  λ_feat·L_feat  +  λ_lat·L_lat  +  λ_adv·L_adv  +  ...`

其中每一项的**本质角色**不同：

### 3.1 `L_diff`：学习“去噪场/score/条件分布”

以 epsilon‑pred 为例：

`L_diff = E[ ||ε - εθ(z_t,t,c)|| ]`

它的强项是建模“分布/先验”，而不是保证每个样本都严格贴合 ground truth。

### 3.2 `L_pix`：像素域 fidelity（医学里最常当“反 hallucination 锚点”）

如果你模型输出的是 `εθ`，像素域损失一般不直接对 `ε`，而是先算 `ẑ0`/`x̂0`：

`ẑ0(z_t,t) = (z_t - √(1-ᾱ_t)·εθ(z_t,t,c)) / √(ᾱ_t)`

然后：

- `L_pix = || D(ẑ0) - x ||_1` 或 `||D(ẑ0)-x||_2²`（latent diffusion）
- 或直接 `L_pix = || x̂0 - x ||`（pixel diffusion）

**关键实现点（来自 LPL 的启发）**：  
`L_pix`/`L_feat` 不一定要对所有 `t` 都开。经验上更合理的是：
- 只在**高 SNR（低噪声、接近 t=0）**时启用，因为此时 `x̂0` 才有清晰语义；  
- 否则在高噪声阶段强行做像素对齐，可能把模型推向“平均解”（过平滑）或带来梯度尺度不稳。

### 3.3 `L_feat`：结构/感知约束（decoder 特征、VGG、seg 特征、INN 特征…）

两条常见路线：

1) **decoder‑feature（LPL）**：  
设 decoder 第 `l` 层特征为 `ϕ^l = F_dec,l(z0)`，对预测 `ẑ0` 得 `ϕ̂^l = F_dec,l(ẑ0)`，构造加权距离（含归一化/掩码）：

`L_LPL = Σ_l ω_l · d( Norm(ϕ^l), Norm(ϕ̂^l) )`  
`L_total = L_diff + w_LPL·L_LPL`

2) **解剖特征（AFP）**：  
用预训练分割网络的多层特征 `ϕ_i(·)`：

`L_AFP(x̂,x) = (1/N) Σ_i || ϕ_i(x̂) - ϕ_i(x) ||_1`

这类 loss 的优点是：它直接把“医学上重要的结构”编码进训练信号里；缺点是：它把分割网络的偏置也带进来（见 §5）。

### 3.4 `L_lat`：latent 正则/对齐（tokenizer 层面的“让 latent 更可学”）

VA‑VAE 的核心不是改 diffusion loss，而是改 tokenizer 的 latent 空间几何：

- 投影：`Z' = W Z`
- 点对点对齐（marginal cosine similarity）：
  `L_mcos = mean_{i,j} ReLU(1 - m1 - cos(z'_{ij}, f_{ij}))`
- 分布结构对齐（distance matrix similarity）：
  `L_mdms = (1/N²) Σ_{i,j} ReLU( |cos(z_i,z_j)-cos(f_i,f_j)| - m2 )`
- 自适应权重（梯度范数比）：
  `w_adaptive = ||∇L_rec|| / ||∇L_vf||`  
  `L_vf = w_hyper · w_adaptive · (L_mcos + L_mdms)`

> 直觉：当 latent 维度变大时，diffusion 要在更高维空间学 score，会变慢/不稳。  
> VA‑VAE 用“对齐到 foundation features”把 latent 限制在更“可学/更不发散”的子空间里，从源头缓解“重建 vs 生成”的矛盾。

### 3.5 `L_adv`：让输出“像真图”（但医学里最需要小心）

HYPIR 把 diffusion prior 注入到初始化，然后用 GAN 目标快速得到单步 restoration：

`min_θ max_ϕ  L_adv(θ,ϕ) + λ_rec·Recon(U_θ(y), x)`

其中 `Recon` 实践里常是 `MSE + LPIPS` 的组合（像素+感知）。

> 对抗/感知项会显著提高“质感”，但在医学中可能提高 hallucination 风险；必须有强 fidelity 锚点（像素/物理一致性/不确定性估计/下游任务一致性等）。

---

## 4. 论文里的“分配/权衡”机制：不是只有手调 λ

你问的“latent loss 和 pixel loss 怎么分配”，在这些论文里出现了几类**机制化答案**：

1) **梯度尺度对齐（VA‑VAE）**：用 `||∇L_a||/||∇L_b||` 让不同 loss 在同一层产生同量级影响，减少 λ 的调参范围。  
2) **只在特定噪声区间启用（LPL）**：对 `t` 做门控（高 SNR 才算特征感知），避免“高噪声阶段对齐像素/特征”的副作用。  
3) **分阶段训练/后训练（LPL、医学里常见实践）**：先把 `L_diff` 学稳（先验/条件分布），再用 `L_pix/L_feat` 做“最后一公里”的锐化/结构对齐。  
4) **目标选择/解耦（RDDM）**：把“确定性残差”和“随机噪声”拆成两套 diffusion 与 loss（`L_res` vs `L_ε`），甚至可以自动选择训练目标。  
5) **把扩散当 prior 而不是 loss（InverseSR / HYPIR）**：扩散提供的是“可逆的生成空间/强初始化”，真正的优化目标是数据一致性（L1/perceptual）或 GAN 目标。

---

## 5. 医学 restoration 的能力边界（务必在脑子里有“护栏”）

### 5.1 “看起来更清晰”≠“更真实/更可诊断”

- `L_adv / perceptual / 纹理增强` 往往提升视觉主观质量，但可能**改写病灶细节**。  
- 医学任务里更安全的思路通常是：
  - 明确数据一致性（物理模型/退化模型）或强 pixel fidelity（至少在 ROI）
  - 结构约束来自**任务模型**（AFP/seg feature）或不确定性估计（例如多次 dropout 得到可靠性）

### 5.2 Latent diffusion 的“tokenizer 上限”

如果 AE 的下采样/通道容量不足：
- 细小结构可能编码不到 `z`，后续 diffusion 再强也“凭空造不回真相”。  
缓解方向（与这些论文一致）：
- 提升 tokenizer 重建能力（更大 latent 容量）但要处理训练困难（VA‑VAE 的对齐正则）
- 采用 decoder‑feature 约束让 latent 更“可解码”（LPL）
- 直接 pixel-level（但算力成本高）

### 5.3 Feature loss 的“偏置注入”

AFP/seg 特征 loss 的边界：
- 它会把分割模型的 domain bias、漏检/误检偏好写进生成器；  
- 对 unseen 病种/设备/序列，可能出现“把分割网络喜欢的形态强行拉出来”的现象。

### 5.4 自监督去噪（DiffDenoise）在医学里要警惕什么

自监督通常依赖噪声独立性/对称性/条件强度等假设。真实医学噪声可能：
- 空间相关、非零均值、与结构相关（例如重建伪影），这会让 score 方向不再“指向干净流形”。

---

## 6. 你要做“医学 diffusion restoration + 额外 L1/L2 约束”时的落地模板（高层实现）

下面给一个不绑定某一篇论文、但能覆盖它们方法论的“可实现模板”：

### 6.1 选择建模空间

- 如果你极度关注小病灶/薄结构且算力允许：优先 **pixel diffusion** 或者在 ROI 走 pixel 分支  
- 如果你要上大分辨率/3D/更强 backbone：优先 **latent diffusion**，同时必须认真对待 tokenizer（VA‑VAE/LPL 思路）

### 6.2 训练目标组合（推荐从“最稳”开始）

1) 先只训 `L_diff`（条件扩散 `p(x|y)` 或 latent 上 `p(z|cond)`）  
2) 再加一项“保真锚点”：
   - `L_pix`：对 `x̂0`（或 `D(ẑ0)`）做 L1/L2  
   - **只在低噪声 timesteps 启用** 或随 SNR 逐步加大权重  
3) 若仍缺结构一致性：
   - `L_feat`：decoder-feature（LPL）或 seg-feature（AFP），同样建议只在低噪声阶段启用  
4) 若追求单步推理且能承受风险：
   - 用 diffusion 模型初始化，再 GAN 微调（HYPIR），但必须强约束 ROI/一致性评估

### 6.3 关键“不要踩坑”的实现细节

- **不要把 L1/L2 直接加在高噪声的 `x̂0` 上当作主监督**：高噪声时 `x̂0` 估计方差大，易把网络推向平均解。  
- **多 loss 的尺度要控制**：要么手调 λ，要么用梯度范数比（VA‑VAE 风格）做自适应。  
- **医学评估要包含“结构/任务一致性”**：只看 PSNR/SSIM 可能掩盖病灶被改写的问题。

---

## 7. 你现在这组论文给到的“组合拳”结论（一句话版）

- **想要 latent diffusion 又不丢细节**：tokenizer 要强且可学（VA‑VAE），训练时用 decoder‑feature loss 把 diffusion 和 decoder 对齐（LPL）。  
- **想要最忠实的医学 restoration**：像素锚点（L1/L2/物理一致性）必须有；结构敏感区域再用 AFP/MGHF 这类特征/高频约束增强。  
- **想要快**：把扩散当 prior 初始化，再用 GAN/重建 loss 微调成单步网络（HYPIR），但要用更严格的医学安全评估护栏。

---

## 8. 对照你们当前代码：Diffusion/Rectified‑Flow loss 里怎么“合理加 L1/L2 像素 loss”？

这一节回答你提的困惑：**模型训练目标是“预测噪声/速度（diffusion loss）”，那 L1/L2 像素 loss 到底怎么参与优化？会不会很奇怪？**

### 8.1 你们当前的“diffusion loss”到底在做什么（不是传统 DDPM 的 ε‑pred）

在 `modeling/bagel/bagel.py` 里，visual generation 的训练是一个 **latent space 的 rectified‑flow/velocity 回归**：

- 先拿到干净 latent token：`z0`（来自 VAE encoder）
- 采样噪声 token：`ε ~ N(0,I)`
- 采样 `t ∈ (0,1)`（代码里是 logit-normal：`sigmoid(N(0,1))`，再做 timestep shift）
- 前向混合（线性插值）：

`z_t = (1-t)·z0 + t·ε = z0 + t·(ε - z0)`

- 训练网络回归的不是 `ε` 本身，而是 **velocity**：

`v := ε - z0`

模型输出 `v̂_θ`，并用 MSE 训练：

`L_diff = || v̂_θ - v ||²`

这在很多论文里会被叫做 velocity prediction / rectified flow（你们的代码也引用了相关训练技巧）。

### 8.2 “像素 L1/L2”应当加在什么变量上？——加在 `x̂0`，而不是直接加在 `ε` 上

关键点：**像素域的监督对象应该是“你最终想要的干净图像”，而不是噪声。**

在 diffusion/flow 训练里，只要你能从模型输出反推出“当前 step 的干净估计”，就能把像素 loss 接上去：

1) 由 `v̂_θ` 得到对干净 latent 的估计（由上式直接变形）：

`ẑ0 = z_t - t·v̂_θ`

2) 把 `ẑ0` 过 VAE decoder 投到像素空间（这一步就是 “projector”，对应 LPL 的思路：用 decoder 把 latent 和像素结构对齐）：

`x̂0 = D(ẑ0)`

3) 然后才是你熟悉的像素损失：

`L_pix = || x̂0 - x_GT ||_1`  或  `|| x̂0 - x_GT ||²`

最终联合训练：

`L_total = L_diff + λ_pix · L_pix`

> 直觉上：  
> - `L_diff` 教模型“在不同噪声水平下应该往哪里走”（学先验/条件分布/去噪场）。  
> - `L_pix` 把“往哪里走”钉在**你关心的 fidelity 指标**上（PSNR/SSIM 本质偏向像素一致性）。

### 8.3 这个像素 loss 在反向传播里怎么“影响 diffusion 目标”？（不奇怪，走链式法则）

因为 `x̂0 = D(z_t - t·v̂_θ)`，所以：

`∂L_pix/∂v̂_θ = -t · (J_D)^T · ∂L_pix/∂x̂0`

含义非常直观：
- decoder 的雅可比 `J_D` 把像素误差投回 latent；
- `t` 是一个天然的“门控系数”：越接近纯噪声（t 大），像素监督越不稳定；越接近干净（t 小），像素监督越靠谱。

这也是为什么论文 `2411.04873 (LPL)` 会强调：**只在高 SNR / 低噪声阶段启用 perceptual/pixel 类的额外损失**。

### 8.4 “怎么分配 diffusion loss vs pixel loss 的权重？”——优先用“门控/分段”，再谈 λ

你们现在遇到的现象（生成/编辑还行，但 SR 的 PSNR/SSIM 不好）非常典型：  
扩散/flow loss 更偏“分布合理”，而 PSNR/SSIM 更偏“逐像素对齐”。

一个**对现有能力影响最小**的做法是：

1) **只在低噪声 timesteps 用像素 loss**  
例如只在 `t ≤ t_max` 时启用，或使用线性 ramp：  
`w(t) = max(0, (t_max - t)/t_max)`  
这等价于把像素监督集中在“快收敛到 x0 的最后阶段”，不去干扰模型在高噪声阶段学先验。

2) **只对“paired restoration”样本启用**  
也就是样本里同时有 conditioning 图（loss=0, t=0）和 target 图（loss=1, t>0）的那类任务（SR/denoise 等），避免把 pixel regression 施加到纯 T2I 生成数据上。

3) λ 的选择策略（从保守到激进）
   - 从很小开始（例如 `λ_pix = 0.01 ~ 0.1` 量级），看 PSNR/SSIM 是否上升、生成多样性是否下降；  
   - 如果 λ 很难调，可以参考 `2501.01423 (VA‑VAE)` 的思想用“梯度范数比”做自适应权重，让不同 loss 的梯度量级对齐（更复杂，但更省调参）。

### 8.5 我已经把这个方案落到你们 repo（默认不启用，不影响现有训练）

- 代码实现：`modeling/bagel/bagel.py:153` 起的 `Bagel.forward()` 新增了可选 pixel loss，逻辑是：
  - 由 `v̂_θ` 还原 `ẑ0`；
  - decode 得 `x̂0`；
  - 对 **paired** 且 **t≤t_max** 的目标图计算 L1/L2；
  - 返回 `pixel` loss（不乘权重）。
- 训练入口接线：`train/main.py:375` 新增 CLI 参数，并在训练时把 `pixel` 加到总 loss：
  - `--pixel_loss_weight`
  - `--pixel_loss_type`
  - `--pixel_loss_max_t`
  - `--pixel_loss_paired_only`

> 你可以把它当成一个“非常轻量的约束项”：不开启时完全不走这条分支；开启时只影响 paired restoration 的低噪声阶段。

#### 8.5.1 为了不和默认训练入口混淆：新增 SR 专用入口

如果你希望“默认训练脚本保持不变、SR 实验单独入口”，可以直接使用：

- `train/main_sr_pixel_loss.py:1`：这是 `train/main.py` 的薄封装，会自动注入一组保守默认值  
  `--pixel_loss_weight 0.05 --pixel_loss_type l2 --pixel_loss_max_t 0.3 --pixel_loss_paired_only True`  
  你仍然可以在命令行显式覆盖这些值。
