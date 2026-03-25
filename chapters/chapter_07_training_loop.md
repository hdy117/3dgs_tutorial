# 第 7 章：训练闭环——从初始化到收敛的完整迭代

**学习路径**：`problem → starting point → invention → verification → example`

---

## 🎯 Problem：高斯有了，怎么让它们"学会"正确的场景？

想象一下：你刚刚完成了第 6 章的初始化。现在有 **N 个高斯**（可能是 10⁵-10⁶），每个都有位置μ、协方差Σ、不透明度α和颜色 c。**但它们是随机的、粗糙的**——渲染出来的图像可能一团糟。

现在你要回答几个问题：

> **Q1: 怎么让这些参数"自动调整"到正确的值？**  
> → 需要优化器（Adam）、损失函数（L1+SSIM）、反向传播...但这只是基础
   
> **Q2: 高斯数量 N 应该固定吗？如果场景某些区域太复杂怎么办？**  
> → 需要动态增删——这就是**密度控制（densification/pruning）**
   
> **Q3: 训练多久算"成功"？怎么判断收敛了还是失败了？**  
> → 需要监控指标（PSNR、损失曲线）、诊断工具

**这三个问题是本章要解决的核心。**

---

## 📍 Starting Point：从推理到训练——缺了什么？

### 你已经有了什么？（第 1-6 章）

```
推理管线（前向传播）：
输入：相机参数 K, R, t + 高斯参数 {μ, Σ, α, c}
     ↓
投影 → 排序 → Tile-based 混合渲染
     ↓
输出：渲染图像 rendered_image (H×W×3)
```

### 训练需要什么额外的东西？

对比**推理 vs 训练**：

| 推理 | 训练（新增） |
|------|-------------|
| 输入：相机参数 | 输入：相机参数 + **真实图像 gt_image** |
| 输出：渲染图 | 输出：**更新后的高斯参数** |
| 操作：前向传播 | 操作：前向 → **损失计算** → **反向传播** → **优化器更新** → **密度控制** |

**关键差异**：训练是"闭环"——渲染结果要和真实图像比较，误差驱动参数更新。

---

## 🔥 Invention：从公理到设计决策

### 第一步：找到不可约的事实（Axioms）

#### 公理 1：梯度必须稳定传递

**事实**：3DGS 渲染管线包含多个可微操作（投影、混合），但有两个"断点"：
- **排序**（argsort）是不可微的离散操作
- **密度控制**（增删高斯）是离散操作，会改变参数数量 N

**矛盾**：如果每帧都重新排序，梯度流会被打断吗？如果频繁增删高斯，优化器状态怎么办？

**解决路径**（这是 3DGS 的洞察！）：
1. **排序只在阶段 1 做一次**（全局深度排序），阶段 2-4 使用这个固定顺序
2. **密度控制每 N 步才执行一次**（比如 1000 步），期间梯度流不被打断

**关键设计**：densify/prune 使用**缓存的梯度**，不干扰当前优化器的动量状态。

---

#### 公理 2：密度控制需要"学习信号"

**问题**：什么时候增加高斯？什么时候删除？

如果你只是随机增减，那和蒙特卡洛采样有什么区别？**必须有信号**。

**3DGS 的洞察**（基于梯度的几何直觉）：

| 信号 | 含义 | 动作 |
|------|------|------|
| **梯度大** (‖∇μ‖ > threshold) | 该高斯对误差贡献大 → 需要更多表示 | **增加** |
| **投影尺度大** (radius > 10px) | 该高斯在 3D 空间太大 → 应该分裂 | **分裂（split）** |
| **梯度小 + 透明度低** (α < 0.001) | 该高斯无贡献 → 浪费显存 | **删除（prune）** |

**这个设计很漂亮**：用可微信号（梯度范数、投影半径）驱动密度调整，而不是启发式规则。

---

#### 公理 3：学习率需要阶段适应性

**观察**：参数在不同训练阶段的"距离最优值有多远"是不同的：
- **初期**（step < 7.5k）：参数远离最优 → 大 LR 快速调整
- **中期**（7.5k < step < 15k）：接近最优 → 中 LR 精细调整  
- **后期**（step > 15k）：微调阶段 → 小 LR 防止过拟合

**结论**：学习率调度应该是**阶梯式衰减**的。

---

### 第二步：推导矛盾（Contradictions）

#### 矛盾 1：优化器状态 vs 参数数量变化

**问题浮现**：
- Adam 优化器维护动量 m_t 和方差 v_t，数量与参数成正比
- densify/prune 会改变高斯数量 N
- **如果 N 变了，优化器的状态怎么办？**

让我们推导三种方案：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A | 每 densify/prune 后**重建优化器** | 简单，状态对齐 | Adam 动量丢失（需要 warm-up） |
| B | **仅增删对应参数的状态** | 保留动量，高效 | 实现复杂（索引管理麻烦） |
| C | 用 SGD（无状态） | 无需管理状态 | 收敛可能慢，手动调 LR 痛苦 |

**3DGS 的选择**：方案 A（重建优化器）。为什么？
- densify/prune **每 1000 步才一次**，开销占比 < 1%
- Adam 动量丢失的影响可以通过 warm-up 缓解
- **实现简单 > 理论最优**

---

#### 矛盾 2：梯度缓存 vs 实时性

**问题浮现**：
- densify 需要梯度信息（判断哪些高斯需要分裂/克隆）
- 但反向传播后，`param.grad` 立即可用
- **为什么需要"缓存"？不能直接用当前步的梯度吗？**

让我们分析噪声来源：

```python
# 假设 batch size = 1（单视角采样）
gt_image, camera = dataset.sample()  # 随机采样一个视角
rendered = render(gaussians, camera)
loss = L(rendered, gt_image)

# 此时 grads_mu[i] 只反映了"第 i 个高斯在当前视角下的误差贡献"
# 但如果这个高斯在其他视角下很重要呢？→ 噪声！
```

**结论**：单步梯度受 batch sampling 随机性影响大，需要缓存 N 步取平均。

**实现设计**：
```python
if step % densify_interval == 0:
    # 缓存最近一次反向传播的梯度（代表过去 1000 步的平均趋势）
    grads_mu_cache = gaussians.mu.grad.detach().norm(dim=1)  # (N,)
    radii_cache = radii.detach()  # (N,)
```

**关键洞察**：densify_interval 越大，缓存越稳定，但响应越慢。3DGS 选择 **1000 步**作为平衡点。

---

#### 矛盾 3：全局排序 vs Tile-based 并行渲染

**问题浮现**：
- Alpha blending 需要**从后往前**混合（深度顺序）
- Tile-based 渲染是**每个 tile 独立处理**（并行化优化）
- **如何保证每个 tile 内的高斯顺序是全局深度顺序的子集？**

让我们画个图理解这个问题：

```
全局高斯列表（按深度排序）: [g1, g2, g3, ..., gN] (z_ascending)
                              ↓ 投影到屏幕空间
Tile A (左上): 需要 {g1, g5, g8} → 顺序是 [g1, g5, g8] ✓
Tile B (右上): 需要 {g2, g3, g6} → 顺序是 [g2, g3, g6] ✓
...
```

**解决方案**（3DGS 的设计）：
1. **阶段 1**：全局排序一次（所有高斯按深度 argsort），得到 `sorted_gaussians`
2. **阶段 2**：将 sorted_gaussians 投影到屏幕空间，计算每个高素覆盖哪些 tiles
3. **阶段 3-4**：每个 tile 从 sorted_gaussians 中分配高斯（保持全局顺序）

**性能分析**：排序 O(N log N) ≈ 5-10ms（N=1M），相比渲染时间可接受。

---

### 第三步：设计完整训练循环（Solution Path）

#### 状态机设计

```
[初始化] → gaussians = init_from_sfm(...)
     ↓
[优化器构建] → optimizer = Adam(params, lr=...)
     ↓
┌─────────────────────────────────────┐
│ for step in range(total_steps):     │ ← 主循环
│     ├─→ 采样视角 (gt_image, camera) │
│     ├─→ 前向渲染 (rendered, radii)  │
│     ├─→ 计算损失 L_total            │
│     ├─→ 反向传播 loss.backward()    │
│     ├─→ 优化器更新 optimizer.step() │
│     ├─→ [step % 1000 == 0]          │
│     │   ├─→ 缓存梯度 grads_cache    │
│     │   ├─→ Densify（分裂/克隆）    │
│     │   └─→ Prune（删除无贡献高斯） │
│     ├─→ [step in decay_steps]       │
│     │   └─→ LR × 0.1                │
│     └─→ [step % 100 == 0]           │
│         └─→ 日志 PSNR, N, loss      │
└─────────────────────────────────────┘
     ↓
[收敛判断] → PSNR 连续 1000 步变化 < 0.1 dB?
     ↓ 是
[结束训练]
```

---

#### 核心算法（伪代码）

```python
# ============ 超参数配置 ============
config = {
    # 训练控制
    "total_steps": 30000,           # 总步数
    "log_interval": 100,            # 日志间隔
    
    # 密度控制
    "densify_interval": 1000,       # densify/prune 每 N 步执行一次
    "densify_from": 500,            # step > 500 才开始 densify（避免初期不稳定）
    "prune_from": 15000,            # step > 15k 才开始 prune（前期多保留高斯）
    
    # 密度控制阈值
    "grad_threshold": 0.0002,       # 梯度范数阈值 → densify 触发条件
    "scale_threshold": 0.01,        # 投影尺度阈值（像素）→ split vs clone
    "prune_alpha": 0.001,           # α < threshold → prune
    "max_gaussians": 2e6,           # 数量上限保护
    
    # LR 调度
    "lr_decay_steps": [7500, 15000],# 衰减点
    "lr_decay_factor": 0.1,         # 每次衰减 ×0.1
    
    # 损失权重
    "lambda_ssim": 0.8,             # SSIM 权重（L_total = (1-λ)L1 + λ·SSIM）
}

# ============ 初始化 ============
gaussians = init_from_sfm(...)  # 第 6 章：从 SFM points 初始化

optimizer = torch.optim.Adam([
    {"params": gaussians.mu, "lr": 1.6e-4},   # 位置 LR 最小（精细调整）
    {"params": gaussians.Sigma, "lr": 1e-3},  # 协方差中等 LR
    {"params": gaussians.alpha, "lr": 5e-2},  # α范围小，需要大 LR
    {"params": gaussians.color, "lr": 5e-3},  # 颜色易调，中等 LR
])

# 缓存（用于 densify）
grads_mu_cache = None
radii_cache = None

# ============ 主训练循环 ============
for step in range(config["total_steps"]):
    # --- 1. 采样视角 ---
    gt_image, camera = dataset.sample()  # (H, W, 3) + {K, R, t}
    
    # --- 2. 前向渲染 ---
    rendered, radii = render(gaussians, camera)  
    # rendered: (H, W, 3), radii: (N,) → 每个高斯在屏幕空间的投影半径（像素）
    
    # --- 3. 损失计算 ---
    L1 = F.l1_loss(rendered, gt_image)
    L_ssim = 1 - ms_ssim(rendered, gt_image)
    L_img = (1 - config["lambda_ssim"]) * L1 + config["lambda_ssim"] * L_ssim
    
    # 正则化：惩罚过大的尺度（防止高斯"爆炸"）
    scales = gaussians.get_scales()  # (N, 3)
    L_scale = torch.clamp(scales - 1.0, min=0).mean()
    
    L_total = L_img + 0.01 * L_scale
    
    # --- 4. 反向传播 ---
    optimizer.zero_grad()
    L_total.backward()  # grads_mu[i] ← ∂L/∂μ_i, grads_Sigma[i] ← ∂L/∂Σ_i
    
    # --- 5. 缓存梯度（用于 densify）---
    if step % config["densify_interval"] == 0:
        grads_mu_cache = gaussians.mu.grad.detach().norm(dim=1)  # (N,)
        radii_cache = radii.detach()  # (N,)
    
    # --- 6. 优化器更新 ---
    optimizer.step()
    
    # --- 7. 密度控制（每 1000 步）---
    if step % config["densify_interval"] == 0 and step >= config["densify_from"]:
        densify_and_prune(gaussians, grads_mu_cache, radii_cache, optimizer, config)
    
    # --- 8. LR 调度 ---
    if step in config["lr_decay_steps"]:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= config["lr_decay_factor"]
    
    # --- 9. 日志 ---
    if step % config["log_interval"] == 0:
        psnr = 10 * torch.log10(1.0 / L1.item())
        print(f"Step {step}: PSNR={psnr:.2f}, Loss={L_total:.4f}, "
              f"#gaussians={len(gaussians)}, LR={optimizer.param_groups[0]['lr']:.2e}")

# ============ 辅助函数：密度控制 ============
def densify_and_prune(gaussians, grads_mu_cache, radii_cache, optimizer, config):
    """每 1000 步执行一次：densify（分裂/克隆）+ prune（删除）"""
    
    # --- Densify: 增加高斯 ---
    if step < config["prune_from"]:  # 前期只 densify，不 prune
        # 条件：投影尺度大 + 梯度大 → 该区域需要更多表示
        scale_condition = radii_cache > config["scale_threshold"]
        grad_condition = grads_mu_cache > config["grad_threshold"]
        
        densify_mask = scale_condition & grad_condition
        indices_to_densify = torch.where(densify_mask)[0]
        
        new_gaussians = []
        for i in indices_to_densify:
            # 判断：尺度大 → split；尺度小 → clone
            if radii_cache[i] > config["scale_threshold"]:
                g1, g2 = split_gaussian(gaussians, i)  # 分裂成两个
                new_gaussians.extend([g1, g2])
            else:
                g_new = clone_gaussian(gaussians, i)   # 克隆一个
                new_gaussians.append(g_new)
        
        if new_gaussians:
            gaussians.extend(new_gaussians)
            optimizer = rebuild_optimizer(gaussians, config)  # N 变了，重建优化器
    
    # --- Prune: 删除高斯 ---
    if step >= config["prune_from"]:  # 后期开启 prune
        # 条件：α太小（透明）或 尺度太小（退化）
        alpha_mask = gaussians.squeeze_alpha() < config["prune_alpha"]
        scales = gaussians.get_scales().max(dim=1)[0]
        scale_mask = scales < 1e-6
        
        prune_mask = alpha_mask | scale_mask
        
        if prune_mask.any():
            gaussians = gaussians[~prune_mask]  # 删除被 mask 的高斯
            optimizer = rebuild_optimizer(gaussians, config)
    
    # --- 数量上限保护 ---
    if len(gaussians) > config["max_gaussians"]:
        n_remove = len(gaussians) - int(config["max_gaussians"])
        indices_to_keep = torch.randperm(len(gaussians))[n_remove:]
        gaussians = gaussians[indices_to_keep]
        optimizer = rebuild_optimizer(gaussians, config)
    
    return gaussians, optimizer

def split_gaussian(gaussians, idx):
    """分裂高斯：沿主方向分成两个"""
    Sigma = gaussians.Sigma[idx]  # (3, 3)
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    
    # 主方向（最大特征值对应的特征向量）
    main_dir = eigvecs[:, -1]  # (3,)
    
    # 新尺度 = √(λ_i / 2) → 保持总体积不变
    new_scales = torch.sqrt(eigvals / 2.0)
    new_Sigma = eigvecs @ torch.diag(new_scales**2) @ eigvecs.T
    
    # 新位置 = μ ± 0.01·主方向（微小偏移）
    offset = 0.01 * main_dir
    mu1 = gaussians.mu[idx] + offset
    mu2 = gaussians.mu[idx] - offset
    
    return Gaussian(mu1, new_Sigma, gaussians.alpha[idx], gaussians.color[idx]), \
           Gaussian(mu2, new_Sigma, gaussians.alpha[idx], gaussians.color[idx])

def clone_gaussian(gaussians, idx):
    """克隆高斯：直接复制"""
    return Gaussian(
        mu=gaussians.mu[idx].clone(),
        Sigma=gaussians.Sigma[idx].clone(),
        alpha=gaussians.alpha[idx].clone(),
        color=gaussians.color[idx].clone()
    )

def rebuild_optimizer(gaussians, config):
    """重建 Adam 优化器（参数数量变化后）"""
    # 注意：这里会丢失动量状态，但 densify_interval=1000 步时影响可接受
    return torch.optim.Adam([
        {"params": gaussians.mu, "lr": get_current_lr(config)},
        {"params": gaussians.Sigma, "lr": get_current_lr(config) * 6.25},
        {"params": gaussians.alpha, "lr": get_current_lr(config) * 312.5},
        {"params": gaussians.color, "lr": get_current_lr(config) * 31.25},
    ])
```

---

## ✅ Verification：怎么判断训练成功或失败？

### 收敛标准（定量）

| 指标 | 计算方法 | 收敛条件 | 说明 |
|------|---------|---------|------|
| **PSNR** | 10·log₁₀(1/L1) | >30 dB（合成数据）/ >25 dB（实景） | 连续 1000 步变化 < 0.1 dB → 收敛 |
| **高斯数量 N** | len(gaussians) | 稳定在 [10⁵, 10⁶] | 连续 1000 步波动 < 1% → 收敛 |
| **总损失 L_total** | L_img + λ·L_reg | 持续下降 | 连续 1000 步变化 < 0.001 → 收敛 |

**满足以上任意两条** → 认为训练成功。

---

### 失败诊断表

```
+------------------+------------------+------------------------+---------------------------+
| 症状             | 可能原因         | 检查点                 | 解决方案                  |
+------------------+------------------+------------------------+---------------------------+
| PSNR 不升        | LR 太小          | grads.mu.norm() ≈ 0    | LR ×10                    |
| (卡在低值)       | 初始化差         | 初始渲染图 vs 真实图   | scale_factor ×5-10        |
|                  | 梯度消失         | Sigma.det() ≈ 0        | Sigma += I·1e-8           |
+------------------+------------------+------------------------+---------------------------+
| 高斯数量爆炸     | densify 太激进   | N 增长曲线             | grad_threshold ↑          |
| (N > 10M)        |                  |                        | max_gaussians ↓           |
+------------------+------------------+------------------------+---------------------------+
| 渲染全黑         | Σ太小            | scales.mean() < 1e-6   | scale_factor ×10          |
|                  | α≈0              | alpha.mean()           | alpha.clamp(min=0.01)     |
+------------------+------------------+------------------------+---------------------------+
| 渲染全白         | Σ太大            | scales.mean() > 100    | scale_factor ÷2           |
|                  | α≈1              | alpha.mean() ≈ 1       | alpha.clamp(max=0.9)      |
+------------------+------------------+------------------------+---------------------------+
| 条纹伪影         | Σ奇异（det≈0）   | Sigma.eigvals()        | Sigma += I·1e-8           |
+------------------+------------------+------------------------+---------------------------+
| 空洞不愈合       | densify 不触发   | grads_mu_cache.max()   | grad_threshold ↓          |
|                  |                  | radii_cache.max()      | scale_threshold ↓         |
+------------------+------------------+------------------------+---------------------------+
```

---

## 📊 Example：训练曲线可视化

### 必须记录的指标（TensorBoard/WandB）

```python
# TensorBoard 日志示例
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/3dgs_training")

for step in range(total_steps):
    # ... 训练代码 ...
    
    if step % config["log_interval"] == 0:
        writer.add_scalar("metrics/psnr", psnr, step)
        writer.add_scalar("metrics/l1_loss", L1.item(), step)
        writer.add_scalar("metrics/ssim_loss", L_ssim.item(), step)
        writer.add_scalar("gaussians/count", len(gaussians), step)
        writer.add_scalar("gaussians/alpha_mean", gaussians.alpha.mean().item(), step)
        writer.add_scalar("gaussians/scale_mean", scales.mean().item(), step)
        
        # 梯度统计（诊断用）
        if gaussians.mu.grad is not None:
            writer.add_histogram("grads/mu", gaussians.mu.grad, step)
```

---

### 典型训练曲线（合成数据）

```
PSNR vs Step:
35 ┤                                   ╭────╮
   │                               ╭───╯    ╰────
30 ┤                         ╭─────╯
   │                     ╭───╯
25 ┤               ╭─────╯
   │           ╭───╯
20 ┤     ╭─────╯
   │ ╭───╯
15 ┼─╯
   └────────────────────────────────────────────
    0k  5k  10k 15k 20k 25k 30k (steps)

N (#gaussians) vs Step:
2M ┤                                   ╭───────╮
   │                               ╭───╯       ╰────
1.5M┤                         ╭─────╯
   │                     ╭───╯
1M ┤               ╭─────╯
   │           ╭───╯
0.5M┤     ╭─────╯
   │ ╭───╯
0  ┼─╯
   └────────────────────────────────────────────
    0k  5k  10k 15k 20k 25k 30k (steps)

Loss vs Step:
0.5 ┤╭────
   ││
0.4 ┤│
   ││
0.3 ┤│    ╰───────────
   ││
0.2 ┤│
   │╰
0.1 ┼──
   └────────────────────────────────────────────
    0k  5k  10k 15k 20k 25k 30k (steps)

LR vs Step:
5e-2┤─────────────╮
   │             ╰─────────────╮
5e-3┤                          ╰──────
   │                                
1.6e-4┼───────────────────────────────
   └────────────────────────────────────────────
    0k     7.5k        15k       30k (steps)
```

**观察要点**：
1. **PSNR**：初期快速上升，后期缓慢收敛 → 检查是否满足收敛条件
2. **N（高斯数量）**：前期增长快，后期稳定或下降 → densify/prune 在正常工作
3. **Loss**：应该持续下降，无剧烈震荡 → 如果震荡说明 LR 太大
4. **LR**：阶梯式衰减，每次衰减后观察 PSNR 是否继续上升

---

## 🧠 思考题（第一性原理式）

### Q1: 为什么 densify 用缓存的梯度，不用当前步的？

**提示链**：
- 当前步梯度 = ∂L/∂θ | θ=当前参数，batch=当前视角
- 如果 batch size = 1（单视角），梯度的方差有多大？
- 缓存 N 步平均后，方差降低多少倍？
- **推导**：假设每步采样独立，方差 σ² → N 步平均后方差 σ²/N

---

### Q2: LR 调度的"经济学"

**提示链**：
- Adam 的自适应 LR 已经考虑了梯度统计（v_t），为什么还需要全局调度？
- 从**收敛性证明**的角度：如果 θ*是最优解，‖θ_t - θ*‖ = η_t · ‖∇L(θ*)‖ + O(η²)
- **推导**：当 t → ∞时，需要 lim η_t = 0（但不要太快）→ 阶梯衰减 vs 线性衰减的收敛速率比较

---

### Q3: Adam vs SGD

**提示链**：
- μ、Σ、α、c 四个参数的梯度量级分别是多少？（从训练日志看）
- 如果用 SGD，需要手动设置多少个不同的 LR？
- **推导**：假设‖∇μ‖ ≈ 1e-4, ‖∇α‖ ≈ 1e-2，用同一个 LR=1e-3 会怎样？

---

### Q4: 早停策略（Early Stopping）

**提示链**：
- 当前是固定步数（30k），但不同场景收敛速度不同
- **设计任务**：如何自适应判断"训练完成"？
  - 如果 PSNR 连续 N 步变化 < ε，可以停吗？会不会陷入局部最优？
  - 如果用验证集 PSNR（不在训练集中的视角）作为停止条件，如何实现？

---

### Q5: "死亡"高斯的复活机制

**提示链**：
- 如果一个高斯长期梯度为 0（‖∇θ‖ ≈ 0），它永远不会被更新 → "死亡"
- **设计任务**：如何检测并处理死亡高斯？
  - 方案 A：定期重置随机子集的参数？
  - 方案 B：基于密度估计，在梯度为 0 的区域添加新高斯？
  - 从信息论角度：梯度为 0 意味着该参数对损失不敏感（∂L/∂θ = 0），如何重新激活？

---

## 📝 本章核心记忆点

✅ **训练循环状态机**：采样 → 渲染 → 损失 → 反向 → 更新 → densify/prune → LR 调度 → 日志

✅ **密度控制时机**：每 1000 步，缓存梯度 + 投影半径，基于双条件（尺度 + 梯度）决策

✅ **学习率调度**：三阶段衰减（7.5k、15k 步 ×0.1），分参数组（μ、Σ、α、c 不同 LR）

✅ **收敛判断**：PSNR 连续 1000 步变化 < 0.1 + N 稳定在合理范围

✅ **关键洞察**：densify/prune 需要缓存梯度（避免噪声），优化器需要重建（参数数量变）

---

## 🚀 下一章预告

训练完成后，你有了高质量的高斯场景。但怎么实现**实时推理**（<10ms/帧）？

第 8 章我们将深入：
- **CUDA kernel 融合**：投影 + 排序 + 混合渲染如何优化成一个 kernel？
- **内存布局优化**：SoA vs AoS，缓存友好性设计
- **Batch 推理**：多视角同时渲染的并行策略
- **生产部署**：从 PyTorch 到 TensorRT/CUDA C++

准备好燃烧更多 GPU 显存了吗？🔥
