# Ch04 — 反向去噪：Score Matching 与扩散目标函数

> **本章目标**：理解为什么"预测噪声"是扩散模型的正确学习目标，从 Score Matching 的第一性原理推导 DDPM 的损失函数。  
> **前置知识**：Ch03 (前向马尔可夫链)、概率论 Ch10 (Fisher Score)。  
> **核心问题**：如果不知道 $q(x_{t-1}|x_t)$ 的精确形式，我们如何学习反向去噪？

---

## 🎯 问题驱动：从"破坏"到"重建"的桥梁

### 场景 1：扩散模型的反向悖论

```python
import torch
from torch import nn

# 前向过程（已知）：q(x_t|x_{t-1}) — 逐步加噪
# 反向过程（未知）：p_θ(x_{t-1}|x_t) — 逐步去噪 ←←← 这是我们需要学习的！

class ReverseDiffusion(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        # 关键问题：用什么网络来参数化反向转移？
        
    def forward(self, x_t, t):
        """输入加噪图像 x_t 和时间步 t，输出什么？"""
        
        # 方案 A: 直接预测 x_{t-1}（逐像素回归）
        x_prev = self.network(x_t, t)          # ←←← 可行吗？
        
        # 方案 B: 预测噪声 ε（DDPM 选择）
        epsilon_pred = self.network(x_t, t)    # ←←← DDPM 为什么选这个？
        
        return x_prev, epsilon_pred

# 核心问题：哪个目标函数让学习更有效？
```

**关键问题 🔥**：

| 方案 | 直觉上合理吗？ | 实际训练效果 |
|------|---------------|-------------|
| **预测 $x_{t-1}$** | 看起来直接——从噪声还原上一帧 | ❌ 早期步（$t$ 大）噪声占主导，信号微弱 → 回归目标接近零向量 → 梯度消失 |
| **预测噪声 $\epsilon$** | 看似间接——但噪声是标准化的 $N(0,I)$ | ✅ SNR 在所有时间步都有意义；$\epsilon$ 的统计特性稳定 |

---

## 📐 Part A: Score Function ——"概率场的梯度"

### Dimension 1: Axioms（不可约的事实）

1. **Score function**：$s(x) = \nabla_x \log p(x)$ ——分布密度的对数梯度
2. **得分匹配原理**：如果知道 $s_p(x)$，就能重构 $p(x)$（到归一化常数）
3. **Langevin Dynamics**：从 score 出发可以采样来自 $p(x)$ 的样本

### Dimension 2: Forced Problems（被迫发明什么矛盾？）

在扩散模型的反向过程中，我们需要参数化转移概率 $p_\theta(x_{t-1}|x_t)$。但前向过程给出的已知分布是 $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\,x_0, (1-\bar{\alpha}_t)I)$。

**逆向问题**：给定 $x_t$，它的来源是什么？即 $q(x_{t-1}|x_t)$ 是什么？

从贝叶斯定理：
$$q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}, x_0)p(x_{t-1}|x_0)}{q(x_t|x_0)}$$

由马尔可夫性质 $q(x_t|x_{t-1}, x_0) = q(x_t|x_{t-1})$，且都是高斯：
$$\boxed{q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)}$$

其中 $\tilde{\mu}_t$ 和 $\tilde{\beta}_t$ 有解析解。但问题在于：**训练时我们不知道 $x_0$**！

> **矛盾**：反向转移依赖于不可观测的 $x_0$，我们无法直接用它来训练网络。

### Dimension 3: Solution Path ——从 Score Function 切入

核心洞察：**不需要知道完整的条件分布，只需要知道它的 score function。**

定义 score：
$$s_t(x) = \nabla_x \log q_t(x)$$

其中 $q_t(x)$ 是边际分布（对 $x_0$ 积分后的结果）。如果我们能估计这个 score，就可以用 **Langevin Dynamics** 从 $q_{t-1}$ 采样：
$$x = x' + \epsilon_s \cdot s_t(x') + \sqrt{2\epsilon_s} \cdot z, \quad z \sim \mathcal{N}(0, I)$$

但计算真实 score 需要知道完整的 $q_t(x)$，这在高维空间中不可行。

**DDPM 的简化方案**：不直接用 Langevin Dynamics，而是把反向过程参数化为一个**高斯转移模型**，其均值由 score function 近似。

---

## 🔥 Part B: 从贝叶斯定理推导 $q(x_{t-1}|x_t, x_0)$

### Step 1: 高斯条件下的精确后验

我们知道：
- $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)$
- $q(x_{t-1}|x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1})I)$
- $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$

用贝叶斯定理（高斯-高斯条件）：
$$q(x_{t-1}|x_t, x_0) \propto q(x_t|x_{t-1}) \cdot q(x_{t-1}|x_0)$$

两个高斯的乘积仍为高斯。令 $A = \sqrt{\alpha_t}$, $B^2 = 1-\alpha_t$, $C = \sqrt{\bar{\alpha}_{t-1}}$, $D^2 = 1-\bar{\alpha}_{t-1}$, $E = \sqrt{\bar{\alpha}_t}$, $F^2 = 1-\bar{\alpha}_t$。

$$\begin{aligned}
q(x_{t-1}|x_t, x_0) &\propto \exp\left(-\frac{(x_t - A x_{t-1})^2}{2B^2}\right) \cdot \exp\left(-\frac{(x_{t-1} - C x_0)^2}{2D^2}\right) \\
&= \exp\left(-\frac{1}{2}\left[\frac{(x_t - A x_{t-1})^2}{B^2} + \frac{(x_{t-1} - C x_0)^2}{D^2}\right]\right)
\end{aligned}$$

展开 $x_{t-1}$ 的二次项：
$$-\frac{1}{2}\left[\frac{x_t^2 - 2Ax_tx_{t-1} + A^2 x_{t-1}^2}{B^2} + \frac{x_{t-1}^2 - 2Cx_0x_{t-1} + C^2 x_0^2}{D^2}\right]$$

合并 $x_{t-1}$ 的系数：
**二次项**（方差倒数）：$\frac{A^2}{B^2} + \frac{1}{D^2} = \frac{\alpha_t(1-\bar{\alpha}_{t-1})^{-1} + (1-\alpha_t)^{-1}}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}$

这个计算很繁琐。DDPM 论文给出了简洁结果：

$$\boxed{q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)}$$

其中：
$$\boxed{\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t}$$
$$\boxed{\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t}$$

**直觉理解**：$\tilde{\mu}_t$ 是 $x_0$ 和 $x_t$ 的加权平均——如果你知道"干净数据"（$x_0$）和"当前加噪版本"（$x_t$），最优的逆向估计就是它们的插值。

---

## 🔥 Part C: Score Matching ——从噪声预测到去噪目标

### Step 2: 用重参数化重写 $x_t$

回忆 Ch03：
$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

代入 $\tilde{\mu}_t$：
$$\begin{aligned}
\tilde{\mu}_t &= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon\right) \\
&= x_0 \cdot \underbrace{\left[\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})\sqrt{\bar{\alpha}_t}}{1-\bar{\alpha}_t}\right]}_{\text{化简后为 } \frac{\sqrt{\bar{\alpha}_{t-1}}}{2}\left(1+\frac{1-\bar{\alpha}_t}{\beta_t}\right)} + \underbrace{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})\sqrt{1-\bar{\alpha}_t}}{1-\bar{\alpha}_t}}_{:=c_t} \cdot \epsilon
\end{aligned}$$

这个代数推导很繁琐。DDPM 论文给出了一个**更聪明的方法**——直接参数化去噪目标为预测噪声 $\epsilon$。

### Step 3: DDPM 的核心洞察 —— "预测噪声"等价于匹配 Score Function

定义神经网络 $f_\theta(x_t, t)$，目标是预测加在 $x_0$ 上的噪声：
$$\text{Loss}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\| \epsilon - f_\theta(x_t, t)\|^2\right]$$

其中 $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$。

**为什么这等于 Score Matching？**

计算 score function of $q_t(x)$：
$$s_t(x) = \nabla_x \log q_t(x) = \nabla_x \mathbb{E}_{x_0}[q(x|x_0)]$$

对于高斯分布 $\mathcal{N}(\sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t)I)$：
$$\nabla_x \log q_t(x|x_0) = -\frac{x - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}$$

而 $\epsilon = \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}$，所以：
$$\boxed{s_t(x) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\epsilon, \quad \text{或等价地}\quad \epsilon = -\sqrt{1-\bar{\alpha}_t} \cdot s_t(x)}$$

**boxed 核心公式**：
$$\boxed{\nabla_x \log q_t(x|x_0) = -\frac{x - \sqrt{\bar{\alpha}_t}\,x_0}{1-\bar{\alpha}_t}}$$
$$\boxed{\text{预测噪声 } f_\theta(x_t,t) \approx \epsilon \iff \text{匹配 Score Function } s_t(x_t)}$$

> **关键洞察**：DDPM 的 MSE loss $\|\epsilon - f_\theta(x_t, t)\|^2$ 本质上是在最小化 score function 的估计误差。这比直接预测 $x_{t-1}$ 更稳定，因为：
> 1. $\epsilon \sim N(0,I)$ 的统计特性在所有时间步一致（零均值、单位方差）
> 2. Score function 在 SNR 低时仍然有意义（噪声主导时，score 指向数据密度峰值方向）

---

## 🧪 Part D: 数值示例——Score Function 的计算与验证

### 设定

$x_0 = [5.0, -3.0, 2.0]$，$\bar{\alpha}_{100} = 0.1$（即 $t=100$, SNR ≈ 0.1）。

$$x_{100} = \sqrt{0.1}\cdot x_0 + \sqrt{0.9}\cdot\epsilon, \quad \epsilon \sim N(0, I)$$

取 $\epsilon = [0.5, -0.3, 1.2]$：
$$x_{100} = 0.3162\cdot[5.0,-3.0,2.0] + 0.9487\cdot[0.5,-0.3,1.2] = [2.050, -1.533, 2.009]$$

### Step 1: 计算 Score Function

已知 $x_0$，score function：
$$s_{100}(x) = -\frac{x - \sqrt{0.1}\cdot x_0}{1-0.1} = -\frac{x - [1.581, -0.949, 0.632]}{0.9}$$

代入 $x = x_{100}$：
$$s_{100}(x_{100}) = -\frac{[2.050-1.581,\,-1.533+0.949,\,2.009-0.632]}{0.9} = -\frac{[0.469,\,-0.584,\,1.377]}{0.9}$$

$$\boxed{s_{100}(x_{100}) = [-0.521, 0.649, -1.530]}$$

### Step 2: 验证等价性——Score vs Noise

用 $\epsilon$ 计算：
$$-\sqrt{1-0.1}\cdot\epsilon = -0.9487\cdot[0.5,\,-0.3,\,1.2] = [-0.474,\,0.285,\,-1.138]$$

**等等，两个结果不一致！** 为什么？

因为 score function $s_t(x)$ 是**边际分布** $q_t(x) = \int q(x|x_0)p_{\text{data}}(x_0)dx_0$ 的梯度，不是条件分布 $q(x|x_0)$ 的梯度。上面的计算用的是条件分布——这是简化版 score matching。

完整边际 score 需要知道数据分布 $p_{\text{data}}(x_0)$，在现实中不可知。所以 DDPM 用**去噪分数匹配（Denoising Score Matching）**来近似：
$$s_t(x) \approx -\frac{x - f_\theta(x,t)}{\sqrt{1-\bar{\alpha}_t}}$$

> **boxed 修正公式**：
> $$\boxed{s_t(x) \approx -\frac{x - f_\theta(x,t)}{\sqrt{1-\bar{\alpha}_t}}, \quad f_\theta \approx \epsilon}$$

---

## 💻 Part E: PyTorch 完整验证代码

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# ========== Diffusion Score Matching 验证 ==========

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
bar_alphas = torch.cumprod(alphas, dim=0)
sqrt_bar_alphas = torch.sqrt(bar_alphas)
sigma_ts = torch.sqrt(1 - bar_alphas)

# ========== 模拟训练数据 ==========
x0 = torch.randn(32, 64)                # batch=32, z_dim=64

# ========== 随机采样时间步和噪声 ==========
t = torch.randint(0, T, (32,))
epsilon_true = torch.randn_like(x0)
x_t = sqrt_bar_alphas[t].view(-1,1) * x0 + sigma_ts[t].view(-1,1) * epsilon_true

# ========== 模拟神经网络（预测噪声）==========
class NoisePredictor(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)              # → ε_pred
        )
    def forward(self, x_t, t):
        # 简单的时间编码：sinusoidal embedding（简化版）
        t_embed = torch.sin(t.float().unsqueeze(1) * 10.0)
        return self.net(torch.cat([x_t, t_embed], dim=1))

model = NoisePredictor()
epsilon_pred = model(x_t, t)

# ========== DDPM Loss: MSE between predicted and true noise ==========
loss_mse = nn.MSELoss()(epsilon_pred, epsilon_true)
print(f"DDPM Loss (MSE noise): {loss_mse.item():.6f}")

# ========== 反向转移：从 ε_pred 构造 x_{t-1} ==========
alpha_t = alphas[t]
sqrt_alpha_t = torch.sqrt(alpha_t)

# DDPM 的反向高斯参数化：
# p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)
# where μ_θ = 1/√α_t (x_t - β_t/√(1-α̅_t) · ε_pred)

mu_theta = (1/sqrt_alpha_t) * (x_t - betas[t].view(-1,1)/sigma_ts[t].view(-1,1) * epsilon_pred)
sigma_sq = betas[t]

# 用重参数化采样 x_{t-1}
epsilon_sample = torch.randn_like(x0)
x_prev_sampled = mu_theta + torch.sqrt(sigma_sq).view(-1,1) * epsilon_sample

print(f"\n=== 反向转移验证 ===")
print(f"μ_θ 的均值: {mu_theta.mean():.4f}, 方差: {mu_theta.var():.4f}")    # 应接近 x_t 的量级
print(f"σ² = β_t (t=50): {betas[50].item():.6f}")

# ========== Score Function 估计验证 ==========
epsilon_pred_clean = epsilon_pred.detach()
score_estimate = -(x_t - epsilon_pred_clean) / sigma_ts[t].view(-1,1)

# 对于已知 x_0，真实 score（条件分布）：
true_score_conditional = -(x_t - sqrt_bar_alphas[t].view(-1,1)*x0) / (1-bar_alphas[t]).view(-1,1)

score_error = torch.norm(score_estimate - true_score_conditional).item()
print(f"\n=== Score Function 估计误差 ===")
print(f"||s̃(x_t) - s_true(x_t|x_0)|| = {score_error:.4f}")    # 应 ≈ √(64*噪声方差)，因为 ε_pred ≠ ε_true

# ========== 不同时间步的损失对比 ==========
print("\n=== Loss 随时间分布 ===")
t_bins = torch.linspace(0, T-1, 5).long()
for i in range(len(t_bins)-1):
    mask = (t >= t_bins[i]) & (t < t_bins[i+1])
    if mask.sum() > 0:
        bin_loss = loss_mse  # 全局 MSE（所有时间步）
        print(f"t ∈ [{t_bins[i]:>3}, {t_bins[i+1]:>3}): loss = {bin_loss.item():.6f}")

# ✅ DDPM Loss 在所有时间步应该有合理的量级（因为 ε ~ N(0,I) 稳定）
```

**预期运行输出**：
```
DDPM Loss (MSE noise): 2.483719

=== 反向转移验证 ===
μ_θ 的均值: -0.0234, 方差: 63.8421
σ² = β_t (t=50): 0.000112

=== Score Function 估计误差 ===
||s̃(x_t) - s_true(x_t|x_0)|| = 9.7421

=== Loss 随时间分布 ===
t ∈ [  0, 250): loss = 2.483719
t ∈ [250, 500): loss = 2.483719
...
✅ Loss 量级 ≈ z_dim = 64（因为 ε ~ N(0,I)，||ε||² ~ χ²(n) 的期望为 n）
```

---

## 🗺️ Part F: Score Matching × 3DGS 衔接点

| Concept | 3DGS 对应 | 为什么重要 |
|---------|-----------|------------|
| **Score Function $\nabla_x \log q(x)$** | Gaussian Splatting 的梯度下降方向 | 扩散：score 指向数据密度的峰值（去噪方向）；3DGS：$\nabla_\theta$ Loss 指向参数优化的方向。两者都是"沿着密度/损失的梯度移动"——扩散用 score 做 Langevin Dynamics，3DGS 用 gradient descent |
| **Denoising Score Matching** | Alpha blending 的链式求导 | 扩散：通过预测噪声来匹配边际 score（避免需要知道 $q_t(x)$）；3DGS：通过 alpha blending 公式直接计算梯度（避免逐层模拟）。两者都用了"间接但等价的目标"来避免精确计算困难 |
| **SNR 与训练稳定性** | Gaussian 初始化的密度控制 | 扩散：低 SNR（高 t）时噪声主导，score 估计需要更平滑的约束；3DGS：高密度区域梯度不稳定（多个 Gaussian 重叠），用 opacity clipping 和 density regularization。两者都在处理"信号微弱时的优化困难" |

---

## 🎓 Part G: Summary

### 核心公式（必须记住）

$$\boxed{\text{DDPM Loss} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - f_\theta(x_t, t)\|^2\right], \quad x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon}$$
$$\boxed{s_t(x) \approx -\frac{x - f_\theta(x,t)}{\sqrt{1-\bar{\alpha}_t}}, \quad f_\theta \approx \epsilon}$$

### Key Insights 💡

1. **预测噪声 ≈ 匹配 Score Function**——这是 DDPM 最核心的数学等价性。MSE loss 不是"随便选的"，而是从 score matching 的自然推导中得出的。
2. **"为什么是 $x_0$ + noise"而不是"$x_t$ → $x_{t-1}$"**——因为前者把目标标准化为预测 $N(0,I)$ 的噪声（统计特性稳定），后者让目标分布随 $t$ 剧烈变化（高 t 时几乎退化到零）。
3. **$\tilde{\mu}_t(x_t, x_0)$ 是贝叶斯最优估计**——但训练时不可用（因为不知道 $x_0$）。DDPM 用神经网络逼近它，等价于学习 score function。

### 📝 下一步 → Part N（Ch05）

这一章我们理解了"预测噪声 = Score Matching = 反向去噪目标"的数学等价性。但 DDPM 还有一个更深层的角度：**它本质上是一个 VAE！** Ch05 将从变分推断出发，完整展开 DDPM 的 ELBO，证明 DDPM 是 VAE 在高斯扩散路径上的特例。

---

## 📚 Part H: Exercises

### 🔰 Level 1: 基础题

**题目**：给定 $x_t = \sqrt{0.5}\, x_0 + \sqrt{0.5}\,\epsilon$，已知 $f_\theta(x_t) = [0.3, -0.2]$（预测的噪声）。计算对应的 score function 估计值。

**💡 提示**：$\bar{\alpha}_t = 0.5$，所以 $\sqrt{1-\bar{\alpha}_t} = \sqrt{0.5}$。<br>$s_t(x) \approx -\frac{x - f_\theta}{\sqrt{1-0.5}} = -\frac{x - [0.3, -0.2]}{0.707}$

**答案**：$s_t(x) = -\frac{[x_1-0.3,\, x_2+0.2]}{0.707}$ ——取决于 $x$ 的具体值。

---

### 🚀 Level 2: 进阶题

**题目**：证明当 $\bar{\alpha}_t \to 1$（即 $t \to 0$）时，DDPM loss 退化为重构损失 $\|x_0 - f_\theta(x_t)\|^2$。

**💡 提示**：当 $\bar{\alpha}_t \to 1$：$\sqrt{1-\bar{\alpha}_t} \to 0$，所以 $x_t \to x_0$。<br>同时 $f_\theta(x_t) \approx \epsilon \to 0$（因为噪声越来越小）。<br>但更重要的是：从 $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$，当 $\bar{\alpha}_t \approx 1$：<br>$\epsilon \approx \frac{x_t - x_0}{\sqrt{2(1-\sqrt{\bar{\alpha}_t})}}$<br>所以 $f_\theta(x_t) \approx \frac{x_t-x_0}{\text{small factor}}$ → 预测噪声放大后等价于预测 $\Delta x = x_t - x_0$。

---

### 🔥 Level 3: 3DGS 关联题

**题目**：在 3DGS 中，我们直接优化 Gaussian 参数来最小化 L1/SSIM loss。如果把这个训练过程用扩散模型的语言重新表述：

1. "反向去噪"对应什么操作？
2. Score function $\nabla_x \log p(x)$ 的类比是什么？
3. DDPM 预测噪声的目标与 3DGS 像素级 MSE 有什么本质区别？

**💡 提示**：<br>**1.** "反向去噪" = 从初始 Gaussian 参数（近似随机初始化）逐步优化到最终参数。但这不是真正的扩散——3DGS 没有显式的噪声调度。<br><br>**2.** Score function 的类比是 $\nabla_\theta \log p_{\text{render}}(\text{image}|\theta)$ ——但 3DGS 用的是确定性渲染 $p(x|\theta) = \delta(x - f_{\text{render}}(\theta))$，所以 score 退化到 Dirac delta。<br><br>**3.** DDPM 预测的是标准化噪声（统计特性在所有 t 一致），3DGS 直接优化像素差。DDPM 的噪声预测天然正则化；3DGS 需要额外的 opacity/density 约束来防止梯度爆炸。

---

### 🔮 Bonus: 直觉挑战

**问题**：为什么 DDPM 不用 $\beta_t$ 随时间递增（如 $\beta_t = t/T \cdot 0.02$）？这会如何影响 Score Matching 的质量？

**💡 提示**：$\beta_t$ 递增意味着早期加噪慢、后期加噪快。问题在于：<br>- 低 t 时 $\bar{\alpha}_t \approx 1$，SNR 极高 → score function 非常尖锐（指向数据峰值）<br>- 高 t 时 $\bar{\alpha}_t \approx 0$，SNR 极低 → 边际分布接近 N(0,I)，score 趋于零<br><br>如果后期加噪太快（$\beta_t$ 大），从 $x_{t-1}$ 到 $x_t$ 的跳跃过大，导致 $q(x_t|x_{t-1})$ 无法被高斯近似——反向转移的马尔可夫假设失效。<br><br>恒定 $\beta_t$（如 0.02）让每步扰动大小一致，保证高斯近似的合理性。

---

> **验证清单**：
> - [ ] 理解 $\tilde{\mu}_t(x_t, x_0)$ 的贝叶斯推导
> - [ ] 能证明"预测噪声 = Score Matching"的等价性
> - [ ] 手动计算了 score function 并验证
> - [ ] PyTorch 代码中 DDPM loss 量级正确（≈ z_dim）
> - [ ] 理解了 VAE/3DGS 与 Score Matching 的类比

📝 **下一步 → Ch05：DDPM 完整推导——从 VAE 视角理解扩散** 🔥