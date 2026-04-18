# VAE & Diffusion 教程 — 一页纸 Cheat Sheet

> **系列概述**：从变分自编码器（VAE）到扩散模型（Diffusion），再到与 3D Gaussian Splatting 的深度衔接。共 7 章，覆盖生成模型的完整数学基础。  
> **前置知识**：概率论与信息论、优化理论。  
> **目标读者**：理解 3DGS 训练原理，想深入生成模型数学的开发者/研究者。

---

## 📚 章节导航

| Ch | 标题 | 核心公式 | 对应文件 |
|----|------|----------|----------|
| Ch01 | 潜空间与变分推断 | ELBO = E[log p(x\|z)] - KL(q‖p) | [链接](./01_一_潜空间与变分推断.md) |
| Ch02 | ELBO优化与重参数化 | z = μ + σ·ε | [链接](./02_二_ELBO优化与重参数化技巧.md) |
| Ch03 | 前向加噪与马尔可夫链 | q(x_t\|x₀) = N(√α̅ₜx₀, (1-α̅ₜ)I) | [链接](./03_三_前向加噪与马尔可夫链.md) |
| Ch04 | 反向去噪与Score Matching | Loss = E[‖ε - f_θ(x_t,t)‖²] | [链接](./04_四_反向去噪与Score Matching.md) |
| Ch05 | DDPM完整推导（VAE视角） | ELBO = Σ KL(q‖p_θ) ≈ Σ MSE | [链接](./05_五_DDPM完整推导与VAE视角.md) |
| Ch06 | Score-based SDE 视角 | dx = f(t)x dt + g(t)dW | [链接](./06_六_Score_based_Models与SDE视角.md) |
| Ch07 | VAE/Diffusion × 3DGS 衔接 | Gaussian二次型 + 乘积压缩 | [链接](./07_七_VAE与Diffusion和3DGS的深度衔接.md) |

---

## 🔑 核心公式总览（按概念分类）

### A. VAE — 变分推断

```
ELBO = E[q(z|x)][log p(x|z)] - KL(q(z|x) ‖ p(z))        ← Ch01

log p(x) = ELBO + KL(q(z|x) ‖ p(z|x)) ≥ ELBO             ← 下界性质

Reparam: z = μ_φ(x) + σ_φ(x) · ε, ε ~ N(0,I)              ← Ch02
∇_φ f(z) = (∂f/∂z)|_{z=μ+σε} · [1; ε]                     ← 梯度传播

KL(N(μ₁,σ₁²) ‖ N(μ₂,σ₂²)) = ½[log(σ₂²/σ₁²) - 1 + (σ₁²+(μ₁-μ₂)²)/σ₂²]
```

### B. Diffusion — 前向过程（Ch03）

```
α_t = 1 - β_t                                              ← 方差调度定义
α̅_t = Π_{s=1}^t α_s                                       ← 累积乘积（信号保留率）

q(x_t|x_{t-1}) = N(√αₜ·x_{t-1}, (1-αₜ)I)                  ← 一步转移
q(x_t|x₀) = N(√α̅ₜ·x₀, (1-α̅ₜ)I)                           ← 闭式解（关键！）

SNR(t) = α̅_t / (1 - α̅_t)                                  ← 信噪比
```

### C. Diffusion — 反向过程（Ch04）

```
DDPM Loss = E_{t,x₀,ε}[‖ε - f_θ(x_t,t)‖²]                  ← 噪声预测MSE
x_t = √α̅ₜ·x₀ + √(1-α̅ₜ)·ε                                  ← 重参数化前向

s_t(x) ≈ -(x - f_θ(x,t)) / √(1-α̅ₜ)                       ← Score Function估计
```

### D. Diffusion — ELBO分解（Ch05）

```
L_T = KL(q(x_T|x₀) ‖ p(x_T)) → 0 (当 α̅_T ≈ 0)             ← 尾部KL消失

L_{t-1} = E[‖ε - f_θ(x_t,t)‖²]                             ← KL退化为MSE
ELBO ≈ Σ_{t=1}^{T-1} L_{t-1}                                ← DDPM损失近似
```

### E. Score-based SDE（Ch06）

```
前向SDE:  dx = f(t)x dt + g(t)dW                           ← Itô形式
反向SDE:  dx = [f(t)x - g²s_{q_t}(x)]dt + g(t)dẆ            ← 去噪方向

Probability Flow ODE:  dx/dt = f(t)x - ½g²s_{q_t}(x)       ← 确定性路径
```

### F. VAE/Diffusion × 3DGS 衔接（Ch07）

| 概念 | Diffusion | 3DGS | 统一表达 |
|------|-----------|------|----------|
| **精度矩阵** | $\frac{1}{1-\bar{\alpha}_t}I$ | $\Sigma^{-1}$ | $A(x-\mu)^T A (x-\mu)$ |
| **乘积压缩** | $\bar{\alpha}_t=\prod(1-\beta_s)$ | $T_i=\prod_{j<i}(1-\alpha_j)$ | 历史信息O(1)编码 |
| **重构损失** | MSE: ‖ε - f_θ‖² | L1/SSIM | E[‖reconstruction‖] |
| **正则化** | KL(q‖p), L_T → 0 | density R(θ) | 防止退化 |

---

## 🧠 关键洞察总结（每章一句）

| Ch | 一句话核心 |
|----|-----------|
| Ch01 | ELBO从KL≥0+贝叶斯定理自然推导，不需要新假设 |
| Ch02 | 重参数化分离确定性与随机性，使autograd可沿确定性路径传播梯度 |
| Ch03 | 马尔可夫+高斯=闭式解，无需模拟T步即可跳到任意时刻t |
| Ch04 | 预测噪声≈Score Matching——MSE loss本质是匹配概率场的梯度 |
| Ch05 | DDPM = VAE的推广：有T层潜变量，每层ELBO项退化为MSE |
| Ch06 | DDPM是SDE的离散化；ODE solver可少步数高质量采样 |
| Ch07 | 高斯二次型+乘积压缩是多目标优化的通用结构 |

---

## 🗺️ 学习路径图

### 🔰 最低限度（3章）
```
Ch01 → Ch04 → Ch05
VAE直觉 → Score Matching → DDPM=VAE特例
```

### ⚡ 标准路线（6章）
```
Ch01→Ch02→Ch03→Ch04→Ch05→Ch06
变分推断 → 重参数化 → 前向扩散 → Score Matching → ELBO分解 → SDE统一
```

### 🔥 深度路径（含3DGS）
```
标准路线 + Ch07
全部生成模型数学 + 与可微渲染的深度衔接分析
```

---

## 📐 公式速查表

| 符号 | 含义 | 所属章节 |
|------|------|----------|
| $\beta_t$ | 方差调度参数，控制每步加噪量 | Ch03 |
| $\alpha_t = 1-\beta_t$ | 一步保留率 | Ch03 |
| $\bar{\alpha}_t = \prod_{s=1}^t\alpha_s$ | T步累积信号率 | Ch03 |
| $f_\theta(x_t,t)$ | 神经网络预测的噪声 | Ch04, Ch05 |
| $q(z|x)$ / $p_\theta(x\|z)$ | 变分后验 / 解码器生成分布 | Ch01 |
| ELBO | $\mathbb{E}[\log p]-\text{KL}(q\|p)$，对数似然下界 | Ch01, Ch05 |
| $s_t(x) = \nabla_x\log q_t(x)$ | Score function，概率场梯度 | Ch04, Ch06 |
| $L_{t-1}$ | 反向转移KL → MSE退化项 | Ch05 |
| $f(t)x dt + g(t)dW$ | SDE前向漂移+扩散项 | Ch06 |

---

## 🔬 PyTorch 验证代码片段汇总

### VAE 重参数化（Ch02）
```python
mu, logvar = encoder(x)
sigma = torch.exp(0.5 * logvar)
epsilon = torch.randn_like(mu)
z = mu + sigma * epsilon        # ←←← autograd可追踪！
```

### Diffusion 前向闭式解（Ch03）
```python
x_t = sqrt_bar_alpha[t] * x_0 + sigma_ts[t] * noise    # ←←← 一步跳到t
# 不需要循环！
```

### DDPM Loss（Ch04, Ch05）
```python
epsilon_pred = net(x_t, t)
loss = ((epsilon - epsilon_pred)**2).mean()              # ←←← DDPM标准损失
```

### Score Function 估计（Ch06）
```python
score = -(x_t - epsilon_pred) / sigma_ts[t]             # ←←← Score ≈ -(x-f)/σ
```

---

## 🎯 3DGS 衔接要点速查

| 扩散概念 | 3DGS对应 | 数学形式 |
|----------|----------|----------|
| Gaussian密度二次型 | Splat高斯形状 | $(p-\mu)^T\Sigma^{-1}(p-\mu)$ |
| 乘积压缩 $\bar{\alpha}_t$ | Alpha blending $T_i$ | $\prod (1-x_j)$ |
| Score Function | Rendering Gradient | $\nabla(\text{标量函数})$ |
| Langevin Dynamics | Gradient Descent | $x_{k+1}=x_k+\frac{\epsilon}{2}s(x_k)+\sqrt{\epsilon}z$ vs $\theta_{k+1}=\theta_k-\eta\nabla L$ |
| ELBO分解 | Composite Loss | 重构项 + KL/正则化 |

---

## 📝 验证清单（完成学习后自查）

- [ ] **Ch01**：能独立从KL≥0推导ELBO，理解VAE的两项博弈
- [ ] **Ch02**：理解为什么直接采样断梯度，重参数化为何有效
- [ ] **Ch03**：能从递归展开推导闭式解 $q(x_t|x_0)$，计算 $\bar{\alpha}_t$
- [ ] **Ch04**：证明"预测噪声=Score Matching"等价性，理解SNR的意义
- [ ] **Ch05**：从ELBO展开推导DDPM损失，证明$L_T\to 0$
- [ ] **Ch06**：理解SDE作为离散扩散的连续极限，ODE solver少步采样原理
- [ ] **Ch07**：能系统性对比VAE/Diffusion与3DGS的数学结构

---

> **最后更新**：2026-04-18  
> **作者**: Ember 🔥 (基于 first-principles 推导 + PyTorch 验证)  
> **许可**：MIT — 自由分享、修改、用于教学

🔥 **恭喜完成 VAE & Diffusion 系列！** 🎉