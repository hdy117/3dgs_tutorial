# VAE & Diffusion 教程

_从变分自编码器到扩散模型，完整推导生成式 AI 的两大支柱_

---

## 📖 教程导览

本教程采用 **第一性原理** 框架：从"为什么要用潜空间？"、"为什么加噪能生成图像？"这些根本问题出发，一步步推导出 VAE、DDPM、Score-based Models 等核心模型的完整数学。

**前置知识建议：**
- 📐 **概率与信息论** — Ch01~Ch08（KL散度、贝叶斯、常见分布）
- 📐 **优化理论** — Ch01~Ch03（梯度下降基础）
- 📐 **微积分** — Ch07（链式法则/反向传播）、Ch05（偏导数）

> 详细前置要求见各章节顶部的 `> **前置知识**` 标注。

---

## 🎯 章节列表

| 章节 | 内容 | 核心问题 |
|------|------|----------|
| [Ch01 · 潜空间与变分推断](./vae_diffusion_chapters/01_一_潜空间与变分推断.md) | $p(z|x)$ 不可计算 → 用近似分布逼近 | 为什么需要压缩？后验怎么算？ |
| [Ch02 · ELBO 优化与重参数化](./vae_diffusion_chapters/02_二_ELBO优化与重参数化技巧.md) | $\mathcal{L}_{ELBO}$ 的推导 & 重参数化技巧 | 梯度怎么穿过随机采样？ |
| [Ch03 · 前向加噪与马尔可夫链](./vae_diffusion_chapters/03_三_前向加噪与马尔可夫链.md) | $q(x_t|x_{t-1})$ — DDPM 的前向过程 | 为什么逐步加噪？ |
| [Ch04 · 反向去噪与 Score Matching](./vae_diffusion_chapters/04_四_反向去噪与Score_Matching.md) | $\nabla_x \log p(x)$ — 分数匹配的本质 | 怎么从噪声恢复数据？ |
| [Ch05 · DDPM 完整推导](./vae_diffusion_chapters/05_五_DDPM完整推导与VAE视角.md) | DDPM = VAE + 高斯先验 | DDPM 为什么能被 VAE 解释？ |
| [Ch06 · Score-based Models & SDE](./vae_diffusion_chapters/06_六_Score_based_Models与SDE视角.md) | 连续时间扩散的 SDE 统一框架 | 扩散 → 随机微分方程 |
| [Ch07 · VAE/Diffusion 与 3DGS 衔接](./vae_diffusion_chapters/07_七_VAE与Diffusion和3DGS的深度衔接.md) | 生成式模型 × 3DGS 的交叉应用 | 它们怎么互相增强？ |

---

## 🔗 学习路径

```
Ch01: 潜空间 + 变分推断          ← 起点：为什么要压缩数据？
    │
Ch02: ELBO + 重参数化技巧         ← VAE 的核心：梯度穿过采样
    │
Ch03: 前向加噪马尔可夫链          ← Diffusion 的起点：逐步破坏信息
    │
Ch04: 反向去噪 + Score Matching   ← Diffusion 的核心：学分数场
    │
Ch05: DDPM 完整推导 (VAE视角)     ← VAE + Diffusion = DDPM
    │
Ch06: Score-based + SDE          ← 统一框架：连续时间扩散
    │
Ch07: VAE/Diffusion × 3DGS       ← 交叉应用（进阶）
```

---

## 📚 与 3DGS 教程的关系

| 主题 | 涉及章节 | 衔接点 |
|------|----------|--------|
| **高斯分布** | Ch01 (变分推断) / Ch05 (DDPM) | 3DGS 的高斯椭球 vs Diffusion 的高斯噪声 |
| **KL散度** | Ch02 (ELBO) | VAE 的 KL 项 ≈ 3DGS 的协方差正则化直觉 |
| **贝叶斯推断** | Ch01 | 3DGS 参数估计 vs 贝叶斯后验 |
| **梯度/链式法则** | 全章节通用 | 所有模型都依赖 autograd |

> VAE & Diffusion 是独立学习模块，不需要走完 3DGS 主线也可以直接学。但如果你已经读过概率论和优化理论的基础章节，理解速度会快很多。

---

## 🗺️ 回到总导航

← [3DGS 教程主入口](./3dgs_tutorial.md)
