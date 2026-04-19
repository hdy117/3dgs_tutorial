# VAE & Diffusion — 从潜空间到扩散模型的数学基础

> **目标读者**：理解 3DGS 训练流程，想深入生成模型数学原理的人  
> **前置知识**：概率论与信息论（Ch01-11）、优化理论（Ch01-06）  
> **核心问题**：如何从噪声中"雕刻"出有意义的结构？VAE 和 Diffusion 给出了两种不同的答案。

---

## 📚 系列大纲与章节链接

### Part 1: VAE — 变分自编码器
| Ch | 标题 | 状态 | 核心内容 |
|----|------|------|----------|
| Ch01 | **潜空间与变分推断** | ✅ 完成 | ELBO 推导、KL散度、变分后验架构直觉 |
| Ch02 | **ELBO 优化与重参数化技巧** | ✅ 完成 | 为什么直接采样断梯度？重参数化的第一性原理证明 |
| Ch03 | **高斯 KL 解析解与 VAE 训练流程** | ⏳ 待写 | ELBO 三项分解、数值示例验证 |

### Part 2: Diffusion — 扩散模型基础
| Ch | 标题 | 状态 | 核心内容 |
|----|------|------|----------|
| Ch04 | **前向加噪过程：马尔可夫链的暴力美学** | ✅ 完成 | 逐步加噪、马尔可夫假设、闭式解 $q(x_t|x_0)$ |
| Ch05 | **反向去噪：Score Matching 与扩散目标函数** | ✅ 完成 | Score function、噪声预测等价性、DDPM loss 推导 |
| Ch06 | **DDPM 完整推导：从 VAE 视角理解扩散** | ✅ 完成 | ELBO 逐项分解、T层潜变量VAE、L_T→0简化 |

### Part 3: 进阶与衔接
| Ch | 标题 | 状态 | 核心内容 |
|----|------|------|----------|
| Ch07 | **Score-based Models & SDE 视角** | ✅ 完成 | Langevin Dynamics、Fokker-Planck、Probability Flow ODE |
| Ch08 | **VAE/Diffusion × 3DGS：深度衔接分析** | ⏳ 待写 | 高斯二次型统一表达、Alpha blending与乘积压缩类比 |

---

## 📐 各章结构模板（统一）

每章遵循以下结构：
1. **🎯 Problem-driven opening** — 真实场景 + "关键问题"框
2. **📐 Part N: 直觉理解** — 对话式讲解，不跳步
3. **🔥 Part M: First-principles 推导** — 编号步骤从公理到结论，boxed 数值结果
4. **💻 Part K: PyTorch 验证代码** — 可运行的 Snippet
5. **🗺️ Connection Table** — Concept | 3DGS对应 | 为什么重要
6. **🎓 Summary** — Boxed 核心公式 + Key Insights + Next Step Teaser
7. **📚 Exercises** — 三级难度（基础/进阶/3DGS关联）+ `<details>` 折叠提示

---

## 🔗 与已有知识体系的衔接

| 已有系列 | VAE/Diffusion 对应概念 |
|----------|----------------------|
| 概率论 Ch05 (KL divergence) | VAE 的 KL 正则化项直接就是 KL(q‖p) |
| 概率论 Ch09 (Bayesian inference) | VI = 近似贝叶斯推断；ELBO = marginal likelihood 下界 |
| 优化理论 Ch02 (Chain rule/autograd) | VAE/Diffusion 反向传播都是链式法则的复杂应用 |
| 数值线性代数 Ch05 (SVD) | PCA → VAE 潜空间压缩直觉一致 |

---

## 🎯 学习路径图

### **最低限度路径**（3章够用）
```
Ch01 (VAE 直觉) → Ch04 (Diffusion 前向过程) → Ch06 (DDPM = VAE特例)
```

### **标准路径**（理解全部数学）
```
Ch01 → Ch02 → Ch03 → Ch04 → Ch05 → Ch06 → Ch07
```

### **深度路径**（含 3DGS 衔接）
```
Ch01 → Ch02 → Ch03 → Ch04 → Ch05 → Ch06 → Ch07 → Ch08 (3DGS deep dive)
```

---

## 📝 文件命名约定
```
NN_中文标题.md
```
