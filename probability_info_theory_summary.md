# 📇 概率与信息论 — 一页纸总结卡片 (Cheat Sheet)

> **适用场景**：复习、面试速查、3DGS 变体设计时的直觉参考。  
> **对应教程**：`probability_info_theory_chapters/` (Ch01-Ch11)。

---

## 🧱 Part 1: 概率论基础 (Ch01 - Ch04)
### "不确定性的数学建模"

| 概念 | Boxed 公式 / 定义 | 3DGS 中的角色 |
|------|------------------|--------------|
| **随机变量** | PMF: $\sum p(k)=1$ \| PDF: $\int f(x)\,dx=1$ | Splat = 连续型 RV (位置/形状) |
| **期望 E[X]** | $E[X] = \sum x \cdot p(x)$ \| 线性性: $E[aX+bY]=aE+ bE$ | Gaussian 的中心 $\mu$ (典型值) |
| **方差 Var(X)** | $\text{Var}(X) = E[(X-\mu)^2] = E[X^2]-(E[X])^2$ | Gaussian 的协方差 $\Sigma$ (波动度) |
| **贝叶斯定理** | $$P(A|B) = \frac{P(B|A)\cdot P(A)}{\sum P(B|X_i)}$$ | 根据像素颜色推断 Splat 来源 |
| **高斯分布** | $N(\mu,\sigma^2): f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\dots}$ | **Splat 的核心形状函数** |
| **Beta/Dirichlet** | Beta: $\text{Beta}(\alpha,\beta)$ \| Dirichlet: $∏x_i^{α-1}$ | 概率参数的不确定性 (如 Alpha 混合) |

---

## 📊 Part 2: 信息论核心 (Ch05 - Ch08)
### "量化不确定性与信息距离"

| 概念 | Boxed 公式 / 定义 | 直觉解读 |
|------|------------------|----------|
| **Shannon熵 H(X)** | $$H(X) = -\sum p(x)\log_2 p(x)$$ | "最优压缩的比特数" (不确定性总量) |
| **自信息 I(x)** | $I(x) = -\log_2 P(x)$ | 事件发生时的"惊讶程度" |
| **交叉熵 H(P,Q)** | $$H(P,Q) = -\sum p(x)\log q(x)$$ | 用模型 Q 描述真实 P 的编码成本 (**Loss**) |
| **KL 散度** | $$D_{KL}(P||Q) = \sum p(x)\ln(p/q) \geq 0$$ | "分布间差异" (非对称距离，非负性由 Jensen 保证) |
| **互信息 MI** | $I(X;Y) = H(Y) - H(Y|X)$ | 知道 X 后，Y 的不确定性减少了多少 (**粘性**) |

---

## 🔗 Part 3: 与 3DGS 的衔接 (Ch09 - Ch11)
### "从数学公式到渲染管线"

#### 🎯 核心映射表
| 数学对象 | 3DGS 实现 | 物理/几何含义 |
|----------|-----------|--------------|
| **Gaussian PDF** $\mathcal{N}(\mu, \Sigma)$ | $e^{-(x-\mu)^T\Sigma^{-1}(x-\mu)}$ (去常数) | Splat 的亮度权重分布 |
| **MLE (最大似然)** | $\arg\max_\theta P(D|\theta)$ | 梯度下降优化目标：让观测数据最可能出现 |
| **高斯噪声假设** | $\to \text{MSE Loss (L2)}$ | 适合平滑、低噪场景 |
| **Laplace 噪声假设** | $\to \text{MAE Loss (L1)}$ | 适合真实图像（鲁棒抗异常值） |

#### 🔥 "大道至简" 三大洞察
> **洞察 1: 3DGS = MLE + Gaussian Shape**  
> 训练过程就是在参数空间中，寻找一组 Gaussian Splat ($\mu, \Sigma$)，使得渲染出的像素序列出现概率最大 (MLE)。
> 
> **洞察 2: Loss = 编码成本近似**  
> L1/Loss 本质上是负对数似然。最小化 Loss ≈ 让模型用最少的比特描述真实世界（信息论压缩极限）。
> 
> **洞察 3: KL 散度是分布的"距离"**  
> $D_{KL}(P||Q) \geq 0$。如果 Q (模型预测) 漏掉了 P (真实场景) 的区域，Loss 会爆炸！这解释了为什么高斯不能太稀疏或太尖锐。

---

## 🧠 终极知识链 (The Golden Thread)
$$\text{随机变量} \xrightarrow{\text{量化}} \text{熵(H)} \xrightarrow{\text{比较}} \text{KL散度} \xrightarrow{\text{优化}} \text{MLE/Loss} \xrightarrow{\text{实现}} \text{3DGS Splatting}$$

### 📚 下一步推荐
- **复习路径**: Ch05(熵) → Ch06(交叉熵) → Ch10(MLE) —— 最短理解 "Loss 为什么有效"。
- **进阶方向**: 
    1. **优化理论** (凸优化/拉格朗日乘子) -> 解释 SGD/Adam 的数学基础。
    2. **微分几何** (曲率/测地线) -> 理解高分辨率重建中的"拉伸与畸变"问题。
