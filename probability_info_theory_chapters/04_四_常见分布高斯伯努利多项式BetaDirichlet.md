# Ch04 — 常见分布：3DGS 中直接用到的"概率积木"

> **本章目标**：掌握 6 个核心分布的定义、性质和与 3DGS 的直接关联。  
> **前置知识**：Ch01-03（随机变量、期望/方差、条件概率、贝叶斯）。  
> **核心问题**：如果随机变量是"积木"，那每种分布是什么形状？为什么有些积木可以互相拼接？

---

## 🎯 问题驱动：为什么需要记住这些分布？

### 场景 1：你在写一个 3DGS 变体

```python
# 你想加入不确定性感知 —— 不直接优化参数，而是采样参数空间
for iteration in range(10000):
    # 从后验中采样高斯的位置和形状
    mu_sample = ...     # 来自什么分布？
    sigma_sample = ...  # 来自什么分布？
    
    rendered = render_with_samples(mu_sample, sigma_sample)
```

**关键问题**：你应该用什么分布来建模"不确定的高斯参数"？不同场景（颜色、权重、位置）需要不同的分布。

### 答案：每种物理现象对应一个最优分布

| 物理量 | 最自然的分布 | 原因 |
|--------|-------------|------|
| 高斯 Splat 的位置/形状 | **正态/多元高斯** | 中心极限定理，大量微小噪声的叠加 |
| 每个像素是否被覆盖（是/否） | **伯努利** | 二元结果 |
| N 个 Splat 对 M 个像素的贡献计数 | **多项式** | 多次独立试验的多类别结果 |
| 概率参数的不确定性（如 α） | **Beta / Dirichlet** | 定义在 [0,1] 或 simplex 上，共轭先验 |

---

## 📐 Part 1: 伯努利分布 — "硬币"的数学模型

### 💡 生活例子：是或否的二元决策

生活中充满了只有两种可能结果的事件。举几个例子：

- **考试及格/不及格**：一次考试的结果要么是 pass（通过），要么是 fail（不通过）。如果你复习得不错，及格的概率 p = 0.85。
- **Splat 覆盖像素**：在 3DGS 中，一个 Splat 是否覆盖了某个特定像素 — 是/否两种结果。如果 Splat 很大且中心靠近像素，这个概率可能高达 α = 0.95；如果 Splat 很小且在边缘，概率可能只有 0.10。
- **邮件是垃圾邮件还是正常邮件**：一封新收到的邮件要么是 spam（垃圾），要么是 ham（正常）。基于发件人地址、关键词等特征，你估算 spam 的概率为 p = 0.73。

所有这些"非即否"的场景，都可以用伯努利分布来建模 — $X \in \{0, 1\}$，其中 1 代表"成功/是"，0 代表"失败/否"。

### 定义

$$\boxed{X \sim \text{Bern}(p): \quad P(X=1) = p, \quad P(X=0) = 1-p}$$

PMF 的统一写法（用指数形式，方便推广到多项式）：
$$P(X=x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

### 性质

| 量 | 公式 | 推导 |
|----|------|------|
| **期望** | $E[X] = p$ | $E[X] = 1·p + 0·(1-p) = p$ |
| **方差** | $\text{Var}(X) = p(1-p)$ | $\text{Var} = E[X^2]-(E[X])^2 = p-p^2 = p(1-p)$ |

### 💡 3DGS 关联：Alpha blending 中的二元决策

$$C_{\text{pixel}} = \sum_{i=1}^{N} w_i c_i, \quad w_i = \alpha_i \prod_{j<i}(1-\alpha_j)$$

每个 Splat i 对像素颜色的贡献可以用一个伯努利变量 $B_i$ 近似建模：
- $B_i=1$: "Splat i 主导了这个像素"，概率 ≈ $\alpha_i \prod_{j<i}(1-\alpha_j)$
- $B_i=0$: "被前面的 Splat 遮挡了"

---

## 📐 Part 2: 多项式分布 — N 次试验的多类别计数

### 💡 生活例子：餐厅点餐统计

假设你经营一家餐厅，顾客每顿饭只能选择三种菜品之一：川菜、粤菜或西餐。你记录了最近 100 位顾客的点餐记录：
- 川菜被点了 50 次
- 粤菜被点了 30 次
- 西餐被点了 20 次

这就是一个**多项式分布**的场景 — $n = 100$ 次独立试验，每个试验有 $k = 3$ 个可能的类别，每个类别有一个概率：$p_1 = 0.5, p_2 = 0.3, p_3 = 0.2$。

用随机变量表示：
- $X_1$ = "川菜被点的次数"（理论期望 = 100 × 0.5 = 50）
- $X_2$ = "粤菜被点的次数"（理论期望 = 100 × 0.3 = 30）
- $X_3$ = "西餐被点的次数"（理论期望 = 100 × 0.2 = 20）

关键约束：$X_1 + X_2 + X_3 = n = 100$ — 总共只有 100 位顾客。这就是多项式分布的精髓：**在固定次数的独立试验中，统计每个类别出现了多少次**。

再举一个 3DGS 相关的例子：如果你有 N=500 个 Splat 和一个像素被渲染了 M=1000 帧，你可以用多项式来统计"第 i 个 Splat 主导了多少帧"。

### 定义

$$\boxed{X \sim \text{Mult}(n, p_1,...,p_k): \quad P(X_1=n_1,...,X_k=n_k) = \frac{n!}{n_1!\cdots n_k!} \prod_{i=1}^{k} p_i^{n_i}}$$

其中 $\sum_i n_i = n$, $\sum_i p_i = 1$。

**直觉**：掷 $n$ 次骰子，统计每个面出现的次数。$X_i$ = "第 i 个类别出现了几次"。

### 性质

| 量 | 公式 |
|----|------|
| **期望** | $E[X_i] = n p_i$ |
| **方差** | $\text{Var}(X_i) = n p_i (1-p_i)$ |
| **协方差** | $\text{Cov}(X_i, X_j) = -n p_i p_j \quad (i≠j)$ |

⚠️ 协方差为负！因为总数 $n$ 固定——一个类别多了，别的必然少。

### 💡 3DGS 关联：多相机训练中的样本分配

在多视角训练中，每个 Splat 可能被 M 个相机观测到。如果用多项式建模"第 i 个 Splat 被多少相机看到"，这就是 $\text{Mult}(M, p_1,...,p_N)$。

---

## 🔥 Part 3: 高斯分布 — 概率论的 "e^(-x²)"（最重要！）

### 💡 生活例子：为什么身高服从高斯分布？

想象你站在大学操场上，随机选一个同学测量他的身高。你会用什么来预测这个人的身高？

**直觉答案**：大多数中国成年男性的身高在 **165-180cm** 之间（均值约 172cm）。特别矮（< 150cm）或特别高（> 195cm）的人很少。**这种"中间密集、两头稀疏"的形状就是高斯分布！**

具体来说，假设男性身高 $H \sim N(172, 6^2)$：
- $P(166 < H < 178) \approx 68\%$（均值 ±1σ）→ 大多数人在这个范围内
- $P(160 < H < 184) \approx 95\%$（均值 ±2σ）→ 几乎所有人都在这个范围
- $P(H > 190) \approx P(Z > 3) \approx 0.13\%$ → 超过 1米90 的人极少

> 🎯 **为什么是"钟形曲线"？** — 因为身高由成千上万个微小因素决定：遗传基因（来自父母双方各约 23,000 个基因）、营养状况、运动习惯、睡眠、环境...根据**中心极限定理**，大量独立微小随机效应的叠加必然趋向高斯分布。这就是为什么自然界中"大多数东西都服从高斯"！

### 💡 第二个例子：传感器噪声的高斯特性

相机拍照时，每个像素的测量值并不是精确的光强 — 它包含了各种随机噪声（热噪声、光子散粒噪声等）。这些噪声的来源非常多且微小：
- 电子电路的热扰动
- 光子的量子涨落
- 温度波动
- 电磁干扰

根据中心极限定理，所有这些微小噪声的叠加 → **像素测量误差近似服从高斯分布** $N(0, \sigma^2)$。这就是为什么在计算机视觉和 3DGS 中，我们几乎总是假设噪声是高斯的 — 不是因为高斯"完美"，而是因为它是大量独立小效应的**通用极限**。

### 定义：一维高斯

$$\boxed{X \sim N(\mu, \sigma^2): \quad f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}}$$

**三个参数中的两个**：均值 $\mu$（中心）、标准差 $\sigma$（宽度）。方差是 $\sigma^2$。

### 定义：多元高斯 — 3DGS 的核心！

$$\boxed{\mathbf{X} \sim N(\boldsymbol{\mu}, \mathbf{\Sigma}): \quad f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\mathbf{\Sigma}|^{1/2}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}}$$

其中 $\mathbf{\Sigma}$ 是 $d×d$ **协方差矩阵**（对称正定）。

- $|\mathbf{\Sigma}|$: 矩阵的行列式，反映"体积"
- $\mathbf{\Sigma}^{-1}$: 逆协方差矩阵（Precision matrix / Information matrix）

### 💡 关键洞察：为什么 Splat 用高斯？

**理由 1 — 中心极限定理 (CLT)**：
$$\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{当 } n→∞$$

大量独立微小随机效应（传感器噪声、光照变化、几何误差）的叠加 → **自然趋向高斯分布**。这是高斯成为"默认选择"的根本原因。

**理由 2 — 最大熵性质**：
在所有均值固定为 $\mu$、方差固定为 $\sigma^2$ 的连续分布中，**高斯分布具有最大的 Shannon 熵**（不确定性最大）。也就是说——在不引入额外假设的前提下，高斯是最"诚实"的选择。

### PyTorch 中的多元高斯操作

```python
import torch
from torch.distributions import MultivariateNormal, Normal

# === 一维高斯 ===
normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
print(f"E[X]={normal.mean.item()}, Var(X)={normal.variance.item()}")

log_prob = normal.log_prob(torch.tensor(1.5))  # ln(f(1.5))
pdf_val = log_prob.exp()                        # f(1.5)

# === 多元高斯（3DGS Splat 的形状）===
mu = torch.tensor([0.0, 0.0, 0.0])             # Gaussian 中心位置
cov = torch.diag(torch.tensor([0.1, 0.2, 0.05]))  # 协方差矩阵（对角）

mvn = MultivariateNormal(mu, cov)
sample = mvn.sample()                             # 从分布采样一个三维点
log_prob_3d = mvn.log_prob(sample)                # 对数概率密度
entropy = mvn.entropy()                           # 微分熵: ½ln((2πe)³|Σ|)

print(f"多元高斯熵: {entropy.item():.4f}")
print(f"行列式 |Σ|: {cov.det().item():.6f} （Splat 的'体积'）")
```

---

## 🔥 Part 4: Beta 分布 — [0,1] 上的概率参数建模器

### 💡 生活例子：你对一个餐厅评分的"信心"

想象你去了一家新开的咖啡店，喝了一杯。你不确定它到底有多好——你可能觉得大概有 70% 的好感度（p ≈ 0.7），但你也不确定。Beta 分布就是用来建模这种**"对概率的概率"**的。

具体来说，假设你对一家咖啡店的真实满意度 $p$ 有一个先验信念：
- **完全不确定**（Alpha=1, Beta=1）→ Beta(1,1) = 均匀分布 → 你觉得 p 可以是 [0,1] 之间任何值，概率都一样。这相当于你从来没去过这家店，对它的评分一无所知。
- **偏向好评但有些犹豫**（Alpha=7, Beta=3）→ Beta(7,3)，均值 ≈ 0.7 → 你觉得这家店大概有 70% 的好感度，但不确定——可能是 50% 也可能是 90%。
- **非常确信好评**（Alpha=20, Beta=2）→ Beta(20,2)，均值 = 0.9 → 你喝过好多次了，基本确定这家店很赞，满意度在 80%-100% 之间。

> 🎯 **关键直觉**：Beta 分布的参数 (α, β) 可以理解为"看到的好结果数"和"坏结果数" + 1。Alpha=7, Beta=3 就像你喝了 6 杯好咖啡、3 杯一般 → 你对满意度 p 的信念是 Beta(7, 3)。

### 💡 第二个例子：A/B 测试中的转化率

假设你在做电商网站的 A/B 测试。版本 A（红色按钮）和版本 B（蓝色按钮）各有不同的点击率，但你不知道哪个更高。你对版本 A 的点击率 p_A 有一个先验信念：Beta(5, 15) → 均值 ≈ 0.25，说明你认为"大约 25% 的用户会点击红色按钮"。

现在你观察到 100 个用户中只有 30 人点击了红色按钮（观测值 = 0.30）。根据 Beta-Bernoulli 共轭性：
$$\text{后验} = \text{Beta}(5 + 30, 15 + 70) = \text{Beta}(35, 85)$$
新均值 = 35/120 ≈ 0.292 — 从 0.25 更新到了 0.29。

这就是 Beta 分布在在线实验中的强大之处：**它让你用一个简洁的参数对来表达"我对某个概率有多确信"**，而观测数据只需简单地加到参数上就能完成贝叶斯更新。

### 定义

$$\boxed{X \sim \text{Beta}(\alpha, \beta): \quad f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad x \in [0,1]}$$

其中 $B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\β)}{\Gamma(\alpha+\beta)}$ 是 Beta 函数（归一化常数）。

### 性质

| 量 | 公式 |
|----|------|
| **期望** | $\frac{\alpha}{\alpha+\beta}$ |
| **方差** | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| **众数** | $\frac{\alpha-1}{\alpha+\beta-2} \quad (\alpha,β > 1)$ |

### 💡 直觉：Beta 是"概率的概率分布"

- α=1, β=1 → 均匀分布 $U(0,1)$（完全不确定）
- α=5, β=1 → 偏向 1（确信概率接近 1）
- α=1, β=5 → 偏向 0（确信概率接近 0）

**3DGS 关联：Alpha 参数的不确定性建模**

渲染中的 alpha 值 $\alpha_i \in [0, 1]$。如果你想表达"这个 Splat 的透明度大约是 0.7，但不确定"——Beta(7, 3) 就是一个自然的先验（均值 ≈ 0.7）。

### Beta-Bernoulli 共轭 — 贝叶斯更新的优雅配对

$$\boxed{\text{先验 } p \sim \text{Beta}(\alpha,\beta), \quad \text{似然 } X \sim \text{Bern}(p) \Rightarrow \text{后验 } p|X \sim \text{Beta}(\alpha+X, \beta+1-X)}$$

**推导**：
- 先验: $p(p) ∝ p^{\alpha-1}(1-p)^{\beta-1}$
- 似然: $P(X|x|p) = p^x(1-p)^{1-x}$
- 后验 ∝ 先验 × 似然: $∝ p^{\alpha+x-1}(1-p)^{\beta+1-x-1} \Rightarrow \text{Beta}(\alpha+x, \beta+1-x)$

**关键洞察**：共轭意味着**更新后的形式不变**——只是参数简单增加了观测数据。这是贝叶斯统计中最方便的设计之一。

---

## 🔥 Part 5: Dirichlet 分布 — Beta 的多维推广（simplex 上的分布）

### 💡 生活例子：餐桌上菜的搭配比例

想象你在一家自助餐厅，每次可以装三个菜到自己的餐盘里。假设你选择川菜的倾向是 α₁ = 3、粤菜的倾向是 α₂ = 5、西餐的倾向是 α₃ = 2（这些数字反映你的口味偏好）。那么你的**实际取菜比例 (x₁, x₂, x₃)** 服从一个 Dirichlet(3, 5, 2) 分布。

关键特征：
- **x₁ + x₂ + x₃ = 1** — 你的餐盘是满的，三种菜的总量占满了整个盘子。这就像 simplex（单纯形）的定义——所有分量非负且和为 1。
- **期望比例**：川菜 ≈ 3/10 = 30%，粤菜 ≈ 5/10 = 50%，西餐 ≈ 2/10 = 20% — 粤菜是你最爱的。
- **每次取菜的组合都不一样**（随机性），但长期平均会趋近这些期望值。

再举一个 3DGS 相关的例子：在颜色混合中，如果一个像素的最终颜色是由红、绿、蓝三个通道按不同比例混合而成，且比例之和为 1（归一化），那么这些通道的权重分布就可以用 Dirichlet 来建模。

### 💡 第二个例子：投票选举的得票比例

假设一个有三位候选人的选举，你作为选民对三位候选人的支持度分别为 α₁ = 5, α₂ = 3, α₃ = 2（基于民调数据）。那么最终**总得票比例 (p₁, p₂, p₃)** 服从 Dirichlet(5, 3, 2)：
- 期望：候选人 A ≈ 50%，B ≈ 30%，C ≈ 20%
- 但每次抽样（模拟选举结果）都会产生不同的比例 — 有时 B 可能反超 A

这就是 Dirichlet 的精髓：**它建模的是"一组互斥类别的比例分布"，其中所有比例加起来等于 1**。

### 定义

$$\boxed{\mathbf{X} \sim \text{Dir}(\alpha_1,...,\alpha_k): \quad f(x_1,...,x_k) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{i=1}^{k} x_i^{\alpha_i-1}, \quad \sum_i x_i = 1, x_i ≥ 0}$$

其中 $B(\boldsymbol{\alpha})$ 是多维 Beta 函数。$\mathbf{x}$ 在 **simplex**（单纯形）上取值——所有分量非负且和为 1。

### 性质

| 量 | 公式 |
|----|------|
| **期望** | $E[X_i] = \frac{\alpha_i}{\sum_j \alpha_j}$ |
| **方差** | $\text{Var}(X_i) = \frac{\alpha_i(\alpha_0-\alpha_i)}{\alpha_0^2(\alpha_0+1)}$ 其中 $\alpha_0=\sum_j\alpha_j$ |

### 💡 3DGS 关联：多类别颜色的混合权重

如果一个像素由 K 种颜色成分混合而成（RGB + alpha → 4 个通道，或者更复杂的频谱表示），且要求混合权重和为 1——Dirichlet 是自然的建模选择。

### Dirichlet-Multinomial 共轭

$$\boxed{\text{先验 } \mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha}), \quad \text{似然 } \mathbf{X}|\mathbf{p} \sim \text{Mult}(n, \mathbf{p}) \Rightarrow \text{后验 } \mathbf{p}|\mathbf{X} \sim \text{Dir}(\boldsymbol{\alpha}+\mathbf{X})}$$

与 Beta-Bernoulli 完全相同的形式——只是从标量变成了向量。

---

## 🧪 Part 6: 数值示例 — Boxed Result

### 案例：Beta 先验 + 观测数据 → 后验

假设你在训练 3DGS，某个 Splat 的 alpha 值不确定。你给它 Beta(2, 8) 先验（均值 = 0.2，预期这个 Splat 比较透明）。

**观测**：10 次独立渲染中，7 次该 Splat"主导"了某像素（伯努利观测）。

后验：
$$\boxed{\text{先验 Beta}(2,8) + \text{数据 } X=7 \Rightarrow \text{后验 Beta}(9,11)}$$

后验均值：$\boxed{\frac{9}{9+11} = 0.45}$ — 从期望的 0.2 更新到了实际的 0.45。

### PyTorch 验证

```python
import torch
from torch.distributions import Beta, Bernoulli

# === Beta-Bernoulli 共轭 ===
alpha_prior, beta_prior = 2, 8
observed_successes = 7
observed_failures = 3

alpha_post = alpha_prior + observed_successes     # = 9
beta_post = beta_prior + observed_failures         # = 11

posterior_mean = alpha_post / (alpha_post + beta_post)
prior_mean = alpha_prior / (alpha_prior + beta_prior)

print(f"Alpha 先验均值: {prior_mean:.4f}")
print(f"Alpha 后验均值: {posterior_mean:.4f} （观测到 7/10 次主导）")
print(f"贝叶斯更新方向: {'→ ↑ (更不透明)'}" if posterior_mean > prior_mean else "→ ↓")

# === 多元高斯（Splat 形状采样）===
from torch.distributions import MultivariateNormal

mu = torch.zeros(3)
cov = torch.diag(torch.tensor([0.5, 0.3, 0.1]))  # x 方向最扩散
mvn = MultivariateNormal(mu, cov)

samples = mvn.sample((1000,))
print(f"\n多元高斯采样:")
print(f"  样本均值: {samples.mean(dim=0)}")     # → ≈ [0, 0, 0]
print(f"  真实 μ:   {mu}")
print(f"  样本协方差:\n{samples.T.cov()}")         # → ≈ diag([0.5, 0.3, 0.1])

# === Dirichlet（Simplex 采样）===
from torch.distributions import Dirichlet

alpha = torch.tensor([2., 5., 3.])  # K=3 类，偏向第 2 类
dirichlet = Dirichlet(alpha)
sample_simplex = dirichlet.sample()   # [0.14, 0.52, 0.34]（和为 1）

print(f"\nDirichlet 采样 (simplex):")
print(f"  {sample_simplex} → sum={sample_simplex.sum().item():.6f}")
```

---

## 🗺️ Part 7: 分布家族总览表（概率论速查）

| 分布 | 类型 | 定义域 | PMF/PDF | E[X] | Var(X) | 3DGS 用途 |
|------|------|--------|---------|------|--------|----------|
| **伯努利** Bern(p) | 离散 | {0,1} | $p^x(1-p)^{1-x}$ | p | p(1-p) | Alpha blending 二元决策 |
| **多项式** Mult(n,p) | 离散 | $\sum n_i=n$ | 多项展开系数 | npᵢ | npᵢ(1-pᵢ) | 多相机观测计数 |
| **高斯** N(μ,σ²) | 连续 | ℝ | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | μ | σ² | **Splat 的核心形状函数** |
| **多元高斯** N(μ,Σ) | 连续 | ℝᵈ | $\frac{1}{(2π)^{d/2}|Σ|^{1/2}}e^{-\frac{1}{2}(x-μ)ᵀΣ⁻¹(x-μ)}$ | μ | Σ | **3D Gaussian Splat 的协方差建模** |
| **Beta** Beta(α,β) | 连续 | [0,1] | $\frac{x^{α-1}(1-x)^{β-1}}{B(α,β)}$ | α/(α+β) | 见正文 | Alpha/权重参数的不确定性 |
| **Dirichlet** Dir(α) | 连续 | simplex | $\frac{1}{B(α)}∏xᵢ^{αᵢ-1}$ | αᵢ/Σαⱼ | 见正文 | 多类混合权重的联合建模 |

---

## 🎓 本章小结

### 核心公式（记住这些就够用了）

$$\boxed{\text{高斯 PDF: } \quad f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}}$$

$$\boxed{\text{Beta-Bernoulli 共轭: Beta}(α,β) + X \sim \text{Bern} → \text{后验 Beta}(α+X, β+1-X)}$$

$$\boxed{\text{Dirichlet-Multinomial 共轭: Dir}(α) + X \sim \text{Mult} → \text{后验 Dir}(α+X)}$$

### 关键洞察

> **高斯是"最诚实的分布"** — 在只知道均值和方差的情况下，选择高斯意味着你没有添加任何额外假设（最大熵原理）。
> 
> **共轭先验 = "更新形式不变"** — Beta-Bernoulli 和 Dirichlet-Multinomial 让贝叶斯后验仍然是同一种分布，只是参数简单累加。这在高斯-伽玛过程等复杂模型中是设计原则。
> 
> **3DGS 的核心**：Splat = 多元高斯函数 $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \mathbf{\Sigma})$ — 位置用均值建模，形状用协方差矩阵建模。整个渲染管线就是对这些高斯的"叠加采样"。

### 📝 Part 1 小结：概率论 → 3DGS 映射完成！

| Part 1 主题 | 学到了什么 | 对应 3DGS |
|-------------|-----------|-----------|
| Ch01: 随机变量与分布 | PMF vs PDF，离散 vs 连续 | Splat = 连续高斯函数，Alpha blending = 离散求和 |
| Ch02: 期望、方差 | E[X], Var(X), 线性性, Jensen不等式 | μ=E[位置], Σ=Var(形状) |
| Ch03: 条件概率与贝叶斯 | P(A|B), Bayes定理, 共轭先验 | 根据像素颜色推断 Splat 来源 |
| Ch04: 常见分布 | Bernoulli, Multinomial, Gaussian, Beta, Dirichlet | Splat 形状 = Gaussian，权重不确定性 = Beta/Dirichlet |

---

### 📚 Part 2 预告：信息论核心！

现在你知道概率是什么、有哪些工具了——但你怎么量化"一个分布包含了多少信息？" "两个分布有多不同？" **下一章进入信息论：熵、交叉熵、KL散度**。这些是损失函数的灵魂 🔥

---

## 📚 习题

### ✅ 基础题

**4.1** 写出以下分布在给定参数下的期望和方差：
- (a) Bern(0.3)
- (b) N(5, 4)（注意这里的第二个参数是方差）
- (c) Beta(3, 7)

<details>
<summary>💡 提示</summary>
(a) E=0.3, Var=0.21
(b) E=5, Var=4
(c) E=3/10=0.3, Var=21/(100×11)=0.01909
</details>

**4.2** 验证 Beta(1,1) = U(0,1)。即写出 PMF/PDF，说明它退化为均匀分布。

<details>
<summary>💡 提示</summary>
Beta(α=1,β=1): f(x) = x⁰(1-x)⁰/B(1,1) = 1/Γ(2)=1 → 常数 1 on [0,1]。这就是均匀分布！
</details>

### 🔥 进阶题

**4.3** 证明多元高斯的协方差矩阵 $\mathbf{\Sigma} = E[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^T]$。从定义出发推导，并说明为什么它必须是对称正定的。

<details>
<summary>💡 提示</summary>
直接展开：$E[(\mathbf{X}-μ)(\mathbf{X}-μ)^T] = E[\mathbf{XX}^T - \mathbf{X}μ^T - μ\mathbf{X}^T + μμ^T] = E[\mathbf{XX}^T] - μμ^T - μμ^T + μμ^T = E[\mathbf{XX}^T]-μμ^T$。对称性来自外积的转置性质。正定性来自协方差矩阵的定义（任意方向上的方差都≥0）。
</details>

**4.4** 如果 $\mathbf{X} \sim N(\boldsymbol{\mu}, \mathbf{\Sigma})$，且你做了一个线性变换 $\mathbf{Y} = A\mathbf{X} + b$，证明 $\mathbf{Y} \sim N(A\boldsymbol{\mu}+b, A\mathbf{\Sigma}A^T)$。

<details>
<summary>💡 提示</summary>
期望：线性性 → E[Y]=AE[X]+b=Aμ+b。协方差：Var(Y)=E[(AX+b-Aμ-b)(...)ᵀ]=AE[(X-μ)(X-μ)ᵀ]Aᵀ=AΣAᵀ。正态分布在线性变换下保持正态（特征函数论证）。
</details>

### 💡 3DGS 关联题

**4.5** 3DGS 中每个 Splat 的协方差矩阵 $\mathbf{\Sigma}$ 通常参数化为旋转矩阵 R 和缩放向量 s：$\mathbf{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$。
- (a) 从多元高斯的角度，R 控制了哪些方向扩散、s 控制了多少。这对应 $\mathbf{\Sigma}$ 的什么数学分解？
- (b) 如果 R 是单位矩阵且 $s = [1, 0.5, 0.2]$，这个 Splat 在三维空间中的形状是什么？

---

> **Part 1 完成！** 概率论基础系列（Ch01-Ch04）全部写完 ✅  
> 
> Part 2 将进入信息论：熵、交叉熵、KL散度、互信息。准备好开始吗 🔥
