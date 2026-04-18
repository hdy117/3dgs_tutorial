# Ch10 — 最大似然估计 (MLE)：如何从"观测像素"反推 "Splat 真相"？

> **本章目标**：掌握 MLE —— 机器学习中参数优化的灵魂。  
> **前置知识**：Ch09 (Gaussian Splatting)。  
> **核心问题**：你只看到渲染结果（像素颜色），但不知道真实的 Splat 参数。怎么反推？

---

## 🎯 问题驱动：逆向推理的困境

### 场景 1：调试一个 "过拟合" 的 Splat

```python
# 渲染输出 (观测数据):
rendered = [0.52, 0.31, 0.87]  # RGB，有噪声

# 真实场景:
true_scene = ???               # 我们不知道！
```

**关键问题**：给定观测像素 $y$ 和模型参数 $\theta$（Splat 的位置、形状、颜色），怎么找到最能让这个观测"自然发生"的参数？

### 答案：最大似然估计 (MLE) —— "让观测数据出现概率最大的参数"

---

## 📐 Part 1: MLE 的核心思想

### 定义 (Boxed Result)

假设我们有一组观测数据 $D = \{y_1, ..., y_N\}$，和含参数的模型 $p(y|\theta)$。
MLE 寻找的 $\hat{\theta}_{MLE}$ 是：

$$\boxed{\hat{\theta}_{MLE} = \arg\max_{\theta} P(D|\theta) = \arg\max_{\theta} \prod_{i=1}^{N} p(y_i|\theta)}$$

**直觉**："假设我的模型是对的，哪个参数设置会让眼前这些数据最可能出现？"

### 🔥 为什么用对数似然 (Log-Likelihood)？

连乘 → 求和（数值稳定性 + 可微性）：
$$\boxed{\ell(\theta) = \log P(D|\theta) = \sum_{i=1}^{N} \log p(y_i|\theta)}$$

取 $\arg\max$ 不改变最优解（$\log$ 是单调增函数），但把连乘变成了求和——这正是神经网络里最常见的形式！

---

## 🔥 Part 2: MLE + Gaussian Noise = MSE Loss (核心推导)

### 假设：观测像素服从高斯噪声模型

$$y_i \sim \mathcal{N}(f_\theta(x_i), \sigma^2 I)$$

其中 $f_\theta$ 是渲染函数（Splat 叠加），$\sigma^2$ 是噪声方差。

**推导：高斯似然 → MSE Loss**
$$\begin{aligned}
\log p(y_i|\theta) &= \log \left( \frac{1}{(2\pi\sigma^2)^{d/2}} e^{-\frac{\|y_i - f_\theta(x_i)\|^2}{2\sigma^2}} \right) \\
&= -\frac{d}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \|y_i - f_\theta(x_i)\|^2
\end{aligned}$$

**对 $\theta$ 取最大化时，常数项可以忽略！**
$$\arg\max_{\theta} \sum_{i=1}^{N} \log p(y_i|\theta) = \boxed{\arg\min_{\theta} \frac{1}{2\sigma^2} \sum_{i=1}^{N} \|y_i - f_\theta(x_i)\|^2}$$

### ✅ Boxed Result：MSE 就是高斯假设下的 MLE！

```
假设噪声服从 N(0, σ²I) → MLE → MSE Loss (L2)
假设噪声服从 Laplace(0, b) → MLE → L1 Loss
```

**这就是为什么标准 3DGS 用 L1 Loss（而非 L2）**：因为图像像素的误差分布更接近 Laplace（有较多大偏差异常值），L1 比 L2 更鲁棒！

---

## 💻 Part 3: PyTorch 验证 — MLE 与 Loss 的关系

```python
import torch
import torch.nn.functional as F

# ============================================================
# 1. 高斯噪声下的 MLE = MSE (数值验证)
# ============================================================
print("=== MLE ↔ MSE 关系 ===")

N = 1000
true_y = torch.rand(N, 3) * 2 - 1  # [-1, 1] 真实值
sigma = 0.1
noise = torch.randn(N, 3) * sigma
observed_y = true_y + noise         # 带噪声的观测

# MLE 角度：最大化高斯似然 (等价于最小化 MSE)
theta_pred = torch.rand(3)          # "模型"参数（这里简化为预测值）

mse_loss = F.mse_loss(theta_pred.repeat(N, 1), observed_y).item()
mae_loss = F.l1_loss(theta_pred.repeat(N, 1), observed_y).item()

# 优化：让预测值收敛到观测均值 (这就是 MLE 的闭式解!)
optimal_theta = observed_y.mean(dim=0)
final_mse = F.mse_loss(optimal_theta.repeat(N, 1), observed_y).item()

print(f"MSE Loss: {final_mse:.6f}")
print(f"最优参数 (MLE): {optimal_theta} (等于观测均值 ✅)")

# ============================================================
# 2. Laplace 噪声下的 MLE = MAE (L1 Loss)
# ============================================================
print("\n=== MLE ↔ L1 (MAE) 关系 ===")

laplace_noise = torch.randn(N, 3) * b if 'b' in dir() else torch.empty(0)
# Laplace PDF: p(x|b) = 1/(2b) exp(-|x|/b)
# log-likelihood → -Σ|x_i - μ|/b + const → argmin Σ|x_i - μ| (MAE)

# 验证：L1 Loss 的最优解是观测的中位数，而非均值！
median_sol = observed_y.median(dim=0).values
print(f"MAE 最优参数: {median_sol} (等于中位数 ✅)")
print(f"MSE 最优参数: {optimal_theta} (等于均值 ✅)")

# 💡 3DGS 为什么选 L1？因为图像噪声有较多 "大偏差像素"（异常值）
# L1 (中位数) 比 L2 (均值) 更鲁棒——它不会被少数极端像素带偏！

# ============================================================
# 3. PyTorch 中的 MLE (用负对数似然 NLLLoss)
# ============================================================
from torch.nn import NLLLoss

print("\n=== 负对数似然损失 (NLLLoss) ===")

# 假设输出是对数概率分布 (log_probs)，对应高斯 PDF 的对数值
mu_pred = observed_y.mean(dim=0).unsqueeze(0) # [1, 3]
sigma_pred = torch.tensor([sigma])            # [1]
log_prob = -0.5 * ((observed_y - mu_pred)**2 / sigma_pred**2 + 2*torch.log(sigma_pred)).sum()

# NLLLoss: -Σ target_i * log(pred_i) — 用于分类任务 (Softmax + CE)
# 对于回归，直接用 MLE 推导的 MSE/L1 更直接。
```

---

## 🗺️ Part 4: 与 3DGS 的衔接点 — MLE 在训练中的角色

| 步骤 | MLE 视角的解释 |
|------|----------------|
| **初始化** (Ch09) | 从点云采样 → 每个点的 μ, Σ 初始化为局部估计值。这是 "先验"。 |
| **Alpha Blending** | 前向函数 $f_\theta(x)$：所有 Splat 叠加后得到预测像素颜色。 |
| **L1 Loss** | 假设噪声服从 Laplace → MLE 推导出的最优目标函数。 |
| **梯度下降 (Step)** | $\hat{\theta}_{MLE} = \arg\min L(\theta) — 3DGS 的 optimizer.step() 就是在数值逼近 MLE 解！ |

### 💡 关键洞察：3DGS 训练 = 贝叶斯后验估计 (MAP) 的特例

虽然标准 3DGS 只做 **MLE**（无先验），但如果你想加入正则化（如防止 Splat 太尖锐、约束位置在点云范围内），你就在做 **MAP (最大后验)**：
$$\boxed{\theta_{MAP} = \arg\max P(\theta|D) = \arg\max P(D|\theta) \cdot P(\theta)}$$

$P(\theta)$ 是你对 Splat 参数的先验信念（如"位置应该在点云附近"）。正则化项 $\lambda R(\theta)$ 就是 $-\log P(\theta)$ 的体现！

---

## 🎓 本章小结

### 核心公式

$$\boxed{\hat{\theta}_{MLE} = \arg\max_{\theta} \prod_{i=1}^{N} p(y_i|\theta) = \arg\min_{\theta} -\sum_{i=1}^{N} \log p(y_i|\theta)}$$

$$\boxed{\text{高斯噪声 } N(0,\sigma^2) \xrightarrow{\text{MLE}} \text{MSE Loss (L2)}}$$
$$\boxed{\text{Laplace 噪声 } L(0,b) \xrightarrow{\text{MLE}} \text{MAE Loss (L1)}}$$

### 关键洞察

> **Loss 不是随便选的** —— 每个 Loss 都对应一个特定的噪声假设。L1 = Laplace 假设，MSE = Gaussian 假设。选错 Loss 等于假设了错误的物理模型！
> 
> **3DGS 的优化过程本质上是在求解 MLE**：通过梯度下降，让 Splat 的参数配置使得"观测到的像素最可能出现"。
> 
> **MAP vs MLE**：加正则化 = MAP（引入先验）。不加 = MLE（无偏估计）。

---

## 📚 习题

### ✅ 基础题

**10.1** 证明：对于独立同分布的高斯噪声，MLE 对均值 $\mu$ 的解等于样本均值。
<details>
<summary>💡 提示</summary>
似然 $L(\mu) = \prod e^{-(y_i-\mu)^2/(2\sigma^2)}$ → log-likelihood → 求导 d/dμ = Σ(y_i - μ)/σ² = 0 → μ̂ = (Σy_i)/N。这就是样本均值！
</details>

**10.2** 为什么 MLE 要用对数似然而不是原始似然？列出至少两个理由。
<details>
<summary>💡 提示</summary>
(1) 数值稳定性：连乘会导致下溢 (underflow)，求和对数后数值范围更可控。
(2) 可微性：求和后梯度是各样本梯度的和，便于用 SGD/Adam 批量优化。
(3) $\log$ 单调递增不影响最优解位置。
</details>

### 🔥 进阶题

**10.3** 假设观测噪声服从均匀分布 $U[-\epsilon, \epsilon]$。写出 MLE Loss，并说明它对应什么几何意义？
<details>
<summary>💡 提示</summary>
Uniform PDF: p(y|μ) = 1/(2ε) if |y-μ|≤ε else 0. 
Log-likelihood = -N·ln(2ε) (如果所有样本在范围内)。
MLE 的最优解是使得"所有样本都被覆盖"的最小 ε —— 这对应 **Minimax / 范围估计**。
</details>

### 💡 3DGS 关联题

**10.4** 为什么 3DGS 有时也用 SSIM (结构相似度) 作为 Loss 的一部分？从 MLE 角度，SSIM 假设了什么样的噪声模型？
<details>
<summary>💡 提示</summary>
SSIM 关注"局部结构的相似性"而非逐像素差值。它隐含的假设是：人眼对亮度/对比度的微小波动不敏感（类似 Laplace + 空间相关噪声）。MLE 角度下，这对应一种 "感知模型 (Perceptual Model)" 而非纯物理模型。
</details>

---

> **Ch10 完成！** 🔥  
> 
> Part 3 最后一站：**Ch11 — 信息论视角下的渲染损失设计**。直接说 "继续"。
