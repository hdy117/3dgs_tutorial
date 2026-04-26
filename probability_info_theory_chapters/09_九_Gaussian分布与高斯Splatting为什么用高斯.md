# Ch09 — Gaussian分布 vs 高斯Splatting：为什么世界选择了高斯？

> **本章目标**：从数学本质到工程实现，理解 3D Gaussian Splatting 中"Splat"为什么是高斯函数。  
> **前置知识**：Ch01-Ch08（概率与信息论全套）。  
> **核心问题**：如果可以用任意形状建模场景，为什么 3DGS 偏偏选了高斯？

---

## 🎯 问题驱动：选择 Splat 形状的权衡

### 💡 生活例子：像素画 vs 水彩画的"边界"问题

想象你要用两种不同的方式在屏幕上绘制一个圆形图标：

**方法 A — 像素艺术（硬边界）**：
你只能使用正方形的像素块来拼出一个圆。每个像素要么完全亮、要么完全暗，没有渐变。结果就是锯齿状的边缘（antialiased？不存在的）：

```
0 0 1 1 0 0    ← 第2行：突然从0变到1！不连续！
0 1 1 1 1 0    ← 第3行：边界清晰但锯齿状
1 1 1 1 1 1
0 1 1 1 1 0
0 0 1 1 0 0
```

**方法 B — 水彩画（高斯模糊）**：
你用画笔轻轻涂抹，中心最深、边缘渐淡。没有突然的跳变，整个圆是平滑过渡到背景的：

```
. . 0.1 0.3 . .     ← 边缘渐变
. 0.2 0.6 0.6 0.2 .   ← 从深到浅自然过渡
0.3 0.6 0.9 0.9 0.6 0.3  ← 中心最浓
. 0.2 0.6 0.6 0.2 .
. . 0.1 0.3 . .
```

**关键问题**：如果这个圆形图标需要被"缩放"或"旋转"——哪种方法效果更好？
- 像素艺术放大后会出现严重的马赛克和锯齿（因为不连续）
- 水彩画可以无限平滑地缩放（因为处处可微）

> 🎯 **这就是 3DGS 选择高斯的核心原因**：渲染本质上要对 Splat 做各种几何变换（平移、旋转、缩放），只有处处光滑的函数才能在变换后保持视觉质量。硬边界的"像素画式"Splat在优化过程中会产生不连续的梯度，导致训练崩溃！

### 场景 1：你在设计一个显式场景表示器

想象你要用一堆"小球"拼出一个 3D 物体。每个球有一个形状函数 $S(\mathbf{x})$，决定它有多"亮"（或贡献多少颜色）。

| 候选形状 | 数学表达 | 优点 | 缺点 |
|---------|----------|------|------|
| **球体 (Top-Hat)** | $\mathbb{I}(\|\mathbf{x}\| \leq r)$ | 简单，硬边界 | 不连续 → 不可微 → 无法用梯度下降优化 |
| **双曲线 (Blobby)** | $(1 - \|\mathbf{x}\|^2)^2$ | 平滑 | 尾部衰减慢 → 计算开销大 |
| **高斯 (Gaussian)** | $\exp(-\|\mathbf{x}\|^2/2)$ | 无限光滑，快速衰减 | ? |

**关键问题**：为什么高斯是工程上的最优解？它有什么数学特权？

---

## 📐 Part 1: 高斯的三大"超能力"

### 超能力 1: Fourier Transform = 另一个高斯 (自相似性)

$$\boxed{\mathcal{F}\left\{ e^{-ax^2} \right\}(\omega) = \sqrt{\frac{\pi}{a}} e^{-\omega^2/(4a)}}$$

**推导思路（极坐标技巧）**：
1. 高斯函数的 Fourier Transform 也是高斯形式。
2. **自相似性**：高斯在空间域和频域都是光滑的，没有突然的截断（不像球体，它的频谱有 $\sin(\omega)/\omega$ 振荡——这就是 Gibbs 现象，会导致振铃伪影）。

**3DGS 意义**：渲染本质上是滤波操作。高频信号（尖锐边缘）在 Fourier 域会"泄露"到低频区域 → 模糊/伪影。**高斯是天然的抗混叠滤波器 (Anti-aliasing Filter)** —— 它在空间域和频域都做了完美的平滑！

### 超能力 2: 中心极限定理 (CLT) — "万物皆趋向高斯"

$$\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} N(0, 1) \quad (n \to \infty)$$

**直觉**：任何由大量独立微小随机效应叠加而成的物理量 → 自动服从高斯分布。

### 超能力 3: 最大熵原理 — "最诚实的假设"

在所有均值 $\mu$、方差 $\sigma^2$ 固定的连续分布中，**高斯具有最大 Shannon 熵**。
$$\boxed{H(X) \leq \frac{1}{2}\ln(2\pi e \sigma^2)}$$

这意味着：如果你只知道一个量的"中心位置"和"波动范围"，却不想引入任何额外假设——选高斯是最安全的！它不会"捏造"不存在的信息。

---

## 🔥 Part 2: 从第一性原理推导 — 为什么 Splat 用 $\mathcal{N}(\mu, \Sigma)$？

### 3DGS Splat 的核心公式

每个 Gaussian Splat 的形状函数是：
$$\boxed{\mathcal{G}(\mathbf{x}) = e^{-(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}}$$

注意：**这不是标准高斯 PDF**（没有前面的归一化常数）。3DGS 只保留了指数部分——因为渲染时我们关心的是"相对贡献权重"，而不是绝对概率密度。

### 🔥 推导：为什么用指数二次型？

假设我们要构造一个形状函数 $S(\mathbf{x})$，满足以下物理要求：

| 要求 | 数学约束 |
|------|----------|
| 1. 中心最亮 | $S(\boldsymbol{\mu}) = \max$, $\nabla S(\boldsymbol{\mu}) = 0$ |
| 2. 各向同性衰减（无方向偏好） | $S$ 只依赖于 $\|\mathbf{x}-\boldsymbol{\mu}\|$ |
| 3. 可微且无限光滑 (C^∞) | 所有阶导数存在 |
| 4. 快速衰减（远处贡献小） | $S(\mathbf{x}) \to 0$ as $\|\mathbf{x}\| \to \infty$ |

**Step 1 — 球对称性假设**：
设 $r^2 = (\mathbf{x}-\boldsymbol{\mu})^T \mathbf{M} (\mathbf{x}-\boldsymbol{\mu})$，其中 $\mathbf{M}$ 是对称正定矩阵（控制形状）。

**Step 2 — 指数形式满足 C^∞ + 快速衰减**：
考虑 $S(r) = e^{-f(r)}$。如果取 $f(r) \propto r^2$，则：
- $r=0$: $e^0 = 1$ (最大值)
- $r \to \infty$: $e^{-\infty} \to 0$ (快速衰减)
- 所有导数都存在（指数函数的特性）

**Step 3 — 为什么不是 $e^{-r}$ 或 $e^{-r^4}$？**
- $e^{-r}$: 在中心不可微（尖点）。 ❌
- $e^{-r^4}$: 衰减太慢，远处计算开销大。 ❌
- $e^{-r^2}$: **黄金平衡** — 快速衰减但不至于"切断"了物理连续性。 ✅

### ✅ Boxed Result：Splat 的协方差矩阵 $\mathbf{\Sigma}$

在 3DGS 中，$\mathbf{\Sigma}$ 参数化为旋转 R 和缩放 s：
$$\boxed{\mathbf{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T, \quad \text{其中 } \mathbf{S}=\text{diag}(s_x,s_y,s_z)}$$

这保证了 $\mathbf{\Sigma}$ 始终是**对称正定矩阵 (SPD)** —— 高斯分布的定义域要求。

---

## 💻 Part 3: PyTorch 验证 — 可视化 Splat 的形状

```python
import torch
import math
import matplotlib.pyplot as plt

# ============================================================
# 1. 高斯 Splat 形状函数 (3DGS 的核心)
# ============================================================
def gaussian_splat(x, mu, Sigma_inv):
    """计算 exp(-0.5 * (x-mu)^T Sigma_inv (x-mu))"""
    diff = x - mu
    # 注意：3DGS 中用的是没有归一化常数的高斯函数！
    exponent = -0.5 * torch.sum(diff * (Sigma_inv @ diff.unsqueeze(-1)), dim=0)
    return torch.exp(exponent.squeeze())

# ============================================================
# 2. 可视化不同 Sigma 的 Splat 形状 (2D 截面)
# ============================================================
print("=== Gaussian Splat 形状可视化 ===")

x = torch.linspace(-3, 3, 200)
y = torch.linspace(-3, 3, 200)
X, Y = torch.meshgrid(x, y, indexing='ij')
coords = torch.stack([X.flatten(), Y.flatten()], dim=1) # (40000, 2)

# --- 案例 A: 圆形 Splat ($\Sigma = I$) ---
mu_A = torch.tensor([0., 0.])
Sigma_inv_A = torch.eye(2)
splat_A = gaussian_splat(coords, mu_A, Sigma_inv_A).reshape(200, 200)

# --- 案例 B: 拉伸 Splat ($\Sigma$ 沿 x 轴拉长) ---
mu_B = torch.tensor([0., 0.])
# Sigma = [[4, 0], [0, 1]] → Sigma_inv = [[0.25, 0], [0, 1]]
Sigma_inv_B = torch.diag(torch.tensor([0.25, 1.]))
splat_B = gaussian_splat(coords, mu_B, Sigma_inv_B).reshape(200, 200)

# --- 案例 C: 旋转 Splat ($\Sigma$ 包含非对角项) ---
mu_C = torch.tensor([0., 0.])
angle = math.pi / 4 # 45度
R = torch.tensor([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle),  math.cos(angle)]])
S_diag = torch.diag(torch.tensor([2.0, 0.5])) # x拉长，y压缩
Sigma_C = R @ S_diag @ S_diag @ R.T
Sigma_inv_C = torch.linalg.inv(Sigma_C)
splat_C = gaussian_splat(coords, mu_C, Sigma_inv_C).reshape(200, 200)

# ============================================================
# 3. 验证：高斯的 Fourier Transform 也是高斯 (数值验证)
# ============================================================
print("\n=== Fourier Transform: Gaussian → Gaussian ===")
from scipy.fft import fft2, fftshift

fft_A = fftshift(fft2(splat_A))
fft_B = fftshift(fft2(splat_B))

# 高斯的傅里叶变换参数：$\sigma_{freq} = 1/(2\pi \sigma_{space})$
# 数值上验证频谱也是"高斯型" (衰减平滑无振铃)
max_A = torch.max(torch.abs(fft_A)).item()
print(f"Splat A (圆形): FFT 峰值归一化 = {max_A:.4f}")

# --- 对比：球体形状 (Top-Hat) 的频谱会有振铃伪影 ---
hat_A = (X**2 + Y**2 <= 1.0).float() # 半径为1的硬圆
fft_hat = fftshift(fft2(hat_A))
print(f"Top-Hat:      FFT 峰值归一化 = {torch.max(torch.abs(fft_hat)).item():.4f}")

# 💡 高斯的频谱衰减极其平滑（无振铃），这正是抗混叠的核心！
```

---

## 🗺️ Part 4: 与 3DGS 的衔接点 — Splat 的本质

| 数学概念 | 3DGS 实现 | 为什么这样设计？ |
|---------|-----------|-----------------|
| **高斯 PDF $\mathcal{N}(\mu, \Sigma)$** | $e^{-(x-\mu)^T\Sigma^{-1}(x-\mu)}$ (去掉归一化常数) | 渲染时只需相对权重，不需要绝对概率密度。省去了计算 $|\Sigma|$ 的开销！ |
| **协方差矩阵 $\mathbf{\Sigma}$** | R × S · S^T × R^T (极分解/对称正定参数化) | 保证优化过程中 $\mathbf{\Sigma}$ 始终合法（正定），不会"崩溃"。 |
| **Fourier 自相似性** | 高斯 Splat → 频域平滑衰减 | **抗混叠**：Splat 在空间域是连续的，频谱不会有 Gibbs 振铃伪影。这是比体积渲染 (NeRF) 更快更清晰的关键！ |

### 💡 关键洞察：3DGS = "用梯度下降优化高斯参数"

```python
# 伪代码：3DGS 优化的核心循环
for step in range(iterations):
    # 1. 前向：渲染图像 (所有 Splat 叠加)
    image = sum(Splat_i(x) * color_i for i in gaussians)
    
    # 2. Loss: 和真实图像的差距 (L1/SSIM)
    loss = L1(image, gt_image)
    
    # 3. 反向传播：调整每个 Splat 的 μ, Σ, R, S
    loss.backward()           # ←←← 链式法则！(Ch07)
    optimizer.step()          # ←←← 梯度下降！(Ch02-05)
```

**整个训练过程 = 在参数空间中，让每个 Gaussian Splat "找到"自己最合适的中心 (μ) 和形状 (Σ)，以最小化信息论意义上的 Loss。**

---

## 🎓 本章小结

### 核心公式

$$\boxed{\mathcal{G}(\mathbf{x}) = e^{-(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}}$$

$$\boxed{\text{高斯 FT: } \mathcal{F}\{e^{-ax^2}\} = \sqrt{\frac{\pi}{a}} e^{-\omega^2/(4a)}}$$

### 关键洞察

> **高斯不是"随便选的"** —— 它是满足 C^∞ + 快速衰减 + 抗混叠的最简数学形式。
> 
> **中心极限定理**保证了真实世界中大量微小误差的叠加自然趋向高斯分布；**最大熵原理**保证了在缺乏先验知识时，高斯是最"诚实"的选择。
> 
> **3DGS 的核心创新**：把传统渲染中的"硬边界"替换为"无限光滑的高斯函数"，使得整个管线可微，可以用梯度下降端到端训练！

---

## 📚 习题

### ✅ 基础题

**9.1** 为什么高斯的 Fourier Transform 还是高斯？这对信号处理有什么意义？
<details>
<summary>💡 提示</summary>
因为指数二次型在积分变换下保持形式不变。这意味着高斯是唯一的"自相似滤波器"，不会在频域引入额外的形状畸变。
</details>

**9.2** 3DGS 中的 Splat 函数为什么去掉了归一化常数 $\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}$？
<details>
<summary>💡 提示</summary>
渲染是相对权重叠加，不需要绝对概率密度。去掉常数省去了行列式计算和开方开销——在实时渲染中这很关键！而且 Alpha Blending 本身做了归一化。
</details>

### 🔥 进阶题

**9.3** 证明：如果 $\mathbf{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$，则 $\mathbf{\Sigma}$ 始终是对称正定的 (SPD)。
<details>
<summary>💡 提示</summary>
对称性：$(RS S^T R^T)^T = RS S^T R^T$。正定性：对任意非零向量 v，$v^T \Sigma v = \|S^T R^T v\|^2 > 0$（因为 R, S 满秩）。
</details>

### 💡 3DGS 关联题

**9.4** (Part 3 Step 10 预告)：假设你有一个点云，每个点的颜色 $c_i$ 和位置 $p_i$ 已知。如果你用 Gaussian Splat 去拟合这些点，从最大似然估计 (MLE) 的角度看，你应该优化什么参数？

---

> **Ch09 完成！** 🔥  
> 
> Part 3 下一站：**Ch10 — MLE (最大似然估计)** —— 如何用观测数据反推 Gaussian 的最佳参数？直接说 "继续"。
