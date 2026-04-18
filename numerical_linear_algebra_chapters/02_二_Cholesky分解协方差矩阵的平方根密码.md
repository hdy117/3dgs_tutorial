# Ch02 — Cholesky 分解：协方差矩阵的"平方根密码"

> **本章目标**：理解为什么对称正定 (SPD) 矩阵永远可以写成 $LL^T$，以及它与 3DGS Splat 参数化的深层联系。  
> **前置知识**：Ch01 (SVD, LU/QR, 条件数)。  
> **核心问题**：如果我知道一个矩阵是对称且正定的——我能比一般方法快多少倍来求逆或采样？

---

## 🎯 问题驱动：协方差矩阵的"特殊体质"

### 场景 1：3DGS 中为什么 $\boldsymbol{\Sigma}$ 永远合法？

```python
# 在 3DGS 中，每个 Splat 有一个协方差矩阵 Σ。
# 我们参数化为: Σ = R · S · S^T · R^T
# 
# 问题：为什么不直接优化 Σ 的 6 个元素，而是用 (R, S)？
# 因为直接优化无法保证 Σ 始终正定！而 Cholesky 分解告诉我们——
# "任何合法的协方差矩阵都能被拆成一个'下三角 + 它的转置'"。
```

**关键问题**：Cholesky 分解如何利用 SPD 矩阵的特殊结构，把 $O(n^3)$ 的通用求逆加速到几乎减半？它在几何上对应什么操作？

### 答案：**Cholesky 分解 (LLᵀ / LDLᵀ)** —— "协方差矩阵的平方根"

---

## 📐 Part 1: Cholesky — SPD 矩阵的专属拆解

### Boxed Result：定义与存在性定理

对于任意 **对称正定 (SPD) 矩阵** $\mathbf{A}$，存在唯一的 **下三角矩阵 $\mathbf{L}$（对角线元素 > 0）**，使得：
$$\boxed{\mathbf{A} = \mathbf{L}\mathbf{L}^T}$$

这被称为 Cholesky 分解。有时也写成 $\mathbf{A} = \mathbf{L}\mathbf{D}\mathbf{L}^T$（$\mathbf{D}$ 是对角矩阵），称为 **LDLᵀ 分解**。

### 💡 Boxed Result：与 SVD / 特征值分解的关系

| 分解 | 形式 | SPD 下的等价性 |
|------|------|---------------|
| **特征值分解** | $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$ | $\mathbf{L} = \mathbf{V}\sqrt{\boldsymbol{\Lambda}}$（取平方根） |
| **Cholesky** | $\mathbf{A} = \mathbf{L}\mathbf{L}^T$ | 更紧凑！不需要旋转矩阵 $\mathbf{V}$，直接给出三角因子。 |

### 💡 核心洞察：为什么 Cholesky 更快？
- **LU**: $O(\frac{2}{3}n^3)$ — 通用分解，无结构利用。
- **Cholesky**: $O(\frac{1}{3}n^3)$ — SPD 的对称性让计算量减半！

---

## 🔥 Part 2: 从第一性原理推导 Cholesky (LDLᵀ)

### Boxed Result：递归拆解过程

假设 $\mathbf{A}$ 是 $n \times n$ SPD。将其分块为：
$$\begin{bmatrix} a_{11} & \mathbf{r}^T \\ \mathbf{r} & \mathbf{B} \end{bmatrix} = \begin{bmatrix} l_{11} & 0 \\ \mathbf{l} & \tilde{\mathbf{L}} \end{bmatrix} \begin{bmatrix} l_{11} & \mathbf{l}^T \\ 0 & \tilde{\mathbf{L}}^T \end{bmatrix}$$

**Step 1 — 提取第一行/列：**
- $l_{11} = \sqrt{a_{11}}$（因为 $a_{11}>0$，SPD 保证！）
- $\mathbf{l} = \frac{1}{l_{11}}\mathbf{r}$

**Step 2 — Schur Complement (舒尔补)：**
$$\boxed{\tilde{\mathbf{B}} = \mathbf{B} - \mathbf{l}\mathbf{l}^T}$$
$\tilde{\mathbf{B}}$ **依然是 SPD！**（这是 Cholesky 能递归下去的关键！）

**Step 3 — 对 $\tilde{\mathbf{B}}$ 重复上述过程。** ∎

### 💡 几何直觉：Cholesky = "逐步剥洋葱"
每一步提取一个下三角因子，把矩阵的"第一层外壳"剥掉，剩下的内核仍然是 SPD。最终得到 $\mathbf{L}$。这就像用正交基逐步投影到坐标轴上——但保留了三角结构带来的计算效率！

---

## 💻 Part 3: PyTorch 验证 — Cholesky 与协方差采样

```python
import torch
import numpy as np

# ============================================================
# 1. Cholesky 分解实战 (SPD 矩阵专属)
# ============================================================
print("=== Cholesky 分解：协方差矩阵的平方根 ===")

# --- Step A: 构造一个合法的 SPD 矩阵（模拟 Splat Σ）---
# 用 R·S·S^T·R^T 生成，保证正定
R = torch.tensor([[0.8, -0.6], [0.6, 0.8]]) # 旋转矩阵 (近似)
S_diag = torch.tensor([2.0, 1.5])           # 缩放

Sigma_3d = R.t() @ torch.diag(S_diag**2) @ R  # Σ = R^T D R (SPD!)
print(f"构造的协方差矩阵 Σ:\n{Sigma_3d.detach().numpy()}")

# --- Step B: Cholesky 分解 A = L·L^T (PyTorch 内置 cholesky) ---
try:
    L = torch.linalg.cholesky(Sigma_3d.float())
    print(f"\nCholesky L (下三角):\n{L.detach().numpy()}")
    
    # 验证: L·L^T == Σ?
    reconstructed = L @ L.T
    error = ((reconstructed - Sigma_3d)**2).sum().item()
    print(f"重建误差 ||LL^T - Σ||²: {error:.10f} ✅")
except torch.linalg.LinAlgError as e:
    print(f"⚠️ 矩阵不正定! 错误: {e}")

# --- Step C: Cholesky vs SVD 对比 (速度 & 精度) ---
print("\n=== Cholesky vs SVD 对比 ===")
_, s, _ = torch.linalg.svd(Sigma_3d.float())
print(f"SVD 奇异值 (σ): {s}")

# SVD 也能给出"平方根": A^{1/2} = U√Σ V^T
sqrt_via_SVD = torch.matmul(torch.matmul(R.t(), torch.diag(np.sqrt(s))), R) # 近似

# Cholesky L 直接给出了三角因子，计算逆矩阵更快!
inv_L = torch.linalg.inv(L) # 下三角求逆 O(n²/3)
Sigma_inv_via_Chol = inv_L.T @ inv_L

print(f"通过 Cholesky L⁻¹ 计算的 Σ⁻¹:\n{Sigma_inv_via_Chol.detach().numpy()}")

# ============================================================
# 2. 用 Cholesky 采样多元高斯 (3DGS 的 Splat 参数化核心)
# ============================================================
print("\n=== Cholesky + 多元高斯采样 ===")

mu = torch.tensor([0., 0.])
L_chol = L # Cholesky factor of Σ

# 从标准正态 Z ~ N(0, I) 采样，然后变换: X = μ + LZ
Z = torch.randn((1000, 2))
samples = mu.unsqueeze(0) + Z @ L.T  # ←←← 关键公式!

print(f"采样均值: {samples.mean(dim=0)} (应≈ [0, 0])")
print(f"采样协方差:\n{samples.t().cov()}") # 应≈ Σ

# 💡 这就是为什么 3DGS 的初始化可以直接从点云采样!
# 给定一个点的局部协方差 Σ，Cholesky L 给出了"如何把单位球变成椭球"的变换矩阵!
```

---

## 🗺️ Part 5: 与 3DGS 的衔接点 — $\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$ 的本质

### Boxed Result：3DGS 参数化就是 Cholesky + 旋转

在 3DGS 中，协方差矩阵被显式参数化为：
$$\boxed{\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T, \quad \text{其中 } \mathbf{S}=\text{diag}(s_x,s_y,s_z)}$$

**这与 Cholesky 的关系**：
- $\mathbf{L}_{Chol} = \mathbf{R}\mathbf{S}$（如果忽略旋转的三角化细节）。
- **3DGS 选择用 R, S 而不是直接优化 $\boldsymbol{\Sigma}$**，因为：
    1. **保证正定性**：$\mathbf{S}\mathbf{S}^T$ 始终对角正定，乘以正交阵 R 后仍为正定。
    2. **可解释性**：R 是旋转方向，S 是主轴长度 —— 这直接对应椭球的几何形状！

### 💡 核心洞察：Cholesky 是"隐式参数化"的等价形式

如果你直接优化 $\mathbf{L}$（下三角矩阵），那么 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$。
3DGS 只是把这个思想推广到了 **R (旋转) + S (缩放)** —— 本质上是一样的！只不过 R, S 提供了更直观的几何语义（主轴方向与长度）。

---

## 🎓 本章小结

### 核心公式 (Boxed)

$$\boxed{\text{Cholesky: } \mathbf{A} = \mathbf{L}\mathbf{L}^T, \quad \mathbf{L} \text{ 下三角, diag}(L)>0}$$

$$\boxed{\mathbf{x} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \xrightarrow{\text{采样}} \boldsymbol{\mu} + \mathbf{L}\mathbf{z}, \quad \mathbf{z} \sim N(0,\mathbf{I})}$$

### 关键洞察

> **Cholesky 是 SPD 矩阵的"平方根"** —— 它把协方差矩阵拆解为一个下三角因子和它的转置，计算量仅为通用 LU 的一半。
> 
> **3DGS 的参数化 $\mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$ 是 Cholesky 的几何变体** —— R 控制主轴方向，S 控制主轴长度。这保证了优化过程中 Σ 始终合法（正定）。
> 
> **Cholesky + Z ~ N(0,I) = Splat 初始化**：通过 $X = \mu + LZ$，我们可以从点云生成符合特定协方差分布的随机样本。这就是 3DGS "高斯球"的诞生方式！

---

## 📚 习题

### ✅ 基础题

**2.1** 证明：如果 $\mathbf{A}$ 是 SPD，则它的对角线元素 $a_{ii} > 0$。
<details>
<summary>💡 提示</summary> 取标准基向量 $\mathbf{e}_i$，则 $\mathbf{e}_i^T \mathbf{A} \mathbf{e}_i = a_{ii}$。因为 A 正定，对所有非零向量 v, $v^TA v > 0$ → $a_{ii}>0$。
</details>

**2.2** Cholesky 分解 $\mathbf{L}\mathbf{L}^T = \begin{bmatrix}4 & 2\\2 & 5\end{bmatrix}$，求 $\mathbf{L}$。
<details>
<summary>💡 提示</summary> $l_{11}=2, l_{21}=1, l_{22}=\sqrt{5-1^2}=2$ → $\mathbf{L}=\begin{bmatrix}2&0\\1&2\end{bmatrix}$。
</details>

### 🔥 进阶题

**2.3** (Schur Complement)：为什么在 Cholesky 的递归过程中，$\tilde{\mathbf{B}} = \mathbf{B} - \mathbf{l}\mathbf{l}^T$ 始终保持 SPD？
<details>
<summary>💡 提示</summary> Schur complement 保持了原矩阵的正定性（从分块矩阵的行列式性质可证）。它是 Cholesky 能逐层递归的核心保证。
</details>

### 💡 3DGS 关联题

**2.4** (参数化选择)：为什么 3DGS 不直接用 Cholesky 因子 $\mathbf{L}$（6 个自由元素），而是用旋转 R（4 元素，四元数）+ 缩放 S（3 元素 = 7 元素）？多出的 1 个自由度有什么用？
<details>
<summary>💡 提示</summary> R (四元数) + S 提供了更好的优化几何性质。旋转部分可以用指数映射回到 $SO(3)$ 流形，避免了 $\mathbf{L}$ 参数化中可能出现的"三角结构扭曲"导致的数值不稳定。多出的自由度让模型能更好地表达任意方向的主轴对齐。
</details>

---

# 📇 数值线性代数 — 一页纸总结卡片 (Cheat Sheet)

```markdown
# 🧱 Numerical Linear Algebra Cheat Sheet (3DGS Edition)
## Part 1: 矩阵分解 (Decomposition)
- **LU**: A = LU → 高效求解 Ax=b. (消元 + 回代).
- **QR**: A = QR → 最小二乘, 正则化优化. Q 正交, R 上三角.
- **SVD**: A = UΣV^T → **"旋转→缩放→再旋转"**. 万能拆解工具.

## Part 2: Cholesky — SPD 矩阵的专属
- **Cholesky**: A = LL^T. L 下三角, diag(L)>0. 计算量 O(n³/3).
- **与 SVD 关系**: L = V√Λ (特征值分解的三角化版本).
- **采样公式**: x ~ N(μ, Σ) → μ + LZ, z~N(0,I).

## Part 3: 条件数与数值稳定性
- **条件数 κ(A) = σ_max / σ_min**. 
    - κ ≈ 1: 完美稳定 (如单位阵).
    - κ >> 10¹⁵: 双精度下求逆会崩溃!
- **3DGS ε-Stabilizer**: B + εI → 人为提升最小奇异值, 降低 κ.

## Part 4: 3DGS Σ 参数化
- **Σ = RS S^T R^T** (R=旋转四元数, S=缩放对角阵).
- **等价于 Cholesky**: L ≈ RS. 保证始终正定!
- **几何语义**: R→主轴方向, S→主轴长度 → 直观可解释.

## 🏆 核心洞察
> "矩阵分解 = 把复杂变换拆成简单步骤."
> "Cholesky = SPD 的平方根密码. 3DGS 用它保证协方差始终合法."
```

---

> **Ch02 (Numerical LA) 完成！** 🔥  
> 
> Part 6 下一站：**Ch03 — SVD 与低秩近似** —— 深入理解为什么只保留几个奇异值就能压缩图像，以及 PCA 降维的矩阵本质。直接说 "继续"。
