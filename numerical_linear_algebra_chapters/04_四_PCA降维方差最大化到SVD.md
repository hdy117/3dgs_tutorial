# Ch04 — PCA 降维：从"方差最大化"到 SVD 的严格推导

> **本章目标**：从零开始推导 PCA（主成分分析），证明它等价于 SVD，并理解它在 3DGS 初始化和训练中的作用。  
> **前置知识**：Ch01 (SVD), Ch03 (低秩近似, Eckart-Young)。  
> **核心问题**：如果你有一堆高斯点云的数据，如何找到最能代表这些数据的"主方向"？

---

## 🎯 问题驱动：从数据中提取"主干结构"

### 场景 1：3DGS 的初始化与去噪

```python
# SfM/MVS 生成的初始点云通常包含大量噪声。
# 问题：如何自动检测并移除这些异常 Splat？
# 答案: PCA! 用前几个主成分捕捉"真实结构"，丢弃小奇异值对应的"噪声方向"。
```

**关键问题**：什么是"最能代表数据的方向"？如何从数学上严格定义"主成分"？

---

## 📐 Part 1: PCA 的第一性原理 — "方差最大化"

### Boxed Result：PCA 的定义

> **第一主成分 (PC1)** = 单位向量 $\mathbf{w}_1$，使得数据在其上的投影**方差最大**。
> 
> **第二主成分 (PC2)** = 与 PC1 正交的单位向量 $\mathbf{w}_2$，在正交约束下使投影方差最大。

### 💡 Boxed Result：优化问题推导 ⚔️

设数据矩阵为 $\mathbf{X}$（已去中心化），第 $i$ 列是样本 $\mathbf{x}_i \in \mathbb{R}^d$。
在方向 $\mathbf{w}$（$\|\mathbf{w}\|_2=1$）上的投影方差：
$$\text{Var}(\mathbf{X}\mathbf{w}) = \frac{1}{n}\mathbf{w}^T \underbrace{\left(\frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i \mathbf{x}_i^T\right)}_{\text{协方差矩阵 } \mathbf{\Sigma}} \mathbf{w} = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$

**PCA 的优化问题：**
$$\boxed{\max_{\|\mathbf{w}\|_2=1} \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$$

---

## 🔥 Part 2: PCA = 协方差矩阵的特征值分解

### Boxed Result：拉格朗日乘子法推导 ⚔️

目标函数加约束：
$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} - \lambda (\mathbf{w}^T \mathbf{w} - 1)$$

对 $\mathbf{w}$ 求导：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{\Sigma}\mathbf{w} - 2\lambda \mathbf{w} = \boxed{0 \implies \mathbf{\Sigma}\mathbf{w} = \lambda \mathbf{w}}$$

**结论**：最优方向 $\mathbf{w}$ 就是协方差矩阵 $\mathbf{\Sigma}$ 的**特征向量！**
最大方差 = 对应的特征值 $\lambda_{\max}$！

### 💡 Boxed Result：PCA 的严格步骤（从算法到 SVD）

1. **去中心化**: $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{x}}$.
2. **计算协方差**: $\mathbf{\Sigma} = \frac{1}{n}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$ (或直接用 SVD).
3. **特征分解**: $\mathbf{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$.
4. **取前 k 个特征向量** → 投影矩阵: $\mathbf{W}_k = [\mathbf{v}_1, ..., \mathbf{v}_k]$.
5. **降维**: $\tilde{\mathbf{X}}_{\text{PCA}, k} = \tilde{\mathbf{X}}\mathbf{W}_k$ (d→k).

**与 SVD 的关系：**
$$\boxed{\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T = \text{SVD}(\tilde{\mathbf{X}) \implies \mathbf{\Sigma}_{cov} = \frac{1}{n}\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^T}$$

所以：**PCA 的特征向量 = SVD 的右奇异向量 V；特征值 = σ²/n！**

---

## 💻 Part 3: PyTorch 验证 — PCA 与 SVD 等价性

```python
import torch
import numpy as np

# ============================================================
# 1. PCA 的实现 (两种方法)：协方差特征分解 vs SVD
# ============================================================
print("=== PCA: 特征分解法 vs SVD法 ===")

np.random.seed(42)
# 构造带相关性的数据（3维 → 想降到2D）
data = np.random.randn(500, 3) @ np.array([[3.0, 1.0], [1.0, 2.0], [0.5, 0.5]])

X = torch.tensor(data, dtype=torch.float64)
X_centered = X - X.mean(dim=0)

# --- 方法 A: 协方差矩阵特征分解 (传统 PCA) ---
cov_matrix = (X_centered.t() @ X_centered) / (X_centered.shape[0] - 1) # 样本协方差
eig_vals, eig_vecs = torch.linalg.eigh(cov_matrix.float()) # eigh: SPD专用
# eigh 按升序排列，取倒序
eig_vals_sorted = eig_vals.flip(dims=[0])
eig_vecs_sorted = eig_vecs[:, :T] # 注意翻转列

print(f"协方差矩阵特征值 (大→小): {eig_vals_sorted}")

# --- 方法 B: SVD (高效，无需显式计算协方差) ---
U_svd, s, Vt_svd = torch.linalg.svd(X_centered.double(), full_matrices=False)
s_pca = s / np.sqrt(X_centered.shape[0] - 1) # PCA 特征值 = σ²/(n-1)
print(f"SVD 奇异值 (大→小): {s}")

# --- 验证等价性: 特征值 ≈ σ²/(n-1) ---
for i in range(3):
    print(f"λ{i+1}={eig_vals_sorted[i].item():.4f}, σ²/(n-1)={s_pca[i]**2:.4f} ✅")

# --- 降维到 2D ---
k = 2
X_2d_from_eigen = X_centered @ eig_vecs_sorted[:, :k]
X_2d_from_SVD = X_centered @ Vt_svd[:k, :] # ←←← SVD PCA!

print(f"降维形状: {X_2d_from_SVD.shape} (500×2)")

# ============================================================
# 2. 重构误差 vs Rank-k
# ============================================================
print("\n=== 重构误差: k=1, 2, 3 ===")

for k in [1, 2, 3]:
    # PCA 重构（从降维恢复）
    X_reconstructed = (X_2d_from_SVD if k==2 else torch.zeros_like(X_centered)) 
    if k == 2:
        X_rec = X_2d_from_SVD @ Vt_svd[:k, :] + X.mean(dim=0) # 加回均值
        error = ((X - X_rec)**2).sum().item() / X.numel()
    else:
        # 用 SVD rank-k 直接重构
        U_k = U_svd[:, :k]
        s_k = s[:k]
        Vt_k = Vt_svd[:k, :]
        X_rec_full = torch.matmul(torch.matmul(U_k, torch.diag(s_k)), Vt_k) + X.mean(dim=0)
        error = ((X - X_rec_full)**2).sum().item() / X.numel()
    
    theoretical_error = s[k:].pow(2).sum().item() / (X_centered.shape[0] * X.shape[1]) # 理论下界
    print(f"Rank-{k}: MSE={error:.6f}, 理论下界={theoretical_error:.6f} ✅")

# ============================================================
# 3. PCA 在 3DGS 初始化中的应用：自动检测"扁平化"Splat
# ============================================================
print("\n=== PCA + Splat 扁平度分析 ===")

# 模拟一组协方差矩阵 (10 个 Splat, 每个 Σ 是 3×3 SPD)
np.random.seed(123)
splat_sigs = []
for _ in range(10):
    # 随机生成 SPD: R·D·R^T，其中 D 的对角线可能极度不均匀 → "扁平化" Splat!
    theta = np.random.uniform(0, np.pi/2)
    R_3d = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    # 扁平化 Splat: 一个方向尺度极小 (σ²→0)
    D = torch.diag(torch.tensor([5.0**2, 3.0**2, 0.01**2])) # ←←← 扁平!
    Sigma_i = R_3d @ D @ R_3d.t()
    splat_sigs.append(Sigma_i)

for i, sig in enumerate(splat_sigs):
    _, s_i, _ = torch.linalg.svd(sig.float())
    flatness_ratio = (s_i[2] / s_i[0]).item() # 最小/最大奇异值 → "扁平度"
    print(f"Splat {i+1}: σ={s_i}, 扁平度 σ_min/σ_max={flatness_ratio:.6f}")
    
    if flatness_ratio < 0.1:
        print(f"  ⚠️ 高度扁平 Splat! 可以用 PCA 降维到 2D 子空间近似.")

# 💡 这就是为什么某些视角下 Splat 会"爆炸"成薄片 —— 
# 它们的协方差矩阵极度病态（条件数极大），投影后变成无限拉伸的椭圆！
```

---

## 🗺️ Part 5: 与 3DGS 的衔接点 — PCA 在训练管线中的角色

### Boxed Result：PCA 在 3DGS 中的三个核心应用

| 场景 | PCA/SVD 的作用 |
|------|---------------|
| **Splat 初始化去噪** | SfM 点云坐标有误差。用 PCA 找到数据的主平面（PC1, PC2），将初始协方差投影到这个平面上，减少噪声方向的影响。 |
| **视角相关外观压缩** | SH 系数随视角变化高度相关 → PCA 提取"主颜色模式"，用前几个主成分近似所有角度。这就是 **4D Gaussian Splatting (4DGS)** 的核心思想！ |
| **Splat Pruning（剪枝）** | 如果某个 Splat 的协方差矩阵条件数 $\kappa \gg 10^{12}$（极度扁平），它在渲染中的贡献可以被降维到二维子空间近似，甚至可以安全移除。 |

### 💡 Boxed Result：为什么"扁平化 Splat"会导致渲染噪声？

当 $\mathbf{B} = \mathbf{J}\boldsymbol{\Sigma}\mathbf{J}^T$ 的最小奇异值接近 0 时（Splat 极度扁平）：
- 投影 Jacobian $\mathbf{J}$ 的奇异值进一步被拉低 → **$\kappa(\mathbf{B})$ 爆炸**。
- $\mathbf{B}^{-1}$ 中的数值误差导致渲染出现**极端高频噪声（"雪花点")**。

这就是 3DGS **"近裁剪/远裁剪"** 和 **"高斯球半径限制"** 的数学根源！PCA/SVD 可以帮助检测并修复这种病态情况。

---

## 🎓 本章小结

### 核心公式 (Boxed)

$$\boxed{\max_{\|\mathbf{w}\|=1} \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} = \lambda_{\max}, \quad \text{解: } \mathbf{w} = \mathbf{v}_{\max} (\text{最大特征向量})}$$

$$\boxed{\text{PCA 特征值} = \frac{\sigma^2}{n-1}, \quad \text{特征向量} = \text{SVD右奇异向量}}$$

### 关键洞察

> **PCA 的第一性原理是"方差最大化"** —— 找到能让数据投影后"最分散"的方向，因为这个方向包含了最多的信息。
> 
> **PCA ≠ SVD，但 PCA 的解就是 SVD** —— 协方差矩阵的特征向量 = SVD 的右奇异向量 V；特征值 = σ²/n。所以实际实现中直接用 SVD 更高效（避免显式计算 $\boldsymbol{\Sigma}$）。
> 
> **3DGS 的"扁平化 Splat"就是 PCA 的低秩结构** —— 极度扁平意味着协方差矩阵只有 1~2 个大奇异值，其余接近 0。PCA/SVD 可以自动检测这种情况并做降维处理！

---

## 📚 习题

### ✅ 基础题

**4.1** 证明：如果 $\mathbf{\Sigma}$ 是协方差矩阵，则它一定是 SPD（正半定）。
<details>
<summary>💡 提示</summary> 对任意非零向量 v, $v^T \boldsymbol{\Sigma} v = \frac{1}{n}\sum_i (v^Tx_i)^2 \geq 0$。因为平方项，所以 ≥ 0 → 正半定！
</details>

**4.2** PCA 降维后，如何用主成分恢复原始数据？误差是多少？
<details>
<summary>💡 提示</summary> $\tilde{\mathbf{X}}_{\text{reconstructed}} = \tilde{\mathbf{X}}_k \mathbf{V}_k^T$。误差的平方和 = $\sum_{j>k} \lambda_j$ (Eckart-Young 定理，用特征值表示)。
</details>

### 🔥 进阶题

**4.3** 为什么 PCA 通常不用协方差矩阵的特征分解，而是直接用 SVD？从计算复杂度分析。
<details>
<summary>💡 提示</summary> 计算 $\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$ 需要 $O(n \cdot d^2)$，再特征分解需要 $O(d^3)$。而 SVD(X) 直接给出 V 和 σ，无需中间步骤。数值上更稳定（避免 $\boldsymbol{\Sigma}$ 的条件数平方化）。
</details>

### 💡 3DGS 关联题

**4.4** (Splat 扁平度检测)：如果一个 Splat 的协方差矩阵奇异值为 $[\sigma_1, \sigma_2] = [5, 0.01]$，它的条件数是多少？你会怎么处理这个 Splat 以避免渲染噪声？
<details>
<summary>💡 提示</summary> $\kappa = 5/0.01 = 500$。虽然不算极度病态（不像 $10^{15}$），但在某些视角下投影后条件数会进一步放大。可以通过增大 ε-Stabilizer、设置最小奇异值下限（如 $\sigma_{\min} \geq 0.1$）、或者在训练时对该 Splat 应用更小的学习率来缓解。
</details>

---

# 📇 数值线性代数 — 一页纸总结卡片 (Cheat Sheet)

```markdown
# 🧱 Numerical Linear Algebra Cheat Sheet (3DGS Edition)
## Part 1: 矩阵分解
- **LU**: A = LU → Ax=b 求解.
- **QR**: A = QR → 最小二乘, 优化正则化.
- **SVD**: A = UΣV^T → **"旋转→缩放→再旋转"**.

## Part 2: Cholesky (SPD 专属)
- **A = LL^T** → 协方差平方根. 计算量 O(n³/3).
- **采样**: x ~ N(μ,Σ) → μ + LZ, z~N(0,I).

## Part 3: SVD 与低秩近似 ⚔️ Eckart-Young 定理
- **A = ∑ σ_i u_i v_i^T** (秩-1 分解).
- **最优 rank-k**: A_k = U_k Σ_k V_k^T.

## Part 4: PCA — 方差最大化 → SVD ⚔️
- **PCA 定义**: max w^TΣw s.t. ||w||=1.
- **解**: w = v_max (最大特征向量), λ_max = 方差.
- **与 SVD**: λ_i = σ²/(n-1), v_i = V[:,i].

## Part 5: 3DGS 应用
| 场景 | 数学工具 |
|------|---------|
| Splat 去噪 | PCA 主平面投影 |
| SH 压缩 | 视角空间低秩 (PCA/SVD) |
| Splat Pruning | κ(Σ) 检测扁平化 → 安全移除 |

## 🏆 核心洞察
> "PCA = 找方差最大的方向. 解是协方差的特征向量 = SVD的右奇异向量."
> "SVD 把矩阵拆成'主干 + 噪声'. 保留前 k 个 σ 就是最优近似!"
```

---

> **Ch04 (Numerical LA) 完成！** 🔥  
> 
> Part 6 下一站：**Ch05 — SVD 与 Rank Defect（秩亏矩阵）** —— 深入理解什么时候矩阵"无法求逆"，以及这在神经网络训练中的表现。直接说 "继续"。
