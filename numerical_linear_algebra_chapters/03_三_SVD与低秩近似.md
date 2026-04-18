# Ch03 — SVD 与低秩近似：为什么"只保留几个数字"就能压缩图像？

> **本章目标**：掌握 Eckart-Young 定理（最优低秩逼近），理解 PCA 降维的矩阵本质，以及 SVD 在模型压缩中的核心作用。  
> **前置知识**：Ch01 (SVD, 条件数), Ch02 (Cholesky)。  
> **核心问题**：如果你要压缩一张高清图片，只保留原图 5% 的信息——你会删掉哪些像素？

---

## 🎯 问题驱动：信息量的"可压缩性"

### 场景 1：3DGS 的显存瓶颈

```python
# 一个高质量的 3DGS 模型可能需要数百万个 Gaussian Splat。
# 每个 Splat 有 μ(3), Σ(6), α(1), c(24, SH系数) → ~35 个浮点数.
# 总显存占用 ≈ 数百 MB.

# 问题：能不能用更少的参数表达同样的视觉信息？
# 答案: SVD + 低秩近似! 只保留"最重要的奇异方向"。
```

**关键问题**：什么样的矩阵可以用少数几个向量来"近似表示"？Singular Values（奇异值）的衰减速度说明了什么？

### 答案：**Eckart-Young 定理** —— "前 k 个奇异值包含了最优信息"

---

## 📐 Part 1: SVD 的几何直觉 — 旋转→缩放→再旋转

回顾 Ch01：
$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T + \sigma_2 \mathbf{u}_2\mathbf{v}_2^T + ... + \sigma_n \mathbf{u}_n\mathbf{v}_n^T$$

### Boxed Result：SVD 的"秩-1 分解"视角

每个项 $\sigma_i \mathbf{u}_i\mathbf{v}_i^T$ 是一个**秩为 1 的矩阵**。
- $\mathbf{u}_i$: 输出空间的主方向。
- $\mathbf{v}_i$: 输入空间的主方向。
- $\sigma_i$: 沿该方向的"拉伸强度"（奇异值）。

**几何直觉**：矩阵 A = "一系列主轴变换的叠加"！
$$\boxed{\mathbf{A} = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T, \quad r=\text{rank}(\mathbf{A})}$$

---

## 🔥 Part 2: Eckart-Young 定理 — "最优低秩逼近"的严格证明

### Boxed Result：核心定理 ⚔️

对于任意矩阵 $\mathbf{A}$，其 **rank-k 近似**（仅保留前 k 个奇异值）在所有可能的 rank-k 矩阵中，最小化 Frobenius 范数误差：
$$\boxed{\min_{\text{rank}(B)=k} \|\mathbf{A} - \mathbf{B}\|_F = \sqrt{\sigma_{k+1}^2 + ... + \sigma_r^2}}$$

最优解就是：**截断 SVD**！
$$\boxed{\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T}$$

### 💡 核心洞察：奇异值的"衰减速度"决定可压缩性！
- **快速衰减**（如 $\sigma_1=50, \sigma_2=3, \sigma_3=0.1$）→ 只需保留前 1~2 个项就能近似原矩阵！图像/视频非常适合 SVD 压缩。
- **缓慢衰减**（所有 $\sigma_i$ 接近）→ 无法有效压缩，信息均匀分布。

---

## 💻 Part 3: PyTorch 验证 — SVD 压缩与 PCA 降维

```python
import torch
import numpy as np
from PIL import Image

# ============================================================
# 1. Eckart-Young 定理验证：秩-k 逼近的误差分析
# ============================================================
print("=== Eckart-Young: 秩-k 逼近误差 ===")

# 构造一个低秩矩阵 (Rank=3) + 少量噪声
R = torch.randn(5, 3) * 10
C = torch.randn(3, 7) * 5
A_lowrank = R @ C # Rank ≤ 3
noise = torch.randn(5, 7) * 0.1
A_noisy = A_lowrank + noise

# SVD
U, s, Vt = torch.linalg.svd(A_noisy.float())
print(f"奇异值 σ: {s}")
print(f"前 3 个奇异值占比: {(s[:3].sum() / s.sum()).item():.4f} (应接近 1! → 低秩!)")

# --- Rank-1, 2, 3 逼近的误差 ---
for k in [1, 2, 3]:
    Ak = torch.matmul(torch.matmul(U[:, :k], torch.diag(s[:k])), Vt[:k, :])
    error = (A_noisy - Ak).norm().item()
    theoretical_error = s[k:].norm().item() # Eckart-Young 预言!
    print(f"Rank-{k}: 实际误差={error:.4f}, 理论误差(后 r-k 个 σ)={theoretical_error:.4f} ✅")

# ============================================================
# 2. PCA (主成分分析) = SVD 的投影
# ============================================================
print("\n=== PCA = SVD: 数据降维 ===")

# 模拟一批高斯 Splat 的位置数据 (N=100, d=3)
np.random.seed(42)
data = np.random.randn(100, 3) @ np.array([[5, 0.5], [0.5, 1], [0, 0]]) + 10 # 带相关性的数据

X = torch.tensor(data, dtype=torch.float32)
X_centered = X - X.mean(dim=0) # 去中心化 (PCA 第一步)

# PCA via SVD: U, s, Vt = svd(X_centered), 主成分在 Vt 的行中
U_pca, s_pca, Vt_pca = torch.linalg.svd(X_centered.float(), full_matrices=False)

print(f"数据协方差矩阵:\n{(X_centered.t() @ X_centered / 100).detach().numpy()}")
print(f"SVD 奇异值: {s_pca}")

# 主成分方向 (Vt 的行):
print(f"\n第一主成分方向 (PC1): {Vt_pca[0].detach().numpy()}")
print(f"第二主成分方向 (PC2): {Vt_pca[1].detach().numpy()}")

# 降维到 2D: 保留前两个主成分
X_2d = X_centered @ Vt_pca[:2, :T] # ←←← PCA 投影!
print(f"降维后数据形状: {X_2d.shape} (100×2)")

# 💡 PCA 的本质：找到方差最大的投影方向（即协方差矩阵的最大特征向量）！
```

---

## 🗺️ Part 5: 与 3DGS 的衔接点 — SVD 在压缩与加速中的角色

### Boxed Result：SVD 的三个核心应用场景

| 场景 | SVD 的作用 |
|------|-----------|
| **Splat 数量压缩 (Pruning)** | 如果某些 Splat 的协方差矩阵秩很低（极度扁平），它们的视觉贡献可被近似为低维子空间，可以安全剪枝。 |
| **神经网络权重压缩** | 3DGS 渲染管线中的 SH 系数矩阵（颜色随视角变化）是高度相关/低秩的。用 SVD 降秩后，只需存储少量基函数即可重建所有视角的颜色！这就是 **4D Gaussian Splatting** 中"共享 SH"的基础。 |
| **初始点云去噪** | 原始 SfM/MVS 生成的点云有大量噪声（协方差矩阵接近满秩但奇异值衰减快）。用 PCA 提取前几个主成分，可以过滤掉噪声方向。 |

### 💡 Boxed Result：为什么 SH (球谐函数) 可以用 SVD 压缩？

3DGS 中每个像素的颜色由 $L$ 阶球谐函数表示（通常 $L=3$, 9 个系数 × 3 通道 = 27 维）。
不同视角下的颜色变化高度相关 → **SH 矩阵的低秩性**允许我们用 SVD 将 27 维压缩到 ~5~8 维，显存占用降低 60%+！

---

## 🎓 本章小结

### 核心公式 (Boxed)

$$\boxed{\mathbf{A} = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T \quad (\text{SVD 的秩-1 分解})}$$

$$\boxed{\min_{\text{rank}(B)=k} \|\mathbf{A}-\mathbf{B}\|_F = \sqrt{\sum_{j=k+1}^{r} \sigma_j^2} \quad (\text{Eckart-Young 定理})}$$

### 关键洞察

> **SVD = 矩阵的"频谱分析"** —— 奇异值 $\sigma_i$ 就是频率。高频（小 σ）对应噪声/细节，低频（大 σ）对应主体结构。
> 
> **低秩近似不是"删数据"，而是"提取主干"** —— 保留前 k 个奇异方向，等价于用最少参数表达矩阵的核心信息。
> 
> **3DGS 的 SH 压缩 = SVD 在视角空间的应用** —— 颜色随视角的变化是高度相关的，SVD 能自动提取出几个"基视角函数"来近似所有角度！

---

## 📚 习题

### ✅ 基础题

**3.1** 证明：如果 $\mathbf{A}$ 的奇异值为 $[\sigma_1, \sigma_2]$，则 $\text{rank}(\mathbf{A}) = 2$。
<details>
<summary>💡 提示</summary> SVD 中非零奇异值的数量等于矩阵的秩（Rank）。因为 $\mathbf{u}_i\mathbf{v}_i^T$ 是线性无关的。
</details>

**3.2** PCA 的第一步为什么要"去中心化"（减去均值）？如果不去中心化会发生什么？
<details>
<summary>💡 提示</summary> 如果不减均值，第一个主成分会指向数据重心方向（直流分量），而不是方差最大的变化方向。PCA 关注的是"波动"而非"绝对位置"。
</details>

### 🔥 进阶题

**3.3** (Eckart-Young)：证明为什么 $\mathbf{A}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$ 是最优 rank-k 逼近。提示：利用酉不变性 ($\|\mathbf{UAV}\|_F = \|\mathbf{A}\|_F$)。
<details>
<summary>💡 提示</summary> Frobenius 范数在正交变换下不变。$\|\mathbf{A}-\mathbf{B}\|_F^2 = \sum(\sigma_i - b_{ii}')^2 + ...$，要最小化必须让前 k 个对角线项等于 σ_i（即保留最大的 k 个）。
</details>

### 💡 3DGS 关联题

**3.4** (模型压缩)：假设一个 3DGS 模型有 10,000 个 Splat，每个存储 27 维 SH 系数。如果用 SVD 将 SH 矩阵降秩到 k=5，显存占用从多少 GB 降到多少？
<details>
<summary>💡 提示</summary> 原始: 10,000 × 27 × 4 bytes ≈ 1.08 MB (不算其他参数). 压缩后: 10,000 × (5×3 + 5) × 4 ≈ 0.8 MB。实际增益来自"共享基函数"+"减少重复存储"。
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
- **3DGS Σ 参数化**: RS S^T R^T ≈ Cholesky 几何变体.

## Part 3: SVD 与低秩近似 ⚔️ Eckart-Young 定理
- **A = ∑ σ_i u_i v_i^T** (秩-1 分解).
- **最优 rank-k**: A_k = U_k Σ_k V_k^T.
- **误差下界**: ||A-A_k||_F² = ∑_{j>k} σ_j².

## Part 4: PCA = SVD 的投影
- **主成分** = 协方差矩阵的特征向量.
- **降维**: X₂d = X_centered · Vt[:k].
- **压缩核心**: σ₁ ≫ σ₂,σ₃... → 前几个方向占绝大多数信息!

## Part 5: 3DGS SVD 实战应用
| 场景 | SVD 作用 |
|------|---------|
| SH 系数压缩 | 视角空间低秩 → 27维降至 ~5~8 维 |
| Splat Pruning | 扁平协方差 = 低秩贡献 → 安全剪枝 |
| 点云去噪 | PCA 过滤噪声方向 (小奇异值) |

## 🏆 核心洞察
> "SVD 是矩阵的'频谱分析'. 大奇异值=主体结构, 小奇异值=噪声."
> "Eckart-Young: 截断 SVD = 数学上最优的信息提取!"
```

---

> **Ch03 (Numerical LA) 完成！** 🔥  
> 
> Part 6 下一站：**Ch04 — PCA 降维的严格推导** —— 从"方差最大化"出发，证明为什么 PCA 就是 SVD。直接说 "继续"。
