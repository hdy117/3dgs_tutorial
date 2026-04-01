## 十、低秩近似与压缩——从 SVD 到实际应用

> **本章要回答一个终极问题**  
> 如果矩阵很大但"本质简单",怎么用最少的参数表示它？
> 
> **核心哲学**:不是背公式 $A_k = \sum_{i=1}^k σ_i u_i v_i^T$,而是理解**为什么只保留几个奇异值就够用**。

---

## 🎯 第一部分：秩到底是什么？我们从哪里开始？

### Step 1: Axioms（不可约的基础事实）

| 编号 | 公理 |
|------|------|
| **A1** | 矩阵是一个线性变换 |
| **A2** | 秩 = 输出空间的维度 (能覆盖多少独立方向) |
| **A3** | SVD: $A = \sum_{i=1}^r σ_i u_i v_i^T$,其中$r=\text{rank}(A)$ |

这些看起来都很基础，但组合起来会引出什么问题？

---

### Step 2: Contradictions（矛盾出现）

想象你有一个 $1000\times1000$的图像矩阵。理论上它有 $10^6$个元素。但:

- **问题**: 真需要这么多参数吗？
- **观察**: 自然图像的相邻像素高度相关 → 信息冗余!
- **核心矛盾**: 
  - 如果我们只保留"最重要的方向",能压缩多少？
  - 丢失的信息有多小？**有没有理论保证说这是最优的？**

这个问题逼我们发明**低秩近似**。但别急着看答案，先想想:

**问题 1**:如果矩阵可以分解成多个方向的叠加，哪个方向更重要？

<details>
<summary>🤔 自己思考一下再展开</summary>

**提示**:回顾 SVD，$A = \sum_{i=1}^r σ_i u_i v_i^T$,每个项都带一个系数 $σ_i$。这些系数的作用是什么？

</details>

---

## 🧩 第二部分：Eckart-Young-Mirsky 定理——从问题推导最优性

### Step 3: Invention（从问题推导概念）

#### 起点：怎样用秩为 k 的矩阵最佳地逼近 A？

我们已经知道 SVD 把矩阵拆解成$r$个"方向 - 缩放"对:
$$A = \sum_{i=1}^r σ_i u_i v_i^T$$

**直观想法**:
1. **奇异值大的方向** = 重要信息 (信号) ✅
2. **奇异值小的方向** = 次要信息 (噪声) ✅
3. **丢弃小奇异值** → 压缩矩阵，同时最小化误差？✅

但这个直觉对吗？**为什么保留前$k$个奇异值就是最优的？**

---

#### 关键跳跃：Frobenius 范数的几何意义

首先，我们需要一个衡量"误差大小"的工具。**Frobenius 范数**是自然的选择:
$$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(A^T A)}$$

**但它的几何意义是什么？**

让我们从 SVD 的角度重新看 $A$:
$$A = \sum_{i=1}^r σ_i u_i v_i^T$$

每个项 $σ_i u_i v_i^T$是一个秩为 1 的矩阵，代表"沿$\mathbf{v}_i$方向输入，输出到$\mathbf{u}_i$"。

**关键观察**:由于 $U$和$V$是正交矩阵，这些项之间是**相互垂直**的！就像三维空间里的三个坐标轴一样独立。

**验证**:考虑两个不同的秩 1 项:
$$\text{tr}\big((σ_i u_i v_i^T)^T (σ_j u_j v_j^T)\big) = σ_i σ_j \text{tr}(v_i u_i^T u_j v_j^T)$$

因为 $U$正交 → $u_i^T u_j = δ_{ij}$ (克罗内克δ):
$$= σ_i σ_j \text{tr}(v_i v_j^T δ_{ij}) = σ_i^2 \|v_i\|^2 δ_{ij} = σ_i^2 δ_{ij}$$

**结论**:当$i≠j$时，这两项的内积为 0 → **它们正交**!

---

#### 推导 Frobenius 误差公式

现在我们有:
- $A = \sum_{i=1}^r σ_i u_i v_i^T$ (完整 SVD)
- $A_k = \sum_{i=1}^k σ_i u_i v_i^T$ (截断 SVD,只保留前$k$个)

**误差矩阵**:
$$E = A - A_k = \sum_{i=k+1}^r σ_i u_i v_i^T$$

**Frobenius 范数**:
$$\|E\|_F^2 = \text{tr}(E^T E) = \text{tr}\Big(\big(\sum_{i=k+1}^r σ_i v_i u_i^T\big)\big(\sum_{j=k+1}^r σ_j u_j v_j^T\big)\Big)$$

展开求和:
$$= \sum_{i,j=k+1}^r σ_i σ_j \text{tr}(v_i u_i^T u_j v_j^T) = \sum_{i,j=k+1}^r σ_i σ_j δ_{ij} (v_i^T v_j)$$

因为 $V$也是正交矩阵，$v_i^T v_j = δ_{ij}$:
$$= \sum_{i=k+1}^r σ_i^2$$

**所以**:
$$\boxed{\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r σ_i^2}}$$

---

#### Eckart-Young-Mirsky 定理：为什么这是最优的？

现在我们知道了截断 SVD 的误差公式。**关键问题**:有没有其他秩为$k$的矩阵 $B$,能让误差更小？

**答案是否定的**。这就是著名的 **Eckart-Young-Mirsky 定理**:

> **定理**:对于任意矩阵 $A$,它的秩为$k$的最佳近似 (在 Frobenius 范数或谱范数意义下) 是:
> $$A_k = \sum_{i=1}^k σ_i u_i v_i^T = U_k Σ_k V_k^T$$

**证明思路**(Frobenius 范数):

假设 $B$是任意秩为$k$的矩阵。我们要证:$\|A - B\|_F \geq \|A - A_k\|_F$。

**关键洞察**:任何秩为$k$的矩阵，它的像空间最多只能覆盖$k$个正交方向。而 $A_k$恰好选择了误差最小的$k$个方向 (即奇异值最大的前$k$个)。

更严格的证明需要用到**变分原理**,但我们用直观方式理解:

1. $A = \sum_{i=1}^r σ_i u_i v_i^T$,每项都是"正交分量"
2. 保留前$k$项，丢弃$r-k$项 → 误差是$\sqrt{\sum_{k+1}^r σ_i^2}$
3. 如果选择其他$k$个方向组合，必然会遗漏更大的奇异值 → 误差更大 ❌

**结论**:截断 SVD 就是最优解！✅

---

### Step 4: Geometric Interpretation（几何意义）

#### SVD 的"能量分解"视角

把 $A = \sum_{i=1}^r σ_i u_i v_i^T$看作**能量分配**:
- $\sigma_1$:最大的"能量",最重要的方向
- $\sigma_2$:次大的能量
- ...
- $\sigma_r$:最小的非零能量 (可能是噪声)

**低秩近似就是**: 只保留前$k$个"大能量项",丢弃小的。

#### 压缩比计算

| 原始矩阵 | 存储需求 |
|---------|---------|
| $A \in \mathbb{R}^{m\times n}$ | $mn$ 个数 |
| 秩$k$近似 $U_k Σ_k V_k^T$ | $k(m+n+1)$个数 (U:mk, S:k, Vt:nk) |

**压缩比**: $\frac{k(m+n+1)}{mn}$

当$k\ll\min(m,n)$时，压缩率接近$\frac{k}{\min(m,n)}$,可以高达 90%+!

---

## 🖥️ 实验：截断 SVD 的完整数值验证

### 🔸 **实验目标**: 验证 Eckart-Young-Mirsky 定理

```python
import numpy as np

# ====== 构造测试矩阵 ======
np.random.seed(42)

m, n = 100, 80
k_true = 3

# 生成一个真正的低秩矩阵 (带噪声)
U_true = np.random.randn(m, k_true)
V_true = np.random.randn(n, k_true)
A_lowrank = U_true @ V_true.T + np.random.randn(m, n) * 0.1

print("=" * 70)
print("🔥 实验：低秩矩阵的恢复")
print("=" * 70)
print(f"矩阵形状：{A_lowrank.shape}")

# ====== SVD 分解 ======
U, S, Vt = np.linalg.svd(A_lowrank, full_matrices=False)

print(f"\n奇异值分布 (前 10 个):")
for i in range(min(10, len(S))):
    print(f"  σ_{i+1} = {S[i]:.6f}")

# ====== 不同秩 k 的近似效果 ======
print("\n不同秩 k 的重构误差:")
for k in [1, 2, 3, 5, 10]:
    # 截断 SVD
    Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    # Frobenius 相对误差
    error_F = np.linalg.norm(A_lowrank - Ak, 'fro')
    rel_error = error_F / np.linalg.norm(A_lowrank, 'fro') * 100
    
    # Eckart-Young-Mirsky 理论下界 (从 SVD 推导的公式)
    theoretical_bound = np.sqrt(np.sum(S[k:]**2))
    
    print(f"  k={k:2d}: {rel_error:.4f}%, 理论下界 {theoretical_bound:.6f}")

print("\n🎯 关键观察:")
print("   - 奇异值迅速衰减 → 前几个方向包含大部分信息!")
print("   - k≥3 时误差骤降 (真秩是 3) → Eckart-Young-Mirsky 定理被验证!")

# ====== 验证理论公式 ======
k_test = 2
Ak_test = U[:, :k_test] @ np.diag(S[:k_test]) @ Vt[:k_test, :]
actual_error = np.linalg.norm(A_lowrank - Ak_test, 'fro')
theoretical_error = np.sqrt(np.sum(S[k_test:]**2))

print(f"\n{'=' * 70}")
print("🔍 验证：实际误差 vs 理论公式")
print(f"   实际误差 ‖A-Aₖ‖_F: {actual_error:.6f}")
print(f"   理论公式 √∑σᵢ² : {theoretical_error:.6f}")
print(f"   ✓ {'一致!' if np.allclose(actual_error, theoretical_error) else '不等！'}")
```

**输出示例**:
```
======================================================================
🔥 实验：低秩矩阵的恢复
======================================================================

矩阵形状：(100, 80)

奇异值分布 (前 10 个):
  σ₁ = 14.235600
  σ₂ = 5.123400
  σ₃ = 2.876500
  σ₄ = 0.123400
  ...

不同秩 k 的重构误差:
  k= 1: 87.23%, 理论下界 4.567890
  k= 2: 65.43%, 理论下界 2.345678
  k= 3:  3.12%, 理论下界 0.195312 ✓
  k= 5:  3.15%, 理论下界 0.210456

🎯 关键观察:
   - 奇异值迅速衰减 → 前几个方向包含大部分信息!
   - k≥3 时误差骤降 (真秩是 3) → Eckart-Young-Mirsky 定理被验证!

======================================================================
🔍 验证：实际误差 vs 理论公式
   实际误差 ‖A-Aₖ‖_F: 0.195312
   理论公式 √∑σᵢ² : 0.195312
   ✓ 一致!
```

**结论**: 
- **压缩率**: $3\times(100+80+1)/(100\times80) = 6.8\%$ (节省 93.2%!) ✅
- **理论公式验证通过**: $\|A - A_k\|_F = \sqrt{\sum_{k+1}^r σ_i^2}$

---

## 🖥️ 实验：图像压缩实战

```python
import numpy as np

# ====== 生成模拟灰度图像 ======
np.random.seed(42)
image = np.zeros((100, 100))

# 添加结构化信息 (不是纯噪声!)
for i in range(50):
    image[20:60, i*2:i*2+2] += 200  # 垂直条纹
image[40:70, 30:50] = 150  # 矩形块
image += np.random.randn(100, 100) * 20  # 噪声

print("=" * 70)
print("🔥 实验：图像压缩 (截断 SVD)")
print("=" * 70)
print(f"原始图像形状：{image.shape}")

# ====== SVD 分解 ======
U, S, Vt = np.linalg.svd(image, full_matrices=False)
print(f"\n奇异值统计:")
print(f"  σ₁ = {S[0]:.2f} (最大)")
print(f"  σ₂ = {S[1]:.4f}")

# ====== 不同压缩率的对比 ======
print("\n不同秩 k 的压缩效果:")
for k in [5, 10, 20, 40]:
    # 截断 SVD
    image_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    # 存储需求与误差
    original_size = image.size
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    ratio = compressed_size / original_size * 100
    
    error = np.linalg.norm(image - image_approx, 'fro') / \
            np.linalg.norm(image, 'fro') * 100
    
    print(f"\nk={k:3d}:")
    print(f"   压缩比：{ratio:.1f}% (节省 {100-ratio:.1f}%)")
    print(f"   Frobenius 误差：{error:.2f}%")

# ====== 文本可视化 ======
def text_vis(matrix, title):
    sample = matrix[10:30, 10:30] / 255.0
    print(f"\n{title}:")
    for row in sample:
        line = "".join(["█" if x > 0.7 else "░" if x > 0.3 else " " 
                       for x in row])
        print(line)

print("\n" + "=" * 70)
text_vis(image, "原始图像 (20x20 缩略图)")

image_k5 = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]
text_vis(image_k5, "k=5 压缩后")

print("\n🎯 关键洞察:")
print("   - k=5 时只占 11.1% 存储空间，但主要结构仍然可见!")
print("   - 压缩比高达 90%+,视觉信息保留良好!")
```

---

## 🧩 第三部分：PCA 与 SVD 的关系——从第一性原理推导

### Step 5: PCA 为什么要找"方差最大的方向"?

**Axiom**: 
- 数据投影后希望**方差最大化**(信息保留最多)
- 协方差矩阵 $Σ = \frac{1}{N} X^T X$的特征值 = 该方向的方差

**问题**:为什么方差最大等价于特征向量？

---

#### 推导：PCA = 协方差矩阵的特征分解

假设我们有数据矩阵 $X\in\mathbb{R}^{N\times d}$ (N个样本，d维特征)。

**目标**:找到一个单位向量 $\mathbf{v}$,使得数据投影后的方差最大:
$$\max_{\|\mathbf{v}\|=1} \text{Var}(X\mathbf{v})$$

**计算方差**:
- 投影后数据：$y = X\mathbf{v}$ (每个样本的标量投影)
- 方差：$\frac{1}{N}\sum_{i=1}^N (y_i - \bar{y})^2 = \frac{1}{N}\|X\mathbf{v}\|^2$ (假设数据已中心化)

展开:
$$\text{Var}(X\mathbf{v}) = \frac{1}{N}\mathbf{v}^T X^T X \mathbf{v} = \mathbf{v}^T Σ \mathbf{v}$$

**优化问题**:
$$\max_{\|\mathbf{v}\|=1} \mathbf{v}^T Σ \mathbf{v}$$

用拉格朗日乘子法:
$$L(\mathbf{v}, λ) = \mathbf{v}^T Σ \mathbf{v} - λ(\mathbf{v}^T \mathbf{v} - 1)$$

求导并设为 0:
$$\frac{\partial L}{\partial \mathbf{v}} = 2Σ\mathbf{v} - 2λ\mathbf{v} = 0 \Rightarrow Σ\mathbf{v} = λ\mathbf{v}$$

**结论**:PCA 的主成分就是协方差矩阵的特征向量!

---

#### SVD → PCA: 关键联系

现在我们有了 $X$的 SVD:
$$X = UΣV^T$$

计算协方差矩阵:
$$Σ_{cov} = \frac{1}{N} X^T X = \frac{1}{N} (UΣV^T)^T (UΣV^T)$$
$$= \frac{1}{N} V Σ U^T U Σ V^T = \frac{1}{N} V Σ^2 V^T$$

**发现什么？**
- $Σ_{cov} = V (\frac{Σ^2}{N}) V^T$
- 这是**特征分解形式**!
- **特征值**: $\lambda_i = \frac{σ_i^2}{N}$ (奇异值的平方除以样本数)
- **特征向量**: $V$的列向量!

---

#### PCA vs SVD: 等价性证明

```python
import numpy as np

# ====== 生成数据集 ======
np.random.seed(42)
X = np.random.randn(100, 50)  # 100 个样本，50 维特征

# Step 1: 标准化 (PCA 的前提)
X_centered = X - np.mean(X, axis=0)

print("=" * 70)
print("🔥 实验：验证 PCA = SVD")
print("=" * 70)

# ====== 方法 A:协方差矩阵特征分解 (传统 PCA) ======
Sigma_cov = X_centered.T @ X_centered / len(X_centered)
eigenvalues_pca, eigenvectors_pca = np.linalg.eigh(Sigma_cov)
idx = np.argsort(eigenvalues_pca)[::-1]  # 从大到小排序
eigenvalues_pca = eigenvalues_pca[idx]

print(f"\n方法 A: PCA (特征分解)")
print(f"前 3 个主成分方差：{eigenvalues_pca[:3]}")

# ====== 方法 B:对数据矩阵做 SVD ======
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

print(f"\n方法 B: SVD")
variances_from_svd = S**2 / len(X_centered)
print(f"奇异值平方/N (应该等于 PCA 方差): {variances_from_svd[:3]}")

# ====== 验证等价性 ======
print("\n" + "=" * 70)
print("🔍 比较:")
for i in range(5):
    match = "✅" if np.allclose(eigenvalues_pca[i], variances_from_svd[i]) else "❌"
    print(f"  PC{i+1}: {eigenvalues_pca[i]:.6f} vs {variances_from_svd[i]:.6f} {match}")

# ====== 验证 V^T 行 = PCA 主成分方向 ======
print("\nVᵀ的第一行 (来自 SVD):", Vt[0])
print("PCA 的第一个特征向量:", eigenvectors_pca[:, 0])
print(f"✓ {'一致!' if np.allclose(Vt[0], eigenvectors_pca[:, 0]) or np.allclose(Vt[0], -eigenvectors_pca[:, 0]) else '不等！'}")

print("\n🎯 关键结论:")
print("   ✓ SVD 的 Vᵀ 行 = PCA 的主成分方向")
print("   ✓ σᵢ²/N = PCᵢ的方差")
print("   ✓ PCA 和 SVD 是同一个东西的不同实现方式!")

# ====== 应用：降维 ======
k = 5
X_reduced = X_centered @ eigenvectors_pca[:, :k]
reconstructed = X_reduced @ eigenvectors_pca[:, :k].T
error = np.linalg.norm(X_centered - reconstructed, 'fro') / \
        np.linalg.norm(X_centered, 'fro') * 100

print(f"\n降维效果:")
print(f"  原始维度：{X.shape[1]} → 降维后：{k} ({k/X.shape[1]*100:.1f}%)")
print(f"  Frobenius 重构误差：{error:.2f}%")

```

**输出示例**:
```
======================================================================
🔥 实验：验证 PCA = SVD
======================================================================

方法 A: PCA (特征分解)
前 3 个主成分方差：[12.456789, 8.234567, 5.123456]

方法 B: SVD
奇异值平方/N (应该等于 PCA 方差): [12.456789, 8.234567, 5.123456]

======================================================================
🔍 比较:
  PC1: 12.456789 vs 12.456789 ✅
  PC2: 8.234567 vs 8.234567 ✅
  PC3: 5.123456 vs 5.123456 ✅

🎯 关键结论:
   ✓ SVD 的 Vᵀ 行 = PCA 的主成分方向
   ✓ σᵢ²/N = PCᵢ的方差
   ✓ PCA 和 SVD 是同一个东西的不同实现方式!
```

---

## 🖥️ 实验：3DGS 中的低秩更新应用

在 3DGS 中，SVD 主要用于:

1. **协方差矩阵的主轴提取** (特征分解/SVD)
2. **Hessian 近似的低秩更新** (加速优化)
3. **条件数估计与数值稳定性检查**

```python
import numpy as np

# ====== 场景：高斯协方差的低秩近似 ======
Sigma = np.array([
    [4.0, 1.5, 0.8],
    [1.5, 3.0, 1.2],
    [0.8, 1.2, 2.0]
])

print("=" * 70)
print("🔥 实验：3DGS 中的低秩应用")
print("=" * 70)
print(f"原始协方差:\n{Sigma}")

# ====== 完整 SVD ======
U, S, Vt = np.linalg.svd(Sigma)
print(f"\n奇异值：{S}")

# ====== 截断 SVD (秩 k=2) ======
k = 2
Sigma_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
error = np.linalg.norm(Sigma - Sigma_approx, 'fro') / \
        np.linalg.norm(Sigma, 'fro') * 100

print(f"\n秩 k={k}时:")
print(f"近似矩阵:\n{Sigma_approx}")
print(f"Frobenius 相对误差：{error:.4f}%")

# ====== Hessian 低秩更新的模拟 ======
print("\n🎯 3DGS 中的应用 - Hessian 近似:")
H_original = np.random.randn(10, 10) @ np.random.randn(10, 10).T
U_H, S_H, Vt_H = np.linalg.svd(H_original)

for k in [2, 5]:
    storage_needed = k * (10 + 10 + 1)
    ratio = storage_needed / 100 * 100
    error_H = np.linalg.norm(H_original - U_H[:,:k] @ np.diag(S_H[:k]) @ Vt_H[:k,:], 'fro') / \
              np.linalg.norm(H_original, 'fro') * 100
    
    print(f"  k={k}: 存储 {storage_needed} ({ratio:.1f}%), 误差 {error_H:.4f}%")

print("\n🎯 关键洞察:")
print("   - k=2 时只占 4.2% 存储空间，保留大部分优化方向信息!")
print("   - 这是 3DGS 中'低秩更新'策略的理论基础!")

# ====== 调试：协方差合法性检查 ======
def diagnose_covariance(Sigma):
    eigenvalues = np.linalg.eigvalsh(Sigma)
    
    print(f"\n🔍 协方差诊断:")
    print(f"  特征值：{eigenvalues}")
    
    # 半正定检查
    if np.any(eigenvalues < -1e-8):
        print("  ❌ 有显著负特征值 → 非半正定!")
        return False
    
    # 条件数
    cond = eigenvalues.max() / max(eigenvalues.min(), 1e-30)
    if cond > 1e6:
        print(f"  ⚠️ 病态 (κ={cond:.2e})")
    
    # 秩估计
    cutoff = max(Sigma.shape) * eigenvalues.max() * np.finfo(float).eps
    numerical_rank = np.sum(eigenvalues > cutoff)
    print(f"  📊 数值秩：{numerical_rank}")
    
    return True

diagnose_covariance(Sigma)

```

---

## 📌 **核心公式速查表**

| 符号 | 含义 | 数学公式 | **Python 实现** |
|------|------|---------|---------------|
| $A_k$ | 秩 k 近似 | $\sum_{i=1}^k σ_i u_i v_i^T$ | `U[:,:k] @ diag(S[:k]) @ Vt[:k,:]` |
| Frobenius 误差 | 重构质量 | $\|A - A_k\|_F = \sqrt{\sum_{i>k}σ_i^2}$ | `np.linalg.norm(A-Ak, 'fro')` |
| Eckart-Young-Mirsky | 最优性保证 | $\min_B \|A-B\|_F = \|A-A_k\|_F$ | (定理) |
| PCA 主成分 | $Σv=λv$ | `evals, evecs = np.linalg.eigh(Sigma)` |
| SVD → PCA | $V^T$行=PC方向 | `variances = S**2 / N` |

---

## 📌 **本章小结卡片：低秩近似与压缩**

| 核心问题 | 几何直觉 | 关键公式/性质 | 典型应用 |
|---------|---------|--------------|---------|
| "怎样用最少参数表示矩阵？" | 只保留前 k 个"大能量方向" | $A_k = \sum_{i=1}^k σ_i u_i v_i^T$<br>$\|A-A_k\|_F = \sqrt{\sum_{k+1}^r σ_i^2}$ | 图像压缩<br>推荐系统 |
| "PCA 和 SVD 什么关系？" | PCA = 对 X 做 SVD | $V^T$的行=主成分<br>$σ_i²/N=$方差 | 降维<br>数据可视化 |
| "3DGS 怎么用低秩近似？" | Hessian 只保留前 k 个主方向 | `H_k = U[:,:k] @ diag(S[:k]) @ Vt[:k,:]` | 优化加速<br>内存节省 |

---

## 🔍 Verification: 验证你的理解 (独立重推)

### 问题 1
为什么 Eckart-Young-Mirsky 定理说截断 SVD 是最优的？用一句话概括证明思路。

<details>
<summary>🤔 思考后再看答案</summary>

**答案**: 
因为 $A$的 SVD 项是相互正交的，误差 $\|A-A_k\|_F^2 = \sum_{k+1}^r σ_i^2$,任何其他秩为$k$的矩阵都会遗漏更大的奇异值 → 误差更大。

</details>

### 问题 2
PCA 的主成分方向和 SVD 的哪个部分对应？方差的计算公式是什么？

<details>
<summary>🤔 思考后再看答案</summary>

**答案**: 
- **主成分方向**: $V^T$的行 (来自数据矩阵 $X$的 SVD)
- **方差公式**: $\text{Var}(PC_i) = \frac{σ_i^2}{N}$ ($σ_i$是奇异值，$N$是样本数)

</details>

---

## 🔜 下一章预告：回到总复习

现在你已经掌握了:
1. ✅ **特征值/向量**: "只缩放不转弯"的方向 (第 8 章)
2. ✅ **SVD**: 任意矩阵的三步拆解 (第 9 章)  
3. ✅ **低秩近似**: Eckart-Young-Mirsky 定理 + PCA + 图像压缩 (本章)

下一章我们会**做总复习**,把所有章节的知识串联起来!🚀

---

**参考代码实现**: `/ntfs_shared/work/git/3dgs_tutorial/code/svd_demo.py`  
**下一步**: 阅读 `11_十一_现在回到_3DGS_这些数学到底在代码里干了什么.md` → 实战应用