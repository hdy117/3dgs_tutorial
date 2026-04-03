# 练习题：特征值与特征向量——变换的"主轴方向"

---

## 第 1 题：缩放矩阵的特征值与特征向量（基础）

给定对角矩阵:
$$S = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}$$

**问题**:
1. $S$ 的特征值和特征向量是什么？
2. 从几何角度解释：这个变换对哪些方向影响最小？

<details>
<summary>💡 提示</summary>

对角矩阵有什么特殊性质？尝试用定义 $A\mathbf{v}=\lambda\mathbf{v}$ 直接验证。
</details>

<details>
<summary>🔑 答案与解析</summary>

**解**:

观察矩阵形式，这是沿坐标轴的缩放变换：
- x 轴方向拉伸 3 倍
- y 轴方向拉伸 2 倍

**特征值** = 对角线元素：$\lambda_1=3, \lambda_2=2$

**验证**:
$$S\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}3\\0\end{bmatrix} = 3\times\begin{bmatrix}1\\0\end{bmatrix}, \quad S\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}0\\2\end{bmatrix} = 2\times\begin{bmatrix}0\\1\end{bmatrix}$$

所以特征向量是 $\mathbf{v}_1=[1,0]^T$ (x 轴), $\mathbf{v}_2=[0,1]^T$ (y 轴)。

**几何解释**:
- x 轴被拉长最多（3 倍）
- y 轴被拉长较少（2 倍）
- **影响最小**的方向是 y 轴方向（相对保持较好）

✅ **关键点**:对角矩阵的特征值就是**对角线元素**,特征向量就是坐标轴方向！

</details>

---

## 第 2 题：旋转矩阵没有实特征值

给定绕原点逆时针旋转 $90^\circ$ 的矩阵:
$$R_{90°} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

**问题**:
1. 求这个矩阵的特征方程 $\det(R-\lambda I)=0$
2. 为什么它没有实特征值？从几何角度解释。

<details>
<summary>💡 提示</summary>

先计算行列式，得到一个二次方程。看看判别式是正还是负。
</details>

<details>
<summary>🔑 答案与解析</summary>

**解**:

Step 1: 写出 $R-\lambda I$:
$$R-\lambda I = \begin{bmatrix} -\lambda & -1 \\ 1 & -\lambda \end{bmatrix}$$

Step 2: 计算行列式:
$$\det(R-\lambda I) = (-\lambda)(-\lambda) - (-1)(1) = \lambda^2 + 1$$

Step 3: 解特征方程 $\lambda^2+1=0$:
$$\lambda^2 = -1 \Rightarrow \lambda = \pm i$$

**为什么没有实特征值？**

几何上，旋转 $90^\circ$ 会把所有方向都"转走":
- x 轴转到 y 轴
- y 轴转到负 x 轴
- 45°线转到 135°线

没有一个方向经过旋转后还留在原来的直线上（除非旋转 $0^\circ$ 或 $180^\circ$）!

✅ **关键点**:非零角度的旋转矩阵在**实数空间没有特征向量**,特征值是复数 $e^{\pm i\theta} = \cos\theta \pm i\sin\theta$。
</details>

---

## 第 3 题：对称矩阵的特征分解验证

给定对称矩阵:
$$A = \begin{bmatrix} 4 & 1 \\ 1 & 3 \end{bmatrix}$$

**任务**:
1. 手动计算特征值和特征向量（保留两位小数）
2. 验证 $A = Q\Lambda Q^T$

<details>
<summary>💡 提示</summary>

先解二次方程 $\det(A-\lambda I)=0$,然后用 $(A-\lambda I)\mathbf{v}=0$ 求特征向量。注意对称矩阵的特征向量自动正交！
</details>

<details>
<summary>🔑 答案与解析</summary>

**Step 1: 计算特征值**

$$\det(A-\lambda I) = \begin{vmatrix} 4-\lambda & 1 \\ 1 & 3-\lambda \end{vmatrix} = (4-\lambda)(3-\lambda) - 1$$
$$= \lambda^2 - 7\lambda + 11 = 0$$

解得：$\lambda = \frac{7 \pm \sqrt{5}}{2}$

所以 $\lambda_1 \approx 5.62, \lambda_2 \approx 1.38$

**Step 2: 求特征向量**

对于 $\lambda_1=5.62$:
$$-1.62x + y = 0 \Rightarrow \mathbf{q}_1 \approx [0.53, 0.85]^T$$

利用正交性：$\mathbf{q}_2 = [-0.85, 0.53]^T$

**验证 $A=Q\Lambda Q^T$**:
$$\begin{bmatrix} 0.53 & -0.85 \\ 0.85 & 0.53 \end{bmatrix}\begin{bmatrix} 5.62 & 0 \\ 0 & 1.38 \end{bmatrix}\begin{bmatrix} 0.53 & 0.85 \\ -0.85 & 0.53 \end{bmatrix} = \begin{bmatrix} 4 & 1 \\ 1 & 3 \end{bmatrix} ✅$$

✅ **关键点**:对称矩阵的特征向量**自动正交**!
</details>

---

## 第 4 题：几何直觉——剪切变换只有一个特征方向

考虑剪切矩阵:
$$H = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}, \quad (k>0)$$

**问题**:
1. $H$ 的特征值和特征向量是什么？
2. 为什么只有一个特征方向？

<details>
<summary>💡 提示</summary>

这是上三角矩阵，特征值就是对角线元素。验证哪些向量经过剪切后还留在同一直线上。
</details>

<details>
<summary>🔑 答案与解析</summary>

**特征值**: $\lambda_1=\lambda_2=1$（重根）

**特征向量**: $(H-I)\mathbf{v}=0 \Rightarrow ky=0$,所以只有 x 轴方向 $[1,0]^T$

**几何解释**:
- x 轴：$(x,0) \to (x,0)$，不变 ✅
- y 轴：$(0,y) \to (ky,y)$，被推歪了 ❌

✅ **关键点**:剪切矩阵有重根但只有一个特征方向 → **不可对角化**!
</details>

---

## 第 5 题：3DGS 协方差矩阵到椭球参数

给定对称正定矩阵:
$$\Sigma = \begin{bmatrix} 4.0 & 1.5 & 0.8 \\ 1.5 & 3.0 & 1.2 \\ 0.8 & 1.2 & 2.0 \end{bmatrix}$$

**任务**:用 Python 提取椭球参数

<details>
<summary>💡 提示</summary>

使用 `numpy.linalg.eigh`，开方得到标准差。
</details>

<details>
<summary>🔑 答案与解析</summary>

```python
import numpy as np

Sigma = np.array([[4.0, 1.5, 0.8], [1.5, 3.0, 1.2], [0.8, 1.2, 2.0]])
eigenvalues, axes_dirs = np.linalg.eigh(Sigma)

# 排序（从大到小）
sort_idx = np.argsort(eigenvalues)[::-1]
axis_lengths = np.sqrt(eigenvalues[sort_idx])      # [2.05, 1.57, 1.15]

print("特征值 (方差):", eigenvalues[sort_idx])
print("半轴长度 σ=√λ:", axis_lengths)
```

✅ **关键点**:对称矩阵 → 实特征值 + 正交特征向量 = 清晰椭球参数!
</details>

---

## 第 6 题：非对称矩阵的陷阱——椭圆主轴≠特征方向！

给定**非对称**矩阵:
$$A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$$

**问题**:椭圆的长轴是特征向量吗？为什么？

<details>
<summary>💡 提示</summary>

用 SVD 看椭圆主轴，对比特征向量的方向。
</details>

<details>
<summary>🔑 答案与解析</summary>

**答案**:❌ **不是**!

**原因**:
- **特征向量**:回答"哪个方向只缩放不转弯？"
- **椭圆主轴**(奇异向量):回答"椭圆的长轴朝哪？"

对于非对称矩阵，两者不同！只有当 $A$**对称**时才重合。

验证：计算 $AA^T = \begin{bmatrix} 10 & 2 \\ 2 & 4 \end{bmatrix}$ 的特征向量才是椭圆主轴!
</details>

---

## 第 7 题：为什么旋转矩阵没有实特征值？

$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**问题**:推导特征方程，证明当 $\theta\notin\{0, \pi\}$时没有实特征值。

<details>
<summary>💡 提示</summary>

计算 $\det(R_\theta-\lambda I)$，用三角恒等式化简。
</details>

<details>
<summary>🔑 答案与解析</summary>

**特征方程**: $\lambda^2 - 2\lambda\cos\theta + 1 = 0$

**判别式**: $4\cos^2\theta-4 < 0$ (当 $|\cos\theta|<1$) → **无实根**!

**几何解释**:旋转定义就是"改变角度",所以所有方向都转弯，不可能有"原地不动"的方向!

特征值是复数：$\lambda = e^{\pm i\theta} = \cos\theta \pm i\sin\theta$
</details>

---

## 第 8 题：PCA 主成分分析实战

给定点云:
```python
np.random.seed(42)
points = np.random.multivariate_normal([0, 0], [[10, 5], [5, 3]], size=1000)
```

**任务**:找数据最分散的方向（主成分）

<details>
<summary>💡 提示</summary>

PCA = 协方差矩阵的特征分解。最大特征值对应的方向就是主成分!
</details>

<details>
<summary>🔑 答案与解析</summary>

```python
# Step 1: 中心化
centered = points - np.mean(points, axis=0)

# Step 2: 协方差矩阵
Sigma = centered.T @ centered / len(points)

# Step 3: 特征分解找主成分
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
sort_idx = np.argsort(eigenvalues)[::-1]
principal_axis = eigenvectors[:, sort_idx[0]]  # [0.85, 0.53]

# Step 4: 验证投影后方差最大
projected = points @ principal_axis
print(f"主成分贡献：{eigenvalues[sort_idx[0]]/np.trace(Sigma)*100:.1f}%")  # ~79%
```

✅ **关键点**:协方差矩阵对称 → 特征向量正交 → PCA 各维度不相关!
</details>

---

## 第 9 题：验证你的代码实现

检查函数是否正确:

<details>
<summary>💡 提示</summary>

用 `np.linalg.matrix_rank(P)`检查 P 是否满秩，检测不可对角化的矩阵。
</details>

<details>
<summary>🔑 答案与解析</summary>

**测试结论**:
- ✅ **对称矩阵**：成功 (P 满秩)
- ✅ **一般可对角化矩阵**:成功  
- ❌ **缺陷矩阵**(如剪切):失败 → P 不满秩，无法求逆

**检测方法**:
```python
rank = np.linalg.matrix_rank(P)
assert rank == A.shape[0], "矩阵不可对角化!"
```

✅ **关键点**:只有 n 个线性无关特征向量才能被对角化!
</details>

---

## 📝 本章核心总结卡片

| 概念 | 关键公式/性质 | 几何意义 | Python API |
|------|--------------|---------|-----------|
| **特征向量** | $A\mathbf{v}=\lambda\mathbf{v}$ | "只缩放不转弯"的方向 | `np.linalg.eig(A)` |
| **特征值** | $\det(A-\lambda I)=0$ | 缩放因子 | `eigenvalues` |
| **对称矩阵** | $A=Q\Lambda Q^T$ | 特征向量自动正交 | `np.linalg.eigh(A)` |
| **非对称陷阱** | 特征方向≠椭圆主轴 | 需用 SVD 找真实主轴 | `np.linalg.svd(A)` |
| **旋转矩阵** | $\lambda=e^{\pm i\theta}$ (复数) | 所有方向都转弯 | 无实特征值 ❌ |

---

## 🔜 下一章预告：SVD——任意矩阵的三步拆解

你已经掌握了:
1. ✅ 特征值/向量 = "最自然"的方向
2. ✅ 特征分解 = 找到让变换变简单的坐标系  
3. ⚠️ 但要求方阵 + 可对角化

**问题**:一般矩阵怎么拆解？**SVD**来了！🔥
