# 第 13 章：五个可以直接跑的 Python / Matplotlib 实验

## 🎬 **为什么需要这五个实验？——从抽象到具象的桥梁**

如果你读完前面的章节后问：**"这些概念到底长什么样？代码里那几行数学公式对应什么几何变换？"** —— 那这一章就是为你准备的。

**核心哲学：** "跑过一遍代码"比"看十遍定义"更有用！

### **本章的叙事 spine：每个实验都按第一性原理组织**

1. **Problem:** 这个问题为什么存在？（不先问"为什么"，就不知道要算什么）
2. **Starting point:** 我们从什么已知条件出发？（站在巨人的肩膀上）
3. **Invention:** 数学/代码是怎么发明出来的？（不是魔法，是必然推导）
4. **Verification:** 如何验证你的理解？（自己重推一遍才是真懂）
5. **Example:** 可运行的代码演示（让直觉变成肌肉记忆）

### **五个实验的路线图**

| 实验 | 核心问题 | 关键洞察 |
|------|----------|----------|
| **1. 向量投影** | 点积为什么能衡量"对齐度"？ | 从余弦定理反推：$v_1\cdot v_2=\|v_1\|\|v_2\|\cos\theta$ |
| **2. 矩阵推网格** | det(A) 为什么代表面积缩放？ | 单位正方形→平行四边形，面积=$|ad-bc|=|\det(A)|$ |
| **3. 协方差椭圆** | Σ的每个元素代表什么？ | 特征分解：λ=半轴长度平方，q=主轴方向 |
| **4. 特征方向** | A·v=λ·v的几何意义是什么？ | 特征向量是"不变直线"——只缩放不转弯！|
| **5. SVD 三步拆解** | A = UΣVᵀ 怎么把圆变成椭圆？ | Vᵀ(输入旋转) → Σ(拉伸) → U(输出旋转) |

准备好了吗？我们一个一个来。🔥

---

### 🧪 实验一：向量、投影、点积——为什么夹角决定"对齐度"？

> **Problem:** 两个向量的关系，到底怎么量化？想象你有一个目标方向 $v_1$（比如相机光轴），然后有一个信号方向 $v_2$。你怎么知道它们"对齐"还是"垂直"？
>
> **Starting point:** 已知：①向量模长 $\|v\|$，②夹角余弦 $\cos\theta$，③投影的几何意义
>
> **Invention:** 从余弦定理反推！三角形三边 $\|v_1\|、\|v_2\|、\|v_2-v_1\|$：
> $$\|v_2 - v_1\|^2 = \|v_1\|^2 + \|v_2\|^2 - 2\|v_1\|\|v_2\|\cos\theta$$
>
> 展开左边：$(v_2-v_1)\cdot(v_2-v_1)=\|v_2\|^2-2 v_1\cdot v_2+\|v_1\|^2$，对比得：**$v_1\cdot v_2=\|v_1\|\|v_2\|\cos\theta$** ✅
>
> **💡 核心洞察：** 
> - 几何版：$\|v_1\|\|v_2\|\cos\theta$ —— 衡量"对齐度"
> - 代数版：$x_1 x_2 + y_1 y_2$ —— 坐标分量乘积之和
>
> **Verification:** <details><summary>问题：投影向量的公式？</summary>答案：$\text{proj}_{v_1}(v_2) = \frac{v_1\cdot v_2}{\|v_1\|^2} v_1$</details>
>
> **Example:**

```python
import numpy as np
import matplotlib.pyplot as plt

# ====== 生成两个向量 ======
v1 = np.array([3.0, 1.0])
v2_base = np.array([1.0, 2.0])


def plot_projection(v1, v2, angle_deg):
    """绘制向量和投影关系"""
    
    # 旋转 v2
    theta = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    v2 = rotation_matrix @ v2_base
    
    # 计算点积和投影长度
    dot_product = np.dot(v1, v2)
    proj_length = dot_product / np.linalg.norm(v1)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制向量
    ax.quiver(0, 0, v1[0], v1[1], color='blue', scale=5, width=0.005, label='v₁')
    ax.quiver(0, 0, v2[0], v2[1], color='red', scale=5, width=0.005, label='v₂')
    
    # 绘制投影向量 (沿 v₁方向的 v₂分量)
    proj_v1 = (dot_product / np.dot(v1, v1)) * v1
    ax.quiver(0, 0, proj_v1[0], proj_v1[1], color='green', scale=5, width=0.005, 
              label=f'投影长度={proj_length:.2f}')
    
    # 绘制角度弧线
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    arc_radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) * 0.7
    theta_vals = np.linspace(0, angle, 50) if angle < np.pi else np.linspace(angle, 0, 50)
    
    # 绘制直角标记 (投影处)
    ax.plot([proj_v1[0], v2[0]], [proj_v1[1], v2[1]], 'k--', alpha=0.3)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f'向量投影关系 (夹角={angle_deg}°)\n点积 = {dot_product:.2f}, 投影长度 = {proj_length:.2f}')
    
    plt.show()


print("=" * 70)
print("🧪 实验一：向量、投影、点积")
print("=" * 70)

# ====== 测试不同夹角 ======
for angle in [0, 30, 60, 90, 120, 150, 180]:
    print(f"\n夹角 = {angle}°:")
    
    # 计算理论值
    theta = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    v2_rotated = rotation_matrix @ v2_base
    
    dot = np.dot(v1, v2_rotated)
    proj_len = dot / np.linalg.norm(v1)
    
    print(f"  点积：{dot:.2f}")
    print(f"  投影长度：{proj_len:.2f}")
    
    if angle == 90:
        plot_projection(v1, v2_rotated, angle)

print("\n🎯 关键洞察:")
print("   - 夹角=0°时点积最大 → 向量完全对齐")
print("   - 夹角=90°时点积=0 → 正交 (无投影)")
print("   - 夹角>90°时点积为负 → 反向分量")

```

---

### 🧪 实验二：矩阵如何推整张网格——为什么 det(A) 代表面积缩放？

> **Problem:** 教科书说"矩阵是线性变换的表示"，但**一个 $2\times 2$ 的数字方阵，怎么就能描述"推一张网"的动作？**
>
> **Starting point:** 已知：①可加性 $f(u+v)=f(u)+f(v)$，②齐次性 $f(cu)=cf(u)$
>
> **Invention:** 
> - 矩阵的每一列告诉你单位向量 $e_1=[1,0]^\top$ 和 $e_2=[0,1]^\top$ 被推到哪去了
> - 考虑单位正方形，变换后变成平行四边形，面积公式是 $|ad-bc|=|\det(A)|$ ✅
>
> **💡 核心洞察：** $|\det(A)|$ = 单位正方形被推成的平行四边形面积 = **面积缩放因子**！✅
>
> **Verification:** <details><summary>问题：det(A)=0 意味着什么？</summary>答案：两列向量共线 → 整个平面被"压塌"到一条直线 → 不可逆变换!</details>
>
> **Example:**

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_grid_transformation(A, title):
    """可视化矩阵对单位网格的变换"""
    
    # 创建网格点
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    
    # 展平为 (2, N) 矩阵
    points = np.vstack([X.flatten(), Y.flatten()])
    
    # 应用变换
    transformed = A @ points
    
    # 重新网格化
    X_trans = transformed[0].reshape(10, 10)
    Y_trans = transformed[1].reshape(10, 10)
    
    # ====== 绘图 ======
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制原始网格 (浅色)
    for xi in x:
        ax.plot([xi]*20, [y[0]]*10 + np.linspace(-2, 2, 10)*2, 'c', alpha=0.2)
    for yi in y:
        ax.plot([x[0]]*10 + np.linspace(-2, 2, 10)*2, [yi]*20, 'c', alpha=0.2)
    
    # 绘制变换后的网格 (深色)
    ax.grid(False)
    for i in range(10):
        ax.plot(X_trans[i, :], Y_trans[i, :], 'b-', linewidth=1.5)
        ax.plot(X_trans[:, i], Y_trans[:, i], 'b-', linewidth=1.5)
    
    # 绘制坐标轴和单位正方形变换后的位置
    ax.quiver(0, 0, A[0, 0], A[1, 0], color='red', scale=3, width=0.005, label='e₁→')
    ax.quiver(0, 0, A[0, 1], A[1, 1], color='green', scale=3, width=0.005, label='e₂→')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title + f"\ndet(A) = {np.linalg.det(A):.2f}")
    
    plt.show()


print("=" * 70)
print("🧪 实验二：矩阵如何推整张网格")
print("=" * 70)

# ====== 1. 缩放矩阵 ======
A_scale = np.array([[2.0, 0], [0, 0.5]])
plot_grid_transformation(A_scale, "缩放 (x×2, y×0.5)")

# ====== 2. 旋转矩阵 ======
theta = np.pi / 4  # 45°
A_rotate = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
plot_grid_transformation(A_rotate, "旋转 (45°)")

# ====== 3. 剪切矩阵 ======
A_shear = np.array([[1.0, 1.0], [0, 1.0]])
plot_grid_transformation(A_shear, "剪切 (x += y)")

# ====== 4. 反射矩阵 ======
A_reflect = np.array([[-1.0, 0], [0, 1.0]])
plot_grid_transformation(A_reflect, "反射 (关于 y 轴)")

print("\n🎯 关键洞察:")
print("   - det>1:面积放大，det<1:面积缩小")
print("   - det=0:网格被压塌成线或点 (不可逆!)")
print("   - det<0:方向翻转 (手性改变)")

```

---

### 🧪 实验三：协方差如何定义椭圆——Σ的每个元素代表什么？

> **Problem:** 协方差矩阵 $\Sigma=\begin{bmatrix}\sigma_{xx}&\sigma_{xy}\\\sigma_{yx}&\sigma_{yy}\end{bmatrix}$，看起来像一堆统计术语。但它在几何上到底是什么？**为什么一个 $2\times 2$ 矩阵能描述一个椭圆？**
>
> **Starting point:** 已知：二维高斯分布的等概率轮廓线 $(\mathbf{x}-\mu)^\top \Sigma^{-1} (\mathbf{x}-\mu)=c$
>
> **Invention:** 
> - 特征分解 $\Sigma=Q\Lambda Q^\top$，其中 $Q$ 是正交矩阵（特征向量），$\Lambda=\text{diag}(\lambda_1,\lambda_2)$
> - 令 $y=Q^\top(\mathbf{x}-\mu)$（旋转到特征向量坐标系）：$\frac{{y_1}^2}{\lambda_1}+\frac{{y_2}^2}{\lambda_2}=c$ → **这是标准椭圆方程！** ✅
>
> **💡 核心洞察：** 
> - $\lambda_i$ → 半轴长度的平方
> - $q_i$ → 第 $i$ 个主轴的方向向量
> - $\sigma_{xy}\neq0$ → 椭圆倾斜（x,y 有相关性）
>
> **Verification:** <details><summary>问题：Σ[i,i] 和 Σ[i,j](i≠j) 分别代表什么？</summary>答案：Σ[i,i]=方差=宽度，Σ[i,j]=协方差=倾斜程度!</details>
>
> **Example:**

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_covariance_ellipse(Sigma, title):
    """从协方差矩阵绘制椭圆"""
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    # 生成单位圆参数 (3σ范围)
    t = np.linspace(0, 2*np.pi, 100)
    circle_x = np.sqrt(eigenvalues[0]) * 3 * np.cos(t)
    circle_y = np.sqrt(eigenvalues[1]) * 3 * np.sin(t)
    
    # 旋转到主轴方向
    rotated_x, rotated_y = [], []
    for x, y in zip(circle_x, circle_y):
        transformed = eigenvectors @ np.array([x, y])
        rotated_x.append(transformed[0])
        rotated_y.append(transformed[1])
    
    # ====== 绘图 ======
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(rotated_x, rotated_y, 'b-', linewidth=2, label='3σ椭圆')
    ax.plot(0, 0, 'ro', markersize=8, label='中心')
    
    # 绘制主轴方向
    for i in range(2):
        axis_length = np.sqrt(eigenvalues[i]) * 3
        end_point = eigenvectors[:, i] * axis_length
        ax.quiver(0, 0, end_point[0], end_point[1], 
                  color='red' if i==0 else 'green', 
                  scale=5, width=0.003,
                  label=f'主轴{i+1}: σ={np.sqrt(eigenvalues[i]):.2f}')
    
    ax.axis('equal')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title + f"\n特征值：{eigenvalues:.2f}, 行列式：{np.linalg.det(Sigma):.2f}")
    
    plt.show()


print("=" * 70)
print("🧪 实验三：协方差如何定义椭圆")
print("=" * 70)

# ====== 1. 圆形 (各向同性) ======
Sigma_1 = np.array([[2.0, 0], [0, 2.0]])
plot_covariance_ellipse(Sigma_1, "各向同性 (圆)")

# ====== 2. 拉伸的椭圆 (主轴对齐坐标轴) ======
Sigma_2 = np.array([[4.0, 0], [0, 1.0]])
plot_covariance_ellipse(Sigma_2, "拉伸 (x:σ=2, y:σ=1)")

# ====== 3. 倾斜的椭圆 (有相关性) ======
Sigma_3 = np.array([[2.5, 1.5], [1.5, 2.0]])
plot_covariance_ellipse(Sigma_3, "倾斜 (Σ[0,1]=1.5)")

# ====== 4. 狭长的椭圆 (强相关) ======
Sigma_4 = np.array([[5.0, 4.5], [4.5, 5.0]])
plot_covariance_ellipse(Sigma_4, "狭长 (Σ[0,1]=4.5)")

print("\n🎯 关键洞察:")
print("   - Σ[i,i] 控制该方向的宽度")
print("   - Σ[i,j](i≠j) 决定椭圆是否倾斜")
print("   - det(Σ)=λ₁λ₂=椭圆的'面积'")

```

---

### 🧪 实验四：特征方向为什么特殊——A·v=λ·v的几何意义

> **Problem:** 想象你把一张纸任意拉伸、旋转、剪切。总有一些向量在变换后**只是变长或变短，但方向不变**。这些方向有多重要？**答案：**它们是理解矩阵的"钥匙"！✅
>
> **Starting point:** 特征方程的定义：$Av=\lambda v$（$v\neq0$）
>
> **Invention:** 
> - 从单位圆 $\{x:\|x\|=1\}$ 开始，应用线性变换 $A$ → 输出是 $\{Ax:\|x\|=1\}$，这是一个**椭圆** ✅
> - 在输入空间画射线（方向不同的向量），应用 $A$ 后：一般方向弯曲成曲线，**特征方向仍然是直线！** ✅
>
> **💡 核心洞察：** 特征向量定义了变换的"主轴"——沿着这些轴，变换就是纯拉伸/压缩。
>
> **Verification:** <details><summary>问题：旋转矩阵有实特征值吗？</summary>答案：没有！(除非旋转角是 0°或 180°) → 所有方向都转弯了!</details>
>
> **Example:**

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_eigenvectors(A, title):
    """可视化特征向量的特殊性"""
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # ====== 创建单位圆和变换后的椭圆 ======
    t = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(t)
    circle_y = np.sin(t)
    
    # 应用变换
    transformed_x, transformed_y = [], []
    for x, y in zip(circle_x, circle_y):
        result = A @ np.array([x, y])
        transformed_x.append(result[0])
        transformed_y.append(result[1])
    
    # ====== 绘图 ======
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：原始空间
    ax1.plot(circle_x, circle_y, 'b--', alpha=0.3, label='单位圆')
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i] * 2
        ax1.quiver(0, 0, v[0], v[1], color='red' if abs(eigenvalues[i])>1 else 'green', 
                  scale=3, width=0.005, labels=f'v_{i+1}, λ={eigenvalues[i]:.2f}' if i==0 else "")
    ax1.set_title(f"输入空间：单位圆\n{title}")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.axis('equal')
    ax1.grid(alpha=0.3)
    
    # 右图：变换后的空间
    ax2.plot(transformed_x, transformed_y, 'g-', linewidth=2, label='变换后椭圆')
    for i in range(len(eigenvalues)):
        v_transformed = A @ eigenvectors[:, i] * 2
        ax2.quiver(0, 0, v_transformed[0], v_transformed[1], 
                  color='red' if abs(eigenvalues[i])>1 else 'green',
                  scale=3, width=0.005, label=f'A·v_{i+1}=λ{eigenvalues[i]:.2f}' if i==0 else "")
    
    # 绘制特征向量在原空间的位置 (投影到右图)
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i] * 2
        ax2.quiver(0, 0, v[0], v[1], color='blue', scale=3, width=0.005, alpha=0.5)
    
    ax2.set_title("输出空间：椭圆\n(A·v = λ·v → 只缩放不转弯!)")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.axis('equal')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


print("=" * 70)
print("🧪 实验四：特征方向为什么特殊")
print("=" * 70)

# ====== 1. 对称矩阵 (实特征值，正交特征向量) ======
A_sym = np.array([[3.0, 1.0], [1.0, 2.0]])
plot_eigenvectors(A_sym, "对称矩阵：λ₁=3.62, λ₂=1.38")

# ====== 2. 非对称但可对角化 ======
A_nonsym = np.array([[2.0, 1.0], [0.5, 1.5]])
plot_eigenvectors(A_nonsym, "非对称：λ₁=2.37, λ₂=1.13")

# ====== 3. 旋转矩阵 (复特征值!) ======
theta = np.pi / 4
A_rotate = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
print(f"\n旋转矩阵 A: λ = {np.linalg.eig(A_rotate)[0]}")
print("⚠️ 注意：实数域内没有特征向量 (所有方向都会转弯!)")

# ====== 数值验证 "只缩放不拐弯" ======
print("\n🔍 验证：A·v = λ·v")
evals, evecs = np.linalg.eig(A_sym)
for i in range(len(evals)):
    v = evecs[:, i]
    Av = A_sym @ v
    lambda_v = evals[i] * v
    
    print(f"\n  v_{i+1} = {v}")
    print(f"  A·v = {Av}")
    print(f"  λ·v = {lambda_v}")
    print(f"  ✅ {'相等!' if np.allclose(Av, lambda_v) else '不等!'}")

print("\n🎯 关键洞察:")
print("   - 特征向量：变换后只被缩放，不转弯!")
print("   - 特征值：缩放因子 (λ>1:拉长，|λ|<1:压缩)")
print("   - 旋转矩阵：实数域无特征向量 → 所有方向都转弯!")

```

---

### 🧪 实验五：SVD 如何把圆变成椭圆——A = UΣVᵀ的三步拆解

> **Problem:** 对称矩阵可以用特征分解 $A=Q\Lambda Q^\top$，但**非对称矩阵怎么理解？**能不能也找到"输入主轴"和"输出主轴"？
>
> **Starting point:** 已知：①特征分解要求 A 是方阵且有完备特征向量集；②旋转矩阵没有实特征值
>
> **Invention:** SVD（奇异值分解）: $A=U\Sigma V^\top$ 
> - Step 1: 用 $V^\top$ 旋转输入圆 → 仍然是单位圆
> - Step 2: 用 $\Sigma$ 拉伸 → 变成椭圆（长轴为 $\sigma_i$）
> - Step 3: 用 $U$ 旋转 → 最终输出椭圆 ✅
>
> **💡 核心洞察：** 
> - SVD = Vᵀ(旋转) → Σ(拉伸) → U(旋转)
> - vᵢ是输入主轴，uᵢ是输出主轴
> - $A\cdot v_i=\sigma_i\cdot u_i$（核心关系！）✅
>
> **Verification:** <details><summary>问题：特征分解和 SVD 的区别？</summary>答案：特征分解要求 A 有完备特征向量集；SVD 对任何矩阵都成立，且输入/输出主轴可以完全不同!</details>
>
> **Example:**

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_svd_three_steps(A, title):
    """可视化 SVD 的三步拆解"""
    
    # 计算 SVD
    U, S, Vt = np.linalg.svd(A)
    
    # ====== Step 1:输入空间 (Vᵀ旋转) ======
    t = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(t)
    circle_y = np.sin(t)
    
    # ====== Step 2:拉伸 (Σ) ======
    stretched_x = S[0] * np.cos(t) if len(S)>0 else [0]*100
    stretched_y = S[1] * np.sin(t) if len(S)>1 else [0]*100
    
    # ====== Step 3:输出空间 (U 旋转) ======
    ellipse_x, ellipse_y = [], []
    for x, y in zip(stretched_x, stretched_y):
        transformed = U @ np.array([x, y])
        ellipse_x.append(transformed[0])
        ellipse_y.append(transformed[1])
    
    # ====== 绘图 ======
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Step 1: Vᵀ旋转 (输入空间)
    v1_t, v2_t = Vt[0], Vt[1]
    axes[0].plot(circle_x, circle_y, 'b-', linewidth=2, label='单位圆')
    axes[0].quiver(0, 0, v1_t[0]*2, v1_t[1]*2, color='red', scale=3, width=0.005, 
                  labels=f'v₁ (σ₁={S[0]:.2f})' if len(S)>0 else "")
    axes[0].quiver(0, 0, v2_t[0]*2, v2_t[1]*2, color='green', scale=3, width=0.005,
                  labels=f'v₂ (σ₂={S[1]:.2f})' if len(S)>1 else "")
    axes[0].set_title(f"Step 1: Vᵀ\n(输入主轴)\nσ₁={S[0]:.2f}, σ₂={S[1] if len(S)>1 else 'N/A'}")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].axis('equal')
    axes[0].grid(alpha=0.3)
    
    # Step 2: Σ拉伸 (中间空间)
    axes[1].plot(stretched_x, stretched_y, 'g-', linewidth=2, label='沿主轴拉伸')
    axes[1].quiver(0, 0, S[0]*2 if len(S)>0 else 0, 0, color='red', scale=3, width=0.005)
    if len(S)>1:
        axes[1].quiver(0, 0, 0, S[1]*2, color='green', scale=3, width=0.005)
    axes[1].set_title(f"Step 2: Σ\n(拉伸 σ₁={S[0]:.2f}, σ₂={S[1] if len(S)>1 else 'N/A'})")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].axis('equal')
    axes[1].grid(alpha=0.3)
    
    # Step 3: U 旋转 (输出空间)
    u1, u2 = U[:, 0], U[:, 1]
    axes[2].plot(ellipse_x, ellipse_y, 'b-', linewidth=2, label='最终椭圆')
    axes[2].quiver(0, 0, S[0]*u1[0], S[0]*u1[1], color='red', scale=3, width=0.005)
    if len(S)>1:
        axes[2].quiver(0, 0, S[1]*u2[0], S[1]*u2[1], color='green', scale=3, width=0.005)
    axes[2].set_title(f"Step 3: U\n(输出主轴)\nA = UΣVᵀ")
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].axis('equal')
    axes[2].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


print("=" * 70)
print("🧪 实验五：SVD 如何把圆变成椭圆")
print("=" * 70)

# ====== 1. 对角矩阵 (只有拉伸，没有旋转) ======
A_diag = np.array([[3.0, 0], [0, 1.0]])
plot_svd_three_steps(A_diag, "对角矩阵：U=I, V=I")

# ====== 2. 一般对称矩阵 (有旋转 + 拉伸) ======
A_sym = np.array([[4.0, 1.5], [1.5, 2.0]])
plot_svd_three_steps(A_sym, "对称矩阵：U≠V (主轴旋转)")

# ====== 3. 非对称矩阵 (完全不同的输入/输出主轴) ======
A_nonsym = np.array([[3.0, 2.0], [1.0, 1.0]])
plot_svd_three_steps(A_nonsym, "非对称：U≠V, 输入/输出主轴完全不同!")

# ====== 数值验证 A @ v_i = σ_i * u_i ======
print("\n🔍 验证核心关系：A·vᵢ = σᵢ·uᵢ")
for i_test, A in enumerate([A_diag, A_sym, A_nonsym], 1):
    U, S, Vt = np.linalg.svd(A)
    
    print(f"\n矩阵 {i_test}:")
    for j in range(len(S)):
        vj = Vt[j]
        uj = U[:, j]
        
        left = A @ vj
        right = S[j] * uj
        
        match = "✅" if np.allclose(left, right) else "❌"
        print(f"  i={j+1}: {match} A·v_{j+1} ≈ σ_{j+1}·u_{j+1}")

print("\n🎯 关键洞察:")
print("   - SVD = Vᵀ(旋转) → Σ(拉伸) → U(旋转)")
print("   - vᵢ是输入主轴，uᵢ是输出主轴")
print("   - σᵢ是在该方向上的放大倍数!")

```

---

## 📌 **实验总结卡片**

| 实验 | 核心洞察 | 可运行代码位置 |
|------|---------|---------------|
| **1. 向量投影** | 点积=对齐度，90°时=0 | `plot_projection(v1, v2, angle)` |
| **2. 矩阵推网格** | det>1 放大，det<0 翻转 | `plot_grid_transformation(A, title)` |
| **3. 协方差椭圆** | Σ[i,i]=宽度，Σ[i,j]=倾斜 | `plot_covariance_ellipse(Sigma, title)` |
| **4. 特征方向** | A·v = λ·v(只缩放不转弯) | `plot_eigenvectors(A, title)` |
| **5. SVD 三步拆解** | A = UΣVᵀ (转→拉→转) | `plot_svd_three_steps(A, title)` |

---

## 🎯 **如何最大化实验价值？**

### ✅ **跑完每一遍后问自己:**

1. "如果我改这个参数，图像会怎么变？"
2. "这个现象对应哪个数学公式？"
3. "在 3DGS 里，这个概念用在哪一步？"

### 🔄 **实验 - 理论循环:**

```
跑实验 → 看到现象 → 查公式 → 再跑实验验证 → 形成直觉
```

**推荐顺序**:先做实验 1-5,再回头读第 2-9 章，感觉会完全不同!🚀

---

## 📌 **本章总结：五个实验的共通主线**

这五个实验看似独立，其实有一条贯穿始终的主线：**从代数到几何的桥梁。** 

每个实验都回答了同一个问题：**代码里的这些数学公式，到底对应什么样的几何变换？**

| 实验 | 公式 | 几何变换 |
|------|------|----------|
| 1. 点积 | $v_1\cdot v_2$ | 投影长度（对齐度） |
| 2. det(A) | $ad-bc$ | 平行四边形面积缩放因子 |
| 3. Σ的特征分解 | $Q\Lambda Q^\top$ | 椭圆的主轴方向和半轴长度 |
| 4. Av=λv | $Av=\lambda v$ | "不变直线"（只缩放不转弯） |
| 5. SVD | $U\Sigma V^\top$ | 三步拆解：旋转→拉伸→旋转 |

**完成这五个实验后，你得到的不是零散的知识点，而是一个完整的线性代数直觉框架。** 🔥
