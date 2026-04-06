"""
第 13 章：五个可以直接跑的 Python/Matplotlib 实验
==================================================

这个 Notebook 包含了线性代数教程中最核心的 5 个可视化实验。
建议先跑完这些实验，再回头读理论章节，会有完全不同的感受!

运行顺序：
1. 实验一：向量、投影、点积 (理解"对齐度")
2. 实验二：矩阵如何推整张网格 (理解"空间动作")  
3. 实验三：协方差如何定义椭圆 (理解"几何编码")
4. 实验四：特征方向为什么特殊 (理解"只缩放不转弯")
5. 实验五：SVD 如何把圆变成椭圆 (理解"三步拆解")

对应章节：第 13 章，也可对照第 3,5,6,8,9 章
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示 (如果需要)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 🧪 实验一：向量、投影、点积
# ============================================================
"""
目标：看懂"对齐度"和"投影长度"的动态关系。

关键公式:
- 点积：u·v = ||u|| ||v|| cosθ (cosθ=1 完全同向，0 正交，-1 反向)
- 投影长度：proj_u(v) = (u·v)/||u||² × u
"""

def experiment_1_vectors_projection():
    """实验一：向量、投影、点积"""
    
    print("=" * 70)
    print("🧪 实验一：向量、投影、点积")
    print("=" * 70)
    
    # 生成两个向量
    v1 = np.array([3.0, 1.0])
    v2_base = np.array([1.0, 2.0])
    
    def plot_projection(angle_deg):
        """绘制不同夹角下的投影关系"""
        
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
        
        print(f"\n夹角 = {angle_deg}°:")
        print(f"  v₁ = {v1}")
        print(f"  v₂ = {v2}")
        print(f"  点积 u·v = {dot_product:.2f}")
        print(f"  投影长度 = {proj_length:.2f}")
        
        # ====== 可视化 ======
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # 绘制向量
        ax.quiver(0, 0, v1[0], v1[1], color='blue', scale=5, width=0.005, label='v₁')
        ax.quiver(0, 0, v2[0], v2[1], color='red', scale=5, width=0.005, label='v₂')
        
        # 绘制投影向量 (沿 v₁方向的 v₂分量)
        proj_v1 = (dot_product / np.dot(v1, v1)) * v1
        ax.quiver(0, 0, proj_v1[0], proj_v1[1], color='green', scale=5, width=0.005, 
                  label=f'投影长度={proj_length:.2f}')
        
        # 绘制直角标记 (投影处)
        ax.plot([proj_v1[0], v2[0]], [proj_v1[1], v2[1]], 'k--', alpha=0.3)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f'向量投影关系 (夹角={angle_deg}°)\n点积 = {dot_product:.2f}, 投影长度 = {proj_length:.2f}')
        
        plt.show()
    
    # ====== 测试不同夹角 ======
    for angle in [0, 30, 60, 90, 120, 150, 180]:
        plot_projection(angle)
    
    print("\n🎯 关键洞察:")
    print("   - 夹角=0°时点积最大 → 向量完全对齐")
    print("   - 夹角=90°时点积=0 → 正交 (无投影)")
    print("   - 夹角>90°时点积为负 → 反向分量")


# ============================================================
# 🧪 实验二：矩阵如何推整张网格
# ============================================================
"""
目标：看懂"矩阵 = 空间动作"。

关键公式:
- 行列式 det(A):面积/体积缩放因子
- det>1:放大，0<det<1:缩小，det=0:压塌 (不可逆),det<0:翻转
"""

def experiment_2_matrix_grid_transformation():
    """实验二：矩阵如何推整张网格"""
    
    print("\n" + "=" * 70)
    print("🧪 实验二：矩阵如何推整张网格")
    print("=" * 70)
    
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
        
        # 绘制变换后的网格 (深色)
        for i in range(10):
            ax.plot(X_trans[i, :], Y_trans[i, :], 'b-', linewidth=1.5)
            ax.plot(X_trans[:, i], Y_trans[:, i], 'b-', linewidth=1.5)
        
        # 绘制坐标轴和单位向量变换后的位置
        ax.quiver(0, 0, A[0, 0], A[1, 0], color='red', scale=3, width=0.005, label='e₁→')
        ax.quiver(0, 0, A[0, 1], A[1, 1], color='green', scale=3, width=0.005, label='e₂→')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.grid(alpha=0.3)
        ax.legend()
        det = np.linalg.det(A)
        ax.set_title(f"{title}\ndet(A) = {det:.2f}")
        
        plt.show()
    
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


# ============================================================
# 🧪 实验三：协方差如何定义椭圆
# ============================================================
"""
目标：看懂 Σ、特征值、特征向量之间的几何关系。

关键公式:
- 椭圆方程：xᵀΣ⁻¹x = 1 (二次型)
- Σ = QΛQᵀ，特征值=半轴长度²，特征向量=主轴方向
"""

def experiment_3_covariance_ellipse():
    """实验三：协方差如何定义椭圆"""
    
    print("\n" + "=" * 70)
    print("🧪 实验三：协方差如何定义椭圆")
    print("=" * 70)
    
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
        det = np.linalg.det(Sigma)
        ax.set_title(f"{title}\n特征值：{eigenvalues:.2f}, 行列式：{det:.2f}")
        
        plt.show()
    
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


# ============================================================
# 🧪 实验四：特征方向为什么特殊
# ============================================================
"""
目标：看懂"某些方向只会缩放不会拐弯"。

关键公式:
- Av = λv → 特征向量 v经过 A变换后，只被拉伸/压缩，不转弯!
"""

def experiment_4_eigenvectors():
    """实验四：特征方向为什么特殊"""
    
    print("\n" + "=" * 70)
    print("🧪 实验四：特征方向为什么特殊")
    print("=" * 70)
    
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
            color = 'red' if abs(eigenvalues[i])>1 else 'green'
            ax1.quiver(0, 0, v[0], v[1], color=color, scale=3, width=0.005)
        ax1.set_title(f"输入空间：单位圆\n{title}")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.axis('equal')
        ax1.grid(alpha=0.3)
        
        # 右图：变换后的空间
        ax2.plot(transformed_x, transformed_y, 'g-', linewidth=2, label='变换后椭圆')
        for i in range(len(eigenvalues)):
            v_transformed = A @ eigenvectors[:, i] * 2
            color = 'red' if abs(eigenvalues[i])>1 else 'green'
            ax2.quiver(0, 0, v_transformed[0], v_transformed[1], 
                      color=color, scale=3, width=0.005, 
                      label=f'A·v_{i+1}=λ{eigenvalues[i]:.2f}' if i==0 else "")
        
        ax2.set_title("输出空间：椭圆\n(A·v = λ·v → 只缩放不转弯!)")
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.axis('equal')
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    # ====== 1. 对称矩阵 (实特征值，正交特征向量) ======
    A_sym = np.array([[3.0, 1.0], [1.0, 2.0]])
    print("\n对称矩阵:")
    print(A_sym)
    plot_eigenvectors(A_sym, "λ₁=3.62, λ₂=1.38")
    
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


# ============================================================
# 🧪 实验五：SVD如何把圆变成椭圆
# ============================================================
"""
目标：看懂"先转、再拉、再转"的结构。

关键公式:
- A = UΣVᵀ (三步拆解：Vᵀ旋转→Σ拉伸→U 旋转)
- Avᵢ = σᵢuᵢ (第 i 个输入主轴被映射到第 i 个输出主轴，缩放σᵢ倍)
"""

def experiment_5_svd_three_steps():
    """实验五：SVD如何把圆变成椭圆"""
    
    print("\n" + "=" * 70)
    print("🧪 实验五：SVD如何把圆变成椭圆")
    print("=" * 70)
    
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
        axes[0].quiver(0, 0, v1_t[0]*2, v1_t[1]*2, color='red', scale=3, width=0.005)
        if len(S)>1:
            axes[0].quiver(0, 0, v2_t[0]*2, v2_t[1]*2, color='green', scale=3, width=0.005)
        sigma_text = f"σ₁={S[0]:.2f}, σ₂={S[1] if len(S)>1 else 'N/A'}"
        axes[0].set_title(f"Step 1: Vᵀ\n(输入主轴)\n{sigma_text}")
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].axis('equal')
        axes[0].grid(alpha=0.3)
        
        # Step 2: Σ拉伸 (中间空间)
        axes[1].plot(stretched_x, stretched_y, 'g-', linewidth=2, label='沿主轴拉伸')
        if len(S)>0:
            axes[1].quiver(0, 0, S[0]*2, 0, color='red', scale=3, width=0.005)
        if len(S)>1:
            axes[1].quiver(0, 0, 0, S[1]*2, color='green', scale=3, width=0.005)
        sigma_text = f"Σ\n(拉伸 σ₁={S[0]:.2f}, σ₂={S[1] if len(S)>1 else 'N/A'})"
        axes[1].set_title(f"Step 2:{sigma_text}")
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].axis('equal')
        axes[1].grid(alpha=0.3)
        
        # Step 3: U 旋转 (输出空间)
        u1, u2 = U[:, 0], U[:, 1]
        axes[2].plot(ellipse_x, ellipse_y, 'b-', linewidth=2, label='最终椭圆')
        if len(S)>0:
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
    
    # ====== 测试不同矩阵 ======
    matrices = [
        (np.array([[3.0, 0], [0, 1.0]]), "对角矩阵：U=I, V=I"),
        (np.array([[4.0, 1.5], [1.5, 2.0]]), "对称矩阵：U≠V (主轴旋转)"),
        (np.array([[3.0, 2.0], [1.0, 1.0]]), "非对称：U≠V，输入/输出主轴不同!")
    ]
    
    for A, title in matrices:
        print(f"\n{'='*70}")
        print(f"矩阵:\n{A}")
        plot_svd_three_steps(A, title)
        
        # 数值验证核心关系
        U, S, Vt = np.linalg.svd(A)
        print("\n🔍 验证：A·vᵢ = σᵢ·uᵢ")
        for j in range(len(S)):
            vj = Vt[j]
            uj = U[:, j]
            
            left = A @ vj
            right = S[j] * uj
            
            match = "✅" if np.allclose(left, right) else "❌"
            print(f"  i={j+1}: {match} A·v_{j+1} ≈ σ_{j+1}·u_{j+1}")


# ============================================================
# 🎯 主函数：运行所有实验
# ============================================================

def main():
    """运行所有实验"""
    
    print("\n" + "=" * 70)
    print("🔥 线性代数教程 - 5 大可视化实验")
    print("=" * 70)
    print("\n推荐顺序:")
    print("1. 实验一：向量投影 (理解点积和正交性)")
    print("2. 实验二：矩阵推网格 (理解行列式和变换)")  
    print("3. 实验三：协方差椭圆 (理解特征分解)")
    print("4. 实验四：特征方向特殊性")
    print("5. 实验五：SVD三步拆解")
    print("\n每个实验都对应理论章节，建议跑完后再回顾!")
    
    # 运行所有实验
    experiment_1_vectors_projection()
    experiment_2_matrix_grid_transformation()
    experiment_3_covariance_ellipse()
    experiment_4_eigenvectors()
    experiment_5_svd_three_steps()
    
    print("\n" + "=" * 70)
    print("🎉 所有实验完成!")
    print("=" * 70)
    print("\n下一步:")
    print("1. 对照理论章节 (第 3,5,6,8,9 章) 深入理解")
    print("2. 修改参数，观察现象变化 → 加深直觉")
    print("3. 尝试自己编写类似的可视化代码")


if __name__ == "__main__":
    main()
