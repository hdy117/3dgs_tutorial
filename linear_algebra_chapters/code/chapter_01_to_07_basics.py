"""
第 1-7 章：基础概念实验合集
================================

这个 Notebook 汇总了线性代数最基础的几个概念的实验代码:
- 向量运算 (点积/投影)
- 基变换与坐标
- 矩阵推网格
- 协方差与椭球
- 行列式与秩

对应章节：第 3,4,5,6,7 章
"""

import numpy as np


# ============================================================
# 🧪 实验一：点积与正交性 (第 3 章)
# ============================================================
def experiment_1_dot_product():
    """验证 u·v = ||u|| ||v|| cosθ"""
    
    print("=" * 70)
    print("🧪 实验一：点积与正交性")
    print("=" * 70)
    
    # ====== 测试不同夹角 ======
    angles_deg = [0, 30, 45, 60, 90, 120, 180]
    
    u = np.array([3.0, 1.0])
    v_base = np.array([1.0, 2.0])
    
    print(f"固定向量 u = {u}")
    print("\n夹角 | u·v   | ||u||·||v||·cosθ | 相对误差")
    print("-" * 70)
    
    for angle in angles_deg:
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        v = rotation_matrix @ v_base
        
        dot_product = np.dot(u, v)
        expected = np.linalg.norm(u) * np.linalg.norm(v) * np.cos(theta)
        
        rel_error = abs(dot_product - expected) / max(abs(expected), 1e-10) * 100
        
        print(f"{angle:5d} | {dot_product:6.2f} | {expected:18.2f} | {rel_error:.4f}%")
    
    # ====== 正交性验证 ======
    print("\n" + "-" * 70)
    print("测试：u⊥v ⇔ u·v = 0")
    v_perp = np.array([-2.0, 6.0])  # 与 [3,1] 正交
    
    dot_result = np.dot(u, v_perp)
    print(f"  u=[3,1], v=[-2,6]: u·v={dot_result} (应该=0)")
    
    if abs(dot_result) < 1e-10:
        print("  ✅ 正交性成立!")


# ============================================================
# 🧪 实验二：基变换与坐标 (第 4 章)
# ============================================================
def experiment_2_basis_transformation():
    """验证 x = Bc 和 c' = B⁻¹B_new @ c"""
    
    print("\n" + "=" * 70)
    print("🧪 实验二：基变换与坐标")
    print("=" * 70)
    
    # ====== 场景设置 ======
    print("几何对象不变，但不同基下数字会变!")
    
    point = np.array([1.0, 2.0])
    
    # 原基：标准正交基 e₁=[1,0], e₂=[0,1]
    B_old = np.eye(2)
    c_old = point
    
    print(f"\n向量 p = [1,2]")
    print(f"在标准基下：c = {c_old}")
    
    # 新基：旋转 30°后的基
    theta = np.pi / 6
    B_new = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    print(f"\n新基 B_new:")
    print(B_new)
    
    # 计算新坐标
    c_new = np.linalg.solve(B_new, point)
    
    print(f"在新基下的坐标：c' = {c_new}")
    
    # ====== 验证恢复原向量 ======
    recovered = B_new @ c_new
    
    print(f"\n验证：B_new @ c' = ?= p")
    print(f"  恢复结果：{recovered}")
    print(f"  {'✅ 正确!' if np.allclose(recovered, point) else '❌ 错误!'}")


# ============================================================
# 🧪 实验三：矩阵推网格 (第 5 章)
# ============================================================
def experiment_3_matrix_grid():
    """可视化矩阵变换对单位网格的影响"""
    
    print("\n" + "=" * 70)
    print("🧪 实验三：矩阵推整张网格")
    print("=" * 70)
    
    def plot_grid(A, title):
        """简化版网格可视化 (文本描述)"""
        
        # 创建单位正方形四个角
        corners = np.array([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1]
        ])
        
        transformed = corners @ A.T
        
        print(f"\n{title}")
        print("-" * 70)
        print("原正方形四个角:")
        for i, c in enumerate(corners):
            print(f"  {i+1}: {c}")
        
        print("\n变换后四个角:")
        for i, t in enumerate(transformed):
            print(f"  {i+1}: [{t[0]:.2f}, {t[1]:.2f}]")
    
    # ====== 不同矩阵对比 ======
    matrices = [
        (np.array([[2.0, 0], [0, 0.5]]), "缩放矩阵 A×=[2, 0.5]"),
        (np.array([[0, -1], [1, 0]]), "旋转 90°(逆时针)"),
        (np.array([[1.5, 0.5], [0, 1]]), "剪切矩阵"),
    ]
    
    for A, title in matrices:
        plot_grid(A, title)


# ============================================================
# 🧪 实验四：协方差→椭球 (第 6 章)
# ============================================================
def experiment_4_covariance_to_ellipse():
    """验证协方差矩阵的特征分解 = 椭球主轴"""
    
    print("\n" + "=" * 70)
    print("🧪 实验四：协方差 → 椭球")
    print("=" * 70)
    
    # ====== 构造正定协方差矩阵 ======
    Sigma = np.array([
        [4.0, 1.5],
        [1.5, 2.0]
    ])
    
    print(f"协方差矩阵 Σ:\n{Sigma}")
    
    # ====== 特征分解 ======
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    print("\nStep 1:特征分解")
    print(f"  λ₁={eigenvalues[1]:.3f}, λ₂={eigenvalues[0]:.3f}")
    print(f"  半轴长度：σ₁={np.sqrt(eigenvalues[1]):.3f}, σ₂={np.sqrt(eigenvalues[0]):.3f}")
    
    # ====== 验证二次型 xᵀΣ⁻¹x = 1 ======
    print("\nStep 2:验证椭球表面")
    for i in range(2):
        point = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
        quad_form = point.T @ np.linalg.inv(Sigma) @ point
        
        print(f"  主轴{i+1}上的点：{point}")
        print(f"    xᵀΣ⁻¹x = {quad_form:.4f} (应该=1)")


# ============================================================
# 🧪 实验五：行列式与秩 (第 7 章)
# ============================================================
def experiment_5_determinant_and_rank():
    """验证 det 和 rank 的关系"""
    
    print("\n" + "=" * 70)
    print("🧪 实验五：行列式 vs 秩")
    print("=" * 70)
    
    matrices = [
        (np.array([[3.0, 1], [1, 2]]), "满秩矩阵"),
        (np.array([[2.0, 4], [1, 2]]), "奇异矩阵 (det=0)"),
    ]
    
    for A, name in matrices:
        det = np.linalg.det(A)
        
        # SVD 计算数值秩
        U, S, Vt = np.linalg.svd(A)
        cutoff = max(A.shape) * S[0] * np.finfo(float).eps
        rank = np.sum(S > cutoff)
        
        print(f"\n{name}:")
        print(A)
        print(f"  det(A) = {det:.6f}")
        print(f"  数值秩 = {rank}/{A.shape[0]}")
    
    print("\n🎯 关键结论:")
    print("   - 满秩方阵：rank=n, det≠0")
    print("   - 奇异矩阵：rank<n, det=0")


# ============================================================
# 🧪 实验六：向量投影长度 (第 3-4 章)
# ============================================================
def experiment_6_projection_length():
    """验证 proj_u(v) = (u·v)/||u||² × u"""
    
    print("\n" + "=" * 70)
    print("🧪 实验六：投影长度计算")
    print("=" * 70)
    
    u = np.array([3.0, 1.0])
    v = np.array([1.0, 2.0])
    
    print(f"向量 u = {u}, ||u|| = {np.linalg.norm(u):.4f}")
    print(f"向量 v = {v}, ||v|| = {np.linalg.norm(v):.4f}")
    
    # ====== 投影长度计算 ======
    dot_uv = np.dot(u, v)
    proj_length = dot_uv / np.linalg.norm(u)
    proj_vector = (dot_uv / np.dot(u, u)) * u
    
    print(f"\n点积：u·v = {dot_uv}")
    print(f"投影长度：||proj_u(v)|| = |u·v|/||u|| = {abs(proj_length):.4f}")
    print(f"投影向量：{proj_vector}")
    
    # ====== 验证投影垂直于余量 ======
    residual = v - proj_vector
    ortho_check = np.dot(proj_vector, residual)
    
    print(f"\n验证：投影⊥余量")
    print(f"  proj·residual = {ortho_check:.10f} (应该≈0)")


# ============================================================
# 🎯 主函数：运行所有实验
# ============================================================

def main():
    """运行所有实验"""
    
    print("\n" + "=" * 70)
    print("🔥 第 1-7 章基础概念实验")
    print("=" * 70)
    print("\n这个 Notebook 演示了:")
    print("1. 点积与正交性 (u·v = ||u|| ||v|| cosθ)")
    print("2. 基变换与坐标 (x = Bc)")
    print("3. 矩阵推网格 (空间动作可视化)")
    print("4. 协方差 → 椭球 (特征分解)")
    print("5. 行列式 vs 秩 (体积缩放/独立维度数)")
    print("6. 投影长度计算")
    
    # 运行所有实验
    experiment_1_dot_product()
    experiment_2_basis_transformation()
    experiment_3_matrix_grid()
    experiment_4_covariance_to_ellipse()
    experiment_5_determinant_and_rank()
    experiment_6_projection_length()
    
    print("\n" + "=" * 70)
    print("🎉 所有实验完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
