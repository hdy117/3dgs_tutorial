"""
第 12 章：最容易混淆的五个点 - Python 验证代码
==================================================

这个 Notebook 完整演示了线性代数学习中最常见的 5 个误解，并通过数值计算澄清它们。

对应章节：第 12 章
"""

import numpy as np


# ============================================================
# ❌ 误解 #1: 线性 vs 仿射变换
# ============================================================
def misconception_1_linear_vs_affine():
    """误解#1:"线性"不是"图像看起来是一条直线" """
    
    print("=" * 70)
    print("❌ 误解#1: 线性 vs 仿射变换")
    print("=" * 70)
    print("\n问题：y = 2x + 3 是线性的吗？")
    
    # ====== 定义两个"变换" ======
    def linear_transform(x):
        """真正的线性变换：y = 2x"""
        return 2 * x
    
    def affine_transform(x):
        """仿射变换 (看起来是直线，但不是线性的)：y = 2x + 3"""
        return 2 * x + 3
    
    # ====== 测试 1:保原点性质 T(0) = 0 ======
    print("\n" + "-" * 70)
    print("测试 1:零向量映射")
    x_zero = np.array([0.0])
    
    linear_result = linear_transform(x_zero)
    affine_result = affine_transform(x_zero)
    
    print(f"  输入：x = {x_zero}")
    print(f"  线性变换 y = 2x → {linear_result} (✅ T(0)=0)")
    print(f"  仿射变换 y = 2x+3 → {affine_result} (❌ T(0)≠0,不是线性的!)")
    
    # ====== 测试 2:叠加性 ======
    print("\n" + "-" * 70)
    print("测试 2:叠加性验证 T(ax+by) = aT(x) + bT(y)")
    x1, x2 = np.array([1.0]), np.array([2.0])
    a, b = 3.0, -1.0
    
    # 线性变换的验证
    left_linear = linear_transform(a * x1 + b * x2)
    right_linear = a * linear_transform(x1) + b * linear_transform(x2)
    
    print(f"\n  对于线性变换 y=2x:")
    print(f"    T(3×[1] - 1×[2]) = {left_linear}")
    print(f"    3×T([1]) - 1×T([2]) = {right_linear}")
    print(f"    {'✅ 相等!' if np.isclose(left_linear, right_linear) else '❌ 不相等!'}")
    
    # 仿射变换的验证
    left_affine = affine_transform(a * x1 + b * x2)
    right_affine = a * affine_transform(x1) + b * affine_transform(x2)
    
    print(f"\n  对于仿射变换 y=2x+3:")
    print(f"    T(3×[1] - 1×[2]) = {left_affine}")
    print(f"    3×T([1]) - 1×T([2]) = {right_affine}")
    print(f"    {'✅ 相等!' if np.isclose(left_affine, right_affine) else '❌ 不相等! (正常，仿射变换不满足叠加性)'}")
    
    # ====== 矩阵形式的验证 ======
    print("\n" + "-" * 70)
    print("测试 3:矩阵形式 T(x)=Ax vs T(x)=Ax+b")
    
    A = np.array([[2.0, 1.0], [1.0, 3.0]])  # 线性变换矩阵
    b_vec = np.array([1.0, 0.5])  # 平移向量 (仿射部分)
    
    x1 = np.array([1.0, 2.0])
    x2 = np.array([3.0, -1.0])
    
    def linear_matrix_transform(x):
        return A @ x
    
    def affine_matrix_transform(x):
        return A @ x + b_vec
    
    # 验证叠加性
    a, b = 2.0, -1.0
    
    linear_left = linear_matrix_transform(a * x1 + b * x2)
    linear_right = a * linear_matrix_transform(x1) + b * linear_matrix_transform(x2)
    
    affine_left = affine_matrix_transform(a * x1 + b * x2)
    affine_right = a * affine_matrix_transform(x1) + b * affine_matrix_transform(x2)
    
    print(f"\n  线性变换 T(x)=Ax:")
    print(f"    {'✅ 满足叠加性' if np.allclose(linear_left, linear_right) else '❌ 不满足'}")
    
    print(f"\n  仿射变换 T(x)=Ax+b:")
    print(f"    {'✅ 满足叠加性' if np.allclose(affine_left, affine_right) else '❌ 不满足 (正常!)'}")
    
    # ====== 结论 ======
    print("\n" + "=" * 70)
    print("🎯 关键洞察:")
    print("=" * 70)
    print("   - **线性变换必须保原点** (`T(0)=0`) ✅")
    print("   - **仿射变换 = 线性 + 平移**,虽然图像是"直线",但不是线性的 ⚠️")


# ============================================================
# ❌ 误解 #2: 主动 vs 被动变换
# ============================================================
def misconception_2_active_vs_passive():
    """误解#2:"坐标变了"不一定表示"物体动了" """
    
    print("\n" + "=" * 70)
    print("❌ 误解#2: 主动 vs 被动变换")
    print("=" * 70)
    print("\n问题：R @ x 和 R.T @ x 有什么区别?")
    
    # ====== 场景设置 ======
    point_world = np.array([1.0, 2.0, 3.0])  # 世界空间中的点
    
    # 相机绕 y 轴旋转 30°
    theta = np.pi / 6
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    print(f"世界空间点：{point_world}")
    
    # ====== 方式 A:主动变换 (点旋转) ======
    point_rotated = R_y @ point_world
    
    print("\n" + "-" * 70)
    print("方式 A:主动变换 (点旋转)")
    print(f"  公式：x_rotated = R @ x")
    print(f"  结果：{point_rotated}")
    
    # ====== 方式 B:被动变换 (坐标系旋转) ======
    point_in_camera = R_y.T @ point_world
    
    print("\n" + "-" * 70)
    print("方式 B:被动变换 (坐标系旋转)")
    print(f"  公式：x_camera = R.T @ x")
    print(f"  结果：{point_in_camera}")
    
    # ====== 验证两者等价 ======
    print("\n" + "=" * 70)
    print("🔍 关键洞察:")
    print("=" * 70)
    print("   - 主动变换：点在世界空间中绕 y 轴旋转了 30°")
    print("   - 被动变换：相机绕 y 轴逆时针旋转 30°,点在相机空间中的坐标变了")
    print("\n   **两者是同一几何事实的不同视角!**")


# ============================================================
# ❌ 误解 #3: 协方差=椭球编码
# ============================================================
def misconception_3_covariance_as_ellipse():
    """误解#3:协方差不只是"噪声表",它是椭球编码!"""
    
    print("\n" + "=" * 70)
    print("❌ 误解#3: 协方差=椭球编码")
    print("=" * 70)
    print("\n问题：Σ 到底编码了什么几何信息?")
    
    # ====== 构造一个协方差矩阵 ======
    Sigma = np.array([
        [4.0, 1.5],
        [1.5, 2.0]
    ])
    
    print(f"协方差矩阵:\n{Sigma}")
    
    # ====== Step 1:特征分解 ======
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    print("\nStep 1:特征分解")
    print(f"  特征值 (方差): {eigenvalues}")
    print(f"  特征向量 (主轴方向):\n{eigenvectors}")
    
    # ====== Step 2:提取椭球参数 ======
    stds = np.sqrt(eigenvalues)  # 半轴长度 (标准差)
    angles = np.degrees(np.arctan2(eigenvectors[1, :], eigenvectors[0, :]))
    
    print(f"\nStep 2:椭球参数")
    print(f"  半轴长度 σ₁={stds[0]:.3f}, σ₂={stds[1]:.3f}")
    print(f"  主轴角度：{angles[0]:.2f}°, {angles[1]:.2f}°")
    
    # ====== 验证二次型 ======
    print("\n" + "-" * 70)
    print("Step 3:验证椭球方程 xᵀΣ⁻¹x = 1")
    
    for i in range(2):
        point = eigenvectors[:, i] * stds[i]  # 沿主轴走到半轴位置
        quad_form = point.T @ np.linalg.inv(Sigma) @ point
        
        print(f"\n  主轴{i+1}上的点 (距离={stds[i]:.3f}):")
        print(f"    x = {point}")
        print(f"    xᵀΣ⁻¹x = {quad_form:.6f} (应该=1)")
    
    # ====== 结论 ======
    print("\n" + "=" * 70)
    print("🎯 关键洞察:")
    print("=" * 70)
    print("   - **统计视角**:协方差 = '噪声表' ✅")
    print("   - **几何视角**:协方差 = '椭球编码' (QΛQᵀ) ✅(3DGS的核心!)")


# ============================================================
# ❌ 误解 #4: 投影的局部线性化
# ============================================================
def misconception_4_projection_local_linearization():
    """误解#4:投影不是线性的，但局部可以线性化"""
    
    print("\n" + "=" * 70)
    print("❌ 误解#4: 透视投影局部线性化")
    print("=" * 70)
    print("\n问题：为什么 J @ delta_x 能近似非线性变换?")
    
    # ====== 透视投影函数 ======
    def perspective_projection(x, y, z, fx, fy):
        """从相机空间到像素坐标的透视投影"""
        u = fx * x / z + 640.0
        v = fy * y / z + 360.0
        return np.array([u, v])
    
    cx, cy = 640.0, 360.0
    fx, fy = 800.0, 800.0
    
    # ====== 定义投影中心点 ======
    x0, y0, z0 = 1.0, 0.5, 2.0
    
    print(f"投影中心点：({x0}, {y0}, {z0})")
    
    # ====== 计算 Jacobian ======
    J = np.array([
        [fx / z0,     0.0,   -fx * x0 / (z0**2)],
        [0.0,       fy / z0, -fy * y0 / (z0**2)]
    ])
    
    print(f"\n投影函数在中心点的 Jacobian ({J.shape}):")
    print(J)
    
    # ====== 测试不同大小的扰动 ======
    def test_linear_approx(dx, dy, dz):
        """测试线性近似的精度"""
        
        # 精确值 (非线性投影)
        p1 = perspective_projection(x0 + dx, y0 + dy, z0 + dz, fx, fy)
        p0 = perspective_projection(x0, y0, z0, fx, fy)
        delta_exact = p1 - p0
        
        # 线性近似 (用 Jacobian)
        delta_linear = J @ np.array([dx, dy, dz])[:2]
        
        # 误差
        error = np.linalg.norm(delta_exact - delta_linear) / max(np.linalg.norm(delta_exact), 1e-10) * 100
        
        return delta_exact, delta_linear, error
    
    print("\n" + "-" * 70)
    print("测试不同扰动大小的线性近似精度:")
    
    # 小扰动
    dx_small, dy_small, dz_small = 0.01, 0.005, 0.02
    exact_small, linear_small, error_small = test_linear_approx(dx_small, dy_small, dz_small)
    
    print(f"\n小扰动 (Δx=0.01, Δy=0.005, Δz=0.02):")
    print(f"  精确投影变化：{exact_small}")
    print(f"  线性近似变化：{linear_small}")
    print(f"  相对误差：{error_small:.4f}% (✅ 非常好!)")
    
    # 中等扰动
    dx_med, dy_med, dz_med = 0.1, 0.05, 0.2
    exact_med, linear_med, error_med = test_linear_approx(dx_med, dy_med, dz_med)
    
    print(f"\n中等扰动 (Δx=0.1, Δy=0.05, Δz=0.2):")
    print(f"  精确投影变化：{exact_med}")
    print(f"  线性近似变化：{linear_med}")
    print(f"  相对误差：{error_med:.4f}% (⚠️ 开始变大)")
    
    # ====== 结论 ======
    print("\n" + "=" * 70)
    print("🎯 关键洞察:")
    print("=" * 70)
    print("   - **全局透视投影是非线性的** ⚠️")
    print("   - **局部可以用 Jacobian 线性化** ✅(3DGS的核心假设!)")


# ============================================================
# ❌ 误解 #5: 秩 vs 行列式
# ============================================================
def misconception_5_rank_vs_determinant():
    """误解#5:行列式和秩不是两个孤立概念!"""
    
    print("\n" + "=" * 70)
    print("❌ 误解#5: 秩 vs 行列式")
    print("=" * 70)
    print("\n问题：这两个概念有什么联系?")
    
    # ====== 定义三个矩阵 ======
    A_full_rank = np.array([
        [3.0, 1.5],
        [1.0, 2.0]
    ])
    
    A_singular = np.array([
        [2.0, 4.0],
        [1.0, 2.0]  # 第 2 行 = 0.5×第 1 行 → 线性相关!
    ])
    
    print("=" * 70)
    print("分析三个矩阵:")
    print("=" * 70)
    
    def analyze_matrix(A, name):
        """分析矩阵的秩和行列式"""
        
        print(f"\n{name}:")
        print(A)
        
        # 1.计算行列式 (仅方阵)
        if A.shape[0] == A.shape[1]:
            det = np.linalg.det(A)
            print(f"  行列式：{det:.6f}")
            
            if abs(det) < 1e-10:
                print(f"  ⚠️ 行列式为 0 → 奇异 (不可逆)")
            else:
                print(f"  ✅ 行列式非零 → 满秩")
        
        # 2.计算数值秩
        U, S, Vt = np.linalg.svd(A)
        cutoff = max(A.shape) * S[0] * np.finfo(float).eps
        numerical_rank = np.sum(S > cutoff)
        
        print(f"  奇异值：{S}")
        print(f"  数值秩：{numerical_rank}/{min(A.shape)}")
        
        # 3.判断是否满秩
        if A.shape[0] == A.shape[1]:
            is_full_rank = numerical_rank == A.shape[0]
            print(f"  {'✅ 满秩' if is_full_rank else '❌ 不满秩'}")
    
    analyze_matrix(A_full_rank, "A: 满秩矩阵 (2×2)")
    analyze_matrix(A_singular, "B: 奇异矩阵 (行列式=0, 秩=1)")
    
    # ====== 结论 ======
    print("\n" + "=" * 70)
    print("🎯 关键洞察:")
    print("=" * 70)
    print("""
   秩和行列式的关系:

   1. **行列式** = "变换后的体积缩放因子"
      - det(A) = 0 → 体积变为 0 → 空间被压塌!

   2. **秩** = "保留的独立维度数"  
      - rank < n → 有维度被压没了!

   3. **联系**:
      - 满秩方阵 (rank=n) → det ≠ 0
      - 不满秩方阵 (rank<n) → det = 0
""")


# ============================================================
# 🎯 主函数：运行所有误解验证
# ============================================================

def main():
    """运行所有实验"""
    
    print("\n" + "=" * 70)
    print("🔥 第 12 章：最容易混淆的五个点 - Python 验证")
    print("=" * 70)
    print("\n这个 Notebook 演示了:")
    print("1. 线性 vs 仿射变换 (保原点性质)")
    print("2. 主动 vs 被动变换 (同一事实不同视角)")
    print("3. 协方差=椭球编码 (特征分解→半轴长度/角度)")
    print("4. 投影局部线性化 (Jacobian 精度测试)")
    print("5. 秩 vs 行列式 (都在问：有没有压塌空间?)")
    
    # 运行所有误解验证
    misconception_1_linear_vs_affine()
    misconception_2_active_vs_passive()
    misconception_3_covariance_as_ellipse()
    misconception_4_projection_local_linearization()
    misconception_5_rank_vs_determinant()
    
    print("\n" + "=" * 70)
    print("🎉 所有误解验证完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
