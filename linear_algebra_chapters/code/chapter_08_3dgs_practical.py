"""
第 11 章：回到 3DGS - 这些数学到底在代码里干了什么
=======================================================

这个 Notebook 演示了线性代数概念如何在 3DGS 中实际应用。
重点理解：协方差传播、Jacobian 局部线性化、像素权重计算。

对应章节：第 11 章，也涉及第 6,8,9,12 章内容
"""

import numpy as np


# ============================================================
# 🔥 实验一：世界→相机空间变换
# ============================================================
def experiment_1_world_to_camera():
    """实验一：世界空间→相机空间的完整流程"""
    
    print("=" * 70)
    print("🔥 实验一：世界空间 → 相机空间")
    print("=" * 70)
    
    # ====== 场景设置 ======
    mu_world = np.array([1.0, 2.0, 3.0])  # 高斯中心
    
    Sigma_world = np.array([
        [4.0, 1.5, 0.8],
        [1.5, 3.0, 1.2],
        [0.8, 1.2, 2.0]
    ])  # 世界空间协方差
    
    # 相机外参 (旋转 + 平移)
    R = np.array([
        [0.99, -0.03, 0.14],   # 近似绕 y 轴旋转 8°
        [0.03, 0.998, -0.06],
        [-0.14, 0.06, 0.99]
    ])
    t = np.array([-1.0, -2.0, -5.0])
    
    print(f"世界空间中心：{mu_world}")
    print(f"世界空间协方差:\n{Sigma_world}")
    
    # ====== Step 1:位置变换 (仿射变换) ======
    mu_cam = R @ mu_world + t
    
    print(f"\nStep 1a: 相机空间中心:")
    print(f"  μ_cam = R @ μ_world + t")
    print(f"  {mu_cam}")
    
    # ====== Step 2:协方差传播 (二次型) ======
    Sigma_cam = R @ Sigma_world @ R.T
    
    print(f"\nStep 1b: 相机空间协方差:")
    print(f"  Σ_cam = R @ Σ_world @ R.T")
    print(f"  ✅ 对称吗？{np.allclose(Sigma_cam, Sigma_cam.T)}")
    
    # ====== Step 3:验证特征分解的正确性 ======
    eigenvalues_world, eigenvectors_world = np.linalg.eigh(Sigma_world)
    eigenvalues_cam, eigenvectors_cam = np.linalg.eigh(Sigma_cam)
    
    print(f"\n世界空间特征值：{eigenvalues_world}")
    print(f"相机空间特征值：{eigenvalues_cam}")
    print(f"✅ 特征值相同？(只旋转不缩放){np.allclose(eigenvalues_world, eigenvalues_cam)}")
    
    # ====== Step 4:验证主轴方向被正确旋转 ======
    v1 = eigenvectors_world[:, 0]
    u1 = eigenvectors_cam[:, 0]
    
    print(f"\n世界空间主轴 v₁ = {v1}")
    print(f"相机空间主轴 u₁ = {u1}")
    print(f"验证：R @ v₁ ≈ u₁ ?")
    print(f"      R @ v₁ = {R @ v1}")
    
    if np.allclose(R @ v1, u1):
        print("✅ 正确!")
    else:
        print("❌ 错误！")
    
    print("\n🎯 关键理解:")
    print("   - 位置用仿射变换：μ_cam = R μ_world + t")
    print("   - 协方差用二次型传播：Σ_cam = R Σ_world R.T (旋转椭球)")
    print("   - 特征值不变，只是主轴方向被旋转到相机基!")


# ============================================================
# 🔥 实验二：投影到屏幕空间
# ============================================================
def experiment_2_projection_to_screen():
    """实验二：相机空间→屏幕空间的投影"""
    
    print("\n" + "=" * 70)
    print("🔥 实验二：相机空间 → 屏幕空间 (透视投影)")
    print("=" * 70)
    
    # ====== 从上一个实验延续 ======
    mu_cam = np.array([-1.87, -1.21, -1.52])  # 假设结果
    
    # 相机内参
    fx, fy = 800.0, 800.0  # 焦距
    cx, cy = 640.0, 360.0  # 主点
    
    X, Y, Z = mu_cam
    
    print(f"相机空间中心：({X:.2f}, {Y:.2f}, {Z:.2f})")
    print(f"相机内参：fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # ====== Step 1:投影到像素坐标 ======
    u_pixel = fx * X / Z + cx
    v_pixel = fy * Y / Z + cy
    
    print(f"\n屏幕中心：({u_pixel:.1f}, {v_pixel:.1f})")
    
    # ====== Step 2:计算 Jacobian (局部线性化) ======
    J = np.array([
        [fx / Z,     0.0,   -fx * X / (Z**2)],
        [0.0,       fy / Z, -fy * Y / (Z**2)]
    ])
    
    print(f"\n投影 Jacobian J ({J.shape}):")
    print(J)
    print("\n每个元素的意义:")
    print("  J[0,0] = fx/Z  → x 方向缩放因子")
    print("  J[0,2] = -fx*X/Z² → 深度变化对 u 的影响 (透视效应)")
    
    # ====== Step 3:协方差传播到屏幕空间 ======
    Sigma_cam_2x2 = np.array([[0.02, 0.00], [0.00, 0.01]])  # 简化示例
    
    Sigma_2d = J[:, :2] @ Sigma_cam_2x2 @ J[:, :2].T
    
    print(f"\n屏幕空间协方差 Σ_2d:")
    print(Sigma_2d)
    print(f"✅ 对称吗？{np.allclose(Sigma_2d, Sigma_2d.T)}")
    
    # ====== Step 4:提取椭圆特征 ======
    eigenvalues_2d, eigenvectors_2d = np.linalg.eigh(Sigma_2d)
    axis_lengths = np.sqrt(eigenvalues_2d)
    angle_deg = np.degrees(np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0]))
    
    print(f"\n椭圆特征:")
    print(f"  半轴长度：{axis_lengths:.3f} pixels")
    print(f"  主轴角度：{angle_deg:.1f}° (从 x 轴逆时针)")


# ============================================================
# 🔥 实验三：完整的 3D→2D转换函数
# ============================================================
def experiment_3_gaussian_to_screen_footprint():
    """实验三：高斯参数到屏幕 footprint 的完整转换"""
    
    print("\n" + "=" * 70)
    print("🔥 实验三：完整的 3D→2D 转换")
    print("=" * 70)
    
    def gaussian_to_screen_footprint(mu_world, Sigma_world, R_cam, t_cam, fx, fy):
        """将 3D 高斯转换为 2D 屏幕 footprint"""
        
        # Step 1:世界→相机空间 (仿射变换)
        mu_cam = R_cam @ mu_world + t_cam
        Sigma_cam = R_cam @ Sigma_world @ R_cam.T
        
        # Step 2:提取 XYZ 和计算 Jacobian
        X, Y, Z = mu_cam
        if Z <= 1e-6:
            raise ValueError(f"深度 Z={Z}过小，无法投影!")
        
        J = np.array([
            [fx / Z,     0.0,   -fx * X / (Z**2)],
            [0.0,       fy / Z, -fy * Y / (Z**2)]
        ])
        
        # Step 3:协方差传播到屏幕空间 Σ_2d = J @ Σ_cam[:2,:2] @ J.T
        Sigma_2d = J[:, :2] @ Sigma_cam[:2, :2] @ J[:, :2].T
        
        # Step 4:计算屏幕中心
        mu_2d = np.array([fx * X / Z, fy * Y / Z])
        
        return mu_2d, Sigma_2d
    
    # ====== 测试 ======
    # 创建测试高斯
    mu_world = np.array([1.0, 2.0, 3.0])
    Sigma_world = np.array([[4.0, 1.5, 0.8], [1.5, 3.0, 1.2], [0.8, 1.2, 2.0]])
    
    # 相机外参 (绕 y 轴旋转 30°)
    theta = np.pi / 6
    R_cam = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t_cam = np.array([-2.0, -3.0, -5.0])
    
    # 相机内参
    fx, fy = 800.0, 800.0
    
    # 转换
    mu_2d, Sigma_2d = gaussian_to_screen_footprint(mu_world, Sigma_world, R_cam, t_cam, fx, fy)
    
    print(f"\n屏幕中心：{mu_2d}")
    print(f"屏幕协方差:\n{Sigma_2d}")
    
    # 验证：椭圆半轴长度
    eigenvalues_2d = np.linalg.eigvalsh(Sigma_2d)
    axis_lengths = np.sqrt(eigenvalues_2d)
    print(f"\n✅ 椭圆半轴长度：{axis_lengths:.3f} pixels")


# ============================================================
# 🔥 实验四：调试案例库 - 协方差合法性检查
# ============================================================
def experiment_4_debug_covariance():
    """实验四：协方差矩阵的常见错误和修复"""
    
    print("\n" + "=" * 70)
    print("🔥 实验四：调试案例库")
    print("=" * 70)
    
    def diagnose_covariance(Sigma, name="Sigma"):
        """诊断协方差矩阵的合法性"""
        
        print(f"\n🔍 诊断 {name}:")
        print("-" * 60)
        
        # 1.检查形状
        if len(Sigma.shape) != 2 or Sigma.shape[0] != Sigma.shape[1]:
            print("❌ 不是方阵!")
            return False
        
        n = Sigma.shape[0]
        print(f"✅ 是 {n}×{n} 方阵")
        
        # 2.检查对称性
        max_asymm = np.max(np.abs(Sigma - Sigma.T))
        if max_asymm > 1e-10:
            print(f"⚠️ 不对称 (最大差异：{max_asymm:.2e})")
            Sigma_sym = (Sigma + Sigma.T) / 2
            print("   → 已对称化")
        else:
            print("✅ 对称")
        
        # 3.检查特征值
        eigenvalues = np.linalg.eigvalsh(Sigma)
        min_eig = eigenvalues.min()
        
        if min_eig < -1e-8:
            print(f"❌ 有显著负特征值：{min_eig:.2e}")
            return False
        elif min_eig < 0:
            print(f"⚠️ 微小负特征值 (数值误差): {min_eig:.2e}")
            eigenvalues_safe = np.maximum(eigenvalues, 1e-10)
        else:
            print(f"✅ 半正定 (最小特征值：{min_eig:.2f})")
        
        # 4.条件数
        cond = eigenvalues.max() / max(eigenvalues.min(), 1e-30)
        if cond > 1e6:
            print(f"⚠️ 病态 (条件数：{cond:.2e})")
        elif cond > 100:
            print(f"⚠️ 中等病态 (条件数：{cond:.2f})")
        else:
            print(f"✅ 良态 (条件数：{cond:.2f})")
        
        # 5.秩估计
        cutoff = max(Sigma.shape) * eigenvalues.max() * np.finfo(float).eps
        numerical_rank = np.sum(eigenvalues > cutoff)
        print(f"📊 数值秩：{numerical_rank}/{n}")
        
        return min_eig >= -1e-8
    
    # ====== 测试良态协方差 ======
    Sigma_good = np.array([[4.0, 1.5], [1.5, 2.0]])
    diagnose_covariance(Sigma_good, "良态协方差")
    
    # ====== 测试病态协方差 ======
    Sigma_bad = np.array([[1.0, 0.999], [1.0, 1.0]])
    diagnose_covariance(Sigma_bad, "病态协方差")
    
    print("\n🎯 关键洞察:")
    print("   - 对称化：(Σ + Σᵀ)/2")
    print("   - 负特征值处理：max(eigenvalues, 1e-10)")
    print("   - 条件数>10³ → 谨慎使用")


# ============================================================
# 🎯 主函数：运行所有实验
# ============================================================

def main():
    """运行所有实验"""
    
    print("\n" + "=" * 70)
    print("🔥 3DGS实战应用 - 线性代数在代码中的角色")
    print("=" * 70)
    print("\n这个 Notebook 演示了:")
    print("1. 协方差从世界到相机空间的传播 (RΣR.T)")
    print("2. Jacobian 局部线性化 (J = ∂(proj)/∂x)")
    print("3. 屏幕 footprint 计算 (Σ_2d = J @ Σ_cam @ J.T)")
    print("4. 常见错误调试技巧")
    
    # 运行所有实验
    experiment_1_world_to_camera()
    experiment_2_projection_to_screen()
    experiment_3_gaussian_to_screen_footprint()
    experiment_4_debug_covariance()
    
    print("\n" + "=" * 70)
    print("🎉 所有实验完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
