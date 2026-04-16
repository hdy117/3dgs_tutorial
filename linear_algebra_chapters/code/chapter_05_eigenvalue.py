"""
第 8 章：特征值与特征向量 - 变换最自然的方向
==================================================

这个 Notebook 演示了特征分解的核心概念和应用。
理解"哪些方向只缩放不转弯"，以及为什么对称矩阵有正交特征向量!

对应章节：第 8 章
"""

import numpy as np


# ============================================================
# 🧪 实验一：验证 Av = λv(只缩放不转弯)
# ============================================================
def experiment_1_av_equals_lambda_v():
    """实验一：验证特征向量的核心性质"""
    
    print("=" * 70)
    print("🧪 实验一：Av = λv 验证")
    print("=" * 70)
    
    # 构造一个对称矩阵 (保证有实数特征值)
    A = np.array([
        [3.0, 1.5],
        [1.5, 2.0]
    ])
    
    print(f"矩阵 A:\n{A}")
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    print(f"\n特征值：λ₁={eigenvalues[1]:.4f}, λ₂={eigenvalues[0]:.4f}(从大到小)")
    
    # 验证核心关系
    for i in range(2):
        v = eigenvectors[:, i]
        
        print(f"\n第{i+1}个特征对:")
        print(f"  λ_{i+1} = {eigenvalues[i]:.4f}")
        print(f"  v_{i+1} = [{v[0]:.4f}, {v[1]:.4f}]")
        
        # 计算 Av 和 λv
        Av = A @ v
        lambda_v = eigenvalues[i] * v
        
        print(f"\n  验证：A · v = ?= λ · v")
        print(f"    A · v = {Av}")
        print(f"    λ · v = {lambda_v}")
        
        if np.allclose(Av, lambda_v):
            print(f"  ✅ Av = λv 成立!")
        else:
            print(f"  ❌ 验证失败!")


# ============================================================
# 🧪 实验二：特征分解 A = VΛVᵀ(对称矩阵)
# ============================================================
def experiment_2_eigen_decomposition():
    """实验二：特征分解 A = VΛVᵀ"""
    
    print("\n" + "=" * 70)
    print("🧪 实验二：特征分解 A = VΛVᵀ")
    print("=" * 70)
    
    # 构造对称矩阵
    np.random.seed(42)
    B = np.random.randn(3, 3)
    A = B @ B.T  # 保证半正定
    
    print(f"对称矩阵 A (3×3):\n{A}")
    print(f"✅ 对称吗？{np.allclose(A, A.T)}")
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    V = eigenvectors
    
    # 从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    print(f"\n特征值 (λ₁ ≥ λ₂ ≥ λ₃): {eigenvalues}")
    
    # 验证 A = VΛVᵀ
    Lambda_diag = np.diag(eigenvalues)
    reconstructed = V @ Lambda_diag @ V.T
    
    error = np.linalg.norm(A - reconstructed, 'fro')
    
    print(f"\n验证：A ≈ VΛVᵀ")
    print(f"Frobenius 误差：{error:.2e}")
    
    if error < 1e-10:
        print("✅ 完美匹配!")
    else:
        print("⚠️ 有数值误差 (正常)")
    
    # ====== 验证特征向量正交性 ======
    ortho_check = V.T @ V
    identity_check = np.allclose(ortho_check, np.eye(3))
    
    print(f"\n✅ 特征向量正交吗？{identity_check}")
    if not identity_check:
        print("最大非对角元:", np.max(np.abs(ortho_check - np.eye(3))))


# ============================================================
# 🧪 实验三：特征值与矩阵幂运算
# ============================================================
def experiment_3_eigenvalues_and_matrix_powers():
    """实验三：特征值在矩阵幂运算中的应用"""
    
    print("\n" + "=" * 70)
    print("🧪 实验三：矩阵幂 Aᵏ")
    print("=" * 70)
    
    # 构造一个对角占优的对称矩阵
    A = np.array([
        [3.0, 1.0],
        [1.0, 2.0]
    ])
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"矩阵 A:\n{A}")
    print(f"\n特征值：λ₁={eigenvalues[0]:.4f}, λ₂={eigenvalues[1]:.4f}")
    
    # 计算 A¹, A², ..., A⁵两种方式对比
    print("\n" + "-" * 70)
    print("k | Aᵏ (直接计算)          | Aᵏ = VΛᵏV⁻¹        | 误差")
    print("-" * 70)
    
    for k in [1, 2, 3, 5]:
        # 方法 1:直接计算
        Ak_direct = np.linalg.matrix_power(A, k)
        
        # 方法 2:通过特征分解
        Lambda_k = np.diag(eigenvalues**k)
        Ak_eigen = eigenvectors @ Lambda_k @ np.linalg.inv(eigenvectors)
        
        error = np.max(np.abs(Ak_direct - Ak_eigen))
        
        print(f"{k} | {Ak_direct.flatten().astype(int)} | [{Ak_eigen[0,0]:.1f}, {Ak_eigen[1,1]:.1f}]   | {error:.2e}")
    
    print("\n🎯 关键洞察:")
    print("   - Aᵏ = V Λᵏ V⁻¹，所以特征值直接变成 λᵏ")
    print("   - 如果|λ|<1, Aᵏ会趋于零矩阵!")


# ============================================================
# 🧪 实验四：迭代法求主特征值 (Power Iteration)
# ============================================================
def experiment_4_power_iteration():
    """实验四：用幂迭代法求最大特征值和主特征向量"""
    
    print("\n" + "=" * 70)
    print("🧪 实验四：幂迭代法求主特征值")
    print("=" * 70)
    
    # 构造一个矩阵，确保有一个主导的特征值
    A = np.array([
        [5.0, 1.0, 0],
        [1.0, 3.0, 1.0],
        [0, 1.0, 2.0]
    ])
    
    true_evals, true_evecs = np.linalg.eig(A)
    true_max_eigval = max(true_evals.real)
    true_max_evec = true_evecs[:, np.argmax(true_evals.real)].real
    
    print(f"矩阵 A:\n{A}")
    print(f"\n主特征值 (真值): {true_max_eigval:.4f}")
    
    # ====== 幂迭代法 ======
    b0 = np.random.randn(3)
    b = b0 / np.linalg.norm(b0)
    
    print("\n" + "-" * 70)
    print("迭代 | 当前向量 bᵏ (前两个分量)    | Rayleigh 商 λ≈bᵀAb / bᵀb")
    print("-" * 70)
    
    for k in range(1, 21):
        # 迭代：b_{k+1} = A b_k / ||A b_k||
        Ab = A @ b
        b_new = Ab / np.linalg.norm(Ab)
        
        # Rayleigh 商估计特征值
        lambda_est = (b.T @ A @ b) / (b.T @ b)
        
        if k % 5 == 0 or k <= 3:
            print(f"{k:4d} | [{b_new[0]:6.4f}, {b_new[1]:6.4f}]    | {lambda_est:.6f}")
        
        b = b_new
    
    # ====== 结果对比 ======
    final_lambda = (b.T @ A @ b) / (b.T @ b)
    
    print("\n" + "=" * 70)
    print("🔍 最终结果:")
    print("=" * 70)
    print(f"主特征值估计：{final_lambda:.6f}")
    print(f"真值：        {true_max_eigval:.6f}")
    print(f"绝对误差：    {abs(final_lambda - true_max_eigval):.2e}")
    
    if abs(final_lambda - true_max_eigval) < 1e-4:
        print("✅ 成功!")
    else:
        print("❌ 收敛失败")


# ============================================================
# 🧪 实验五：对称 vs 非对称矩阵的特征值性质对比
# ============================================================
def experiment_5_symmetric_vs_nonsymmetric():
    """实验五：对称 vs 非对称矩阵的特征值对比"""
    
    print("\n" + "=" * 70)
    print("🧪 实验五：对称 vs 非对称")
    print("=" * 70)
    
    # 构造对称矩阵
    B = np.random.randn(3, 3)
    A_symmetric = B @ B.T
    
    # 构造非对称矩阵 (同样尺寸)
    A_nonsymmetric = np.random.randn(3, 3)
    
    print(f"对称矩阵:\n{A_symmetric}")
    sym_evals, sym_evecs = np.linalg.eig(A_symmetric)
    
    print(f"\n对称矩阵特征值：{sym_evals.real}")
    print(f"✅ 都是实数吗？{np.allclose(sym_evals.imag, 0)}")
    
    # 验证正交性
    ortho_check = sym_evecs.conj().T @ sym_evecs
    is_orthonormal = np.allclose(ortho_check, np.eye(3))
    print(f"✅ 特征向量正交归一吗？{is_orthonormal}")
    
    print("\n非对称矩阵:\n{A_nonsymmetric}")
    nonsym_evals, nonsym_evecs = np.linalg.eig(A_nonsymmetric)
    
    print(f"\n非对称矩阵特征值：")
    for i, ev in enumerate(nonsym_evals):
        if abs(ev.imag) > 1e-6:
            print(f"  λ{i+1} = {ev.real:.4f} + {ev.imag:.4f}i (复数!)")
        else:
            print(f"  λ{i+1} = {ev.real:.4f} (实数)")
    
    print("\n🎯 关键结论:")
    print("   ✓ 对称矩阵：所有特征值是实数，特征向量正交")
    print("   ✗ 非对称矩阵：可能有复特征值，特征向量不一定正交")


# ============================================================
# 🧪 实验六：协方差矩阵的特征分解 (与第 6 章呼应)
# ============================================================
def experiment_6_covariance_eigendecomposition():
    """实验六：协方差矩阵的特征分解 → 椭球主轴"""
    
    print("\n" + "=" * 70)
    print("🧪 实验六：协方差 → 椭球 (3DGS 应用)")
    print("=" * 70)
    
    # 构造一个正定协方差矩阵
    Sigma = np.array([
        [4.0, 1.5, 0.8],
        [1.5, 3.0, 1.2],
        [0.8, 1.2, 2.0]
    ])
    
    print(f"协方差矩阵 Σ:\n{Sigma}")
    
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\n特征值 (方差): λ₁={eigenvalues[0]:.3f}, λ₂={eigenvalues[1]:.3f}, λ₃={eigenvalues[2]:.3f}")
    print(f"半轴长度：σ₁={np.sqrt(eigenvalues[0]):.3f}, σ₂={np.sqrt(eigenvalues[1]):.3f}, σ₃={np.sqrt(eigenvalues[2]):.3f}")
    
    # 提取主轴方向
    for i in range(3):
        angle_x = np.degrees(np.arccos(np.clip(eigenvectors[0, i], -1, 1)))
        print(f"  主轴{i+1}方向：[{eigenvectors[0,i]:.3f}, {eigenvectors[1,i]:.3f}, {eigenvectors[2,i]:.3f}]")
    
    # ====== 验证椭球方程 ======
    print("\n🔍 验证：二次型 xᵀΣ⁻¹x = 1 → 椭球表面")
    # 生成主轴方向的点，应该在椭球表面上
    for i in range(3):
        point = eigenvectors[:, i] * np.sqrt(eigenvalues[i])  # 沿主轴走到半轴位置
        quad_form = point.T @ np.linalg.inv(Sigma) @ point
        
        print(f"  主轴{i+1}上的点：{point}")
        print(f"    xᵀΣ⁻¹x = {quad_form:.3f} (应该≈1)")
    
    if abs(quad_form - 1.0) < 1e-5:
        print("✅ 验证通过!")


# ============================================================
# 🎯 主函数：运行所有实验
# ============================================================

def main():
    """运行所有实验"""
    
    print("\n" + "=" * 70)
    print("🔥 第 8 章：特征值与特征向量")
    print("=" * 70)
    print("\n这个 Notebook 演示了:")
    print("1. Av = λv验证(只缩放不转弯)")
    print("2. 特征分解 A = VΛVᵀ")
    print("3. 矩阵幂运算与特征值的关系")
    print("4. 幂迭代法求主特征值")
    print("5. 对称 vs 非对称矩阵的对比")
    print("6. 协方差 → 椭球 (3DGS 应用)")
    
    # 运行所有实验
    experiment_1_av_equals_lambda_v()
    experiment_2_eigen_decomposition()
    experiment_3_eigenvalues_and_matrix_powers()
    experiment_4_power_iteration()
    experiment_5_symmetric_vs_nonsymmetric()
    experiment_6_covariance_eigendecomposition()
    
    print("\n" + "=" * 70)
    print("🎉 所有实验完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
