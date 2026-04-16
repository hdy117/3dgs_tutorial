"""
第 9 章：SVD - 任意矩阵的三步拆解
====================================

这个 Notebook 完整演示了 SVD(奇异值分解)的核心概念和应用。
理解"旋转→拉伸→再旋转"的结构，以及为什么它总是存在!

对应章节：第 9 章
"""

import numpy as np


# ============================================================
# 🧪 实验一：验证 SVD 的核心关系 A @ v_i = σ_i * u_i
# ============================================================
def experiment_1_core_relationship():
    """实验一：验证 SVD 的核心数学关系"""
    
    print("=" * 70)
    print("🧪 实验一：SVD 核心关系验证")
    print("=" * 70)
    
    # 构造一个一般的矩阵 (非方阵!)
    A = np.array([
        [3.0, 1.0, 0.5],
        [0.5, 2.0, 1.0]
    ])
    
    print(f"矩阵 A ({A.shape[0]} x {A.shape[1]}):")
    print(A)
    
    # 计算 SVD
    U, S, Vt = np.linalg.svd(A)
    
    print(f"\n奇异值：{S}")
    
    for i in range(len(S)):
        vi = Vt[i]   # v_i^T (Vt 的行是 v_i^T，所以取行就是 v_i)
        ui = U[:, i]  # u_i
        
        print(f"\n第{i+1}个奇异对:")
        print(f"  v_{i+1} = {vi}")
        print(f"  σ_{i+1} = {S[i]:.4f}")
        
        # 验证核心关系：A @ v_i = σ_i * u_i
        left = A @ vi
        right = S[i] * ui
        
        print(f"\n  验证：A · v_{i+1} = ?= σ_{i+1} · u_{i+1}")
        print(f"    A · v = {left}")
        print(f"    σ · u = {right}")
        
        if np.allclose(left, right):
            print(f"  ✅ 验证通过!")
        else:
            print(f"  ❌ 验证失败!")


# ============================================================
# 🧪 实验二：从 A^T A 推导 SVD (手动过程)
# ============================================================
def experiment_2_manual_svd_derivation():
    """实验二：从 A^TA 手动推导 SVD"""
    
    print("\n" + "=" * 70)
    print("🧪 实验二：从 A^T A 推导 SVD")
    print("=" * 70)
    
    # 一个简单的矩阵
    A = np.array([
        [3.0, 1.0],
        [1.0, 2.0]
    ])
    
    print(f"矩阵 A:\n{A}")
    
    # ====== Step 1:计算 A^T A (对称半正定)!======
    ATA = A.T @ A
    
    print(f"\nStep 1: Aᵀ A:")
    print(ATA)
    print(f"✅ 对称吗？{np.allclose(ATA, ATA.T)}")
    
    # ====== Step 2:对 A^T A 做特征分解 → V 和 σ²======
    eigenvalues_AT_A, Vt_from_eig = np.linalg.eigh(ATA)
    V_from_eig = Vt_from_eig.T
    
    print(f"\nStep 2: AᵀA 的特征分解")
    print(f"特征值 (σ²): {eigenvalues_AT_A}")
    print(f"特征向量矩阵 Vᵀ:\n{Vt_from_eig}")
    
    # ====== Step 3:提取奇异值 σ = √λ======
    singular_values = np.sqrt(eigenvalues_AT_A)
    
    print(f"\nStep 3: 奇异值 σ = √λ:")
    print(singular_values)
    
    # ====== Step 4:与 numpy 的 SVD 对比 ======
    U_direct, S_direct, Vt_direct = np.linalg.svd(A)
    
    print(f"\n直接计算 (np.linalg.svd):")
    print(f"奇异值：{S_direct}")
    print(f"Vᵀ:\n{Vt_direct}")
    
    print(f"\n✅ 奇异值一致？{np.allclose(singular_values, S_direct)}")


# ============================================================
# 🧪 实验三：可视化 SVD的三步拆解
# ============================================================
def experiment_3_visualize_svd_three_steps():
    """实验三：可视化 SVD 的三步拆解过程"""
    
    print("\n" + "=" * 70)
    print("🧪 实验三：SVD 三步拆解可视化")
    print("=" * 70)
    
    # 构造矩阵
    A = np.array([[3.0, 1.5], [1.5, 2.0]])
    
    print(f"矩阵 A:\n{A}")
    
    # 计算 SVD
    U, S, Vt = np.linalg.svd(A)
    
    print(f"\n奇异值：σ₁={S[0]:.3f}, σ₂={S[1]:.3f}")
    
    # ====== 可视化描述 (文本版) ======
    print("\n三步拆解过程:")
    print("-" * 70)
    print("Step 1: Vᵀ旋转")
    for i in range(2):
        vi = Vt[i]
        print(f"   v_{i+1} = [{vi[0]:.3f}, {vi[1]:.3f}]")
    
    print("\nStep 2: Σ拉伸 (对角缩放)")
    for i in range(2):
        print(f"   σ_{i+1} = {S[i]:.3f}")
    
    print("\nStep 3: U 旋转")
    for i in range(2):
        ui = U[:, i]
        print(f"   u_{i+1} = [{ui[0]:.3f}, {ui[1]:.3f}]")
    
    # ====== 验证 A = U Σ Vᵀ ======
    reconstructed = U @ np.diag(S) @ Vt
    
    print("\n验证：A ≈ U Σ Vᵀ ?")
    print(f"原始矩阵:\n{A}")
    print(f"重构矩阵:\n{reconstructed}")
    
    error = np.linalg.norm(A - reconstructed, 'fro')
    print(f"Frobenius 误差：{error:.2e}")
    
    if error < 1e-10:
        print("✅ 完美匹配!")
    else:
        print("⚠️ 有数值误差 (正常)")


# ============================================================
# 🧪 实验四：低秩近似与 Eckart-Young-Mirsky 定理
# ============================================================
def experiment_4_low_rank_approximation():
    """实验四：SVD 低秩近似 - Eckart-Young 定理验证"""
    
    print("\n" + "=" * 70)
    print("🧪 实验四：低秩近似与压缩")
    print("=" * 70)
    
    # 生成一个低秩矩阵 (带噪声)
    np.random.seed(42)
    m, n = 50, 40
    k_true = 3
    
    U_true = np.random.randn(m, k_true)
    V_true = np.random.randn(n, k_true)
    A_lowrank = U_true @ V_true.T + np.random.randn(m, n) * 0.1
    
    print(f"矩阵形状：{A_lowrank.shape}")
    
    # ====== SVD 分解 ======
    U, S, Vt = np.linalg.svd(A_lowrank, full_matrices=False)
    
    print(f"\n奇异值分布:")
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
        
        # Eckart-Young-Mirsky 理论下界
        theoretical_bound = np.sqrt(np.sum(S[k:]**2))
        
        print(f"  k={k:2d}: {rel_error:.4f}%, 理论下界 = {theoretical_bound:.6f}")
    
    # ====== 压缩比计算 ======
    original_size = m * n
    for k in [3, 5]:
        compressed_size = k * (m + n + 1)
        compression_ratio = compressed_size / original_size * 100
        
        print(f"\nk={k}时:")
        print(f"  压缩比：{compression_ratio:.1f}%")
        print(f"  节省空间：{100-compression_ratio:.1f}%")


# ============================================================
# 🧪 实验五：PCA = SVD的关系验证
# ============================================================
def experiment_5_pca_equals_svd():
    """实验五：验证 PCA = SVD"""
    
    print("\n" + "=" * 70)
    print("🧪 实验五：PCA 与 SVD 的等价性")
    print("=" * 70)
    
    # ====== 生成数据集 ======
    np.random.seed(42)
    X = np.random.randn(100, 50)  # 100 个样本，50 维特征
    
    # Step 1:标准化 (PCA 的前提)
    X_centered = X - np.mean(X, axis=0)
    
    print(f"数据矩阵形状：{X_centered.shape}")
    
    # ====== 方法 A:协方差矩阵特征分解 (传统 PCA) ======
    Sigma = X_centered.T @ X_centered / len(X_centered)
    eigenvalues_pca, eigenvectors_pca = np.linalg.eigh(Sigma)
    
    # 从大到小排序
    idx = np.argsort(eigenvalues_pca)[::-1]
    eigenvalues_pca = eigenvalues_pca[idx]
    eigenvectors_pca = eigenvectors_pca[:, idx]
    
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
    print("=" * 70)
    
    for i in range(5):
        match = "✅" if np.allclose(eigenvalues_pca[i], variances_from_svd[i]) else "❌"
        print(f"  PC{i+1}: {eigenvalues_pca[i]:.6f} vs {variances_from_svd[i]:.6f} {match}")
    
    # ====== 应用：降维 ======
    k = 5
    X_reduced = X_centered @ eigenvectors_pca[:, :k]
    reconstructed = X_reduced @ eigenvectors_pca[:, :k].T
    error = np.linalg.norm(X_centered - reconstructed, 'fro') / \
            np.linalg.norm(X_centered, 'fro') * 100
    
    print(f"\n降维效果:")
    print(f"  原始维度：{X.shape[1]} → 降维后：{k} ({k/X.shape[1]*100:.1f}%)")
    print(f"  重构误差：{error:.2f}%")
    
    print("\n🎯 关键结论:")
    print("   ✓ SVD 的 Vᵀ 行 = PCA 的主成分方向 (允许符号差异)")
    print("   ✓ σᵢ²/N = PCᵢ的方差")
    print("   ✓ PCA 和 SVD 是同一个东西的不同实现方式!")


# ============================================================
# 🎯 主函数：运行所有实验
# ============================================================

def main():
    """运行所有实验"""
    
    print("\n" + "=" * 70)
    print("🔥 第 9 章：SVD - 任意矩阵的三步拆解")
    print("=" * 70)
    print("\n这个 Notebook 演示了:")
    print("1. SVD 核心关系 A·vᵢ = σᵢ·uᵢ验证")
    print("2. 从 A^T A 手动推导 SVD 的过程")
    print("3. SVD三步拆解可视化 (Vᵀ→Σ→U)")
    print("4. Eckart-Young 低秩近似定理")
    print("5. PCA = SVD的等价性证明")
    
    # 运行所有实验
    experiment_1_core_relationship()
    experiment_2_manual_svd_derivation()
    experiment_3_visualize_svd_three_steps()
    experiment_4_low_rank_approximation()
    experiment_5_pca_equals_svd()
    
    print("\n" + "=" * 70)
    print("🎉 所有实验完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
