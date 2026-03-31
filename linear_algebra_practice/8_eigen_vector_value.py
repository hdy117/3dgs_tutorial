import numpy as np

# 非对称矩阵（带剪切）
A = np.array([[1, 2], [0, 1]])
eigenvalues_A, eigenvectors_A = np.linalg.eig(A)

# 对称矩阵（协方差典型形式）
B = np.array([[3, 1], [1, 2]])
eigenvalues_B, eigenvectors_B = np.linalg.eigh(B)

print(f"非对称矩阵 A:")
print(f"  特征值：{eigenvalues_A}, 特征向量：{eigenvectors_A}")
print(f"  特征向量正交吗？{np.allclose(eigenvectors_A[:,0] @ eigenvectors_A[:,1], 0)}")

print(f"\n对称矩阵 B:")
print(f"  特征值：{eigenvalues_B}, 特征向量：{eigenvectors_B}")
print(f"  特征向量正交吗？{np.allclose(eigenvectors_B[:,0] @ eigenvectors_B[:,1], 0)}")