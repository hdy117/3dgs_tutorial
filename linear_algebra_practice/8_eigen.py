import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 GTK 错误
import matplotlib.pyplot as plt

# 构造一个对称的正定矩阵（协方差矩阵的典型形式）
A = np.array([
    [3.0, 1.0],
    [1.0, 2.0],
])

print("=" * 60)
print("特征值与特征向量的可视化")
print("=" * 60)
print(f"\n矩阵 A（对称正定）:")
print(A)

# 计算特征分解 (eigh 专门用于对称矩阵)
eigvals, eigvecs = np.linalg.eigh(A)

print(f"\n1. 特征值：{eigvals}")
for i, val in enumerate(eigvals):
    print(f"   λ_{i+1} = {val:.4f}")

print(f"\n2. 特征向量（列）:")
for i, vec in enumerate(eigvecs.T):
    print(f"   v_{i+1} = [{vec[0]:.4f}, {vec[1]:.4f}]")
    # 验证 A @ v = λ @ v
    Av = A @ vec
    print(f"      A@v = [{Av[0]:.4f}, {Av[1]:.4f}] = {eigvals[i]} * v ✓")

print("\n" + "=" * 60)

# 可视化：单位圆变椭圆
t = np.linspace(0, 2*np.pi, 200)
circle = np.stack([np.cos(t), np.sin(t)])
ellipse = A @ circle

plt.figure(figsize=(8, 4))

# 左图：输入空间，显示特征方向
plt.subplot(1, 2, 1)
plt.plot(circle[0], circle[1], 'b-', alpha=0.3, label='Unit Circle')
for i in range(len(eigvals)):
    vec = eigvecs[:, i]
    scaled = eigvals[i] * vec
    plt.quiver(0, 0, vec[0], vec[1], color='red', scale=5, width=0.003, 
               label=f'Eigenvector v_{i+1} (λ={eigvals[i]:.2f})' if i==0 else "")
    plt.quiver(0, 0, scaled[0], scaled[1], color='orange', scale=5, width=0.003, alpha=0.7)
plt.title('Input Space: Eigen Directions')
plt.axis('equal')
plt.legend()

# 右图：输出空间，显示变换后的结果
plt.subplot(1, 2, 2)
plt.plot(circle[0], circle[1], 'b-', alpha=0.3, label='Unit Circle')
plt.plot(ellipse[0], ellipse[1], 'g-', linewidth=2, label='A @ Circle = Ellipse')
for i in range(len(eigvals)):
    vec = eigvecs[:, i]
    scaled = A @ vec  # 这就是 λ_i * v_i
    plt.quiver(0, 0, scaled[0], scaled[1], color='orange', scale=5, width=0.003, 
               label=f'A@v_{i+1} (scaled {eigvals[i]:.2f}x)' if i==0 else "")
plt.title('Output Space: Eigenvectors Scaled')
plt.axis('equal')
plt.legend()

plt.tight_layout()
# 保存到文件（使用 Agg 后端，不弹出窗口）
output_path = '8_eigen_result.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n📊 图像已保存到: {output_path}")

print("\n" + "=" * 60)
print("🎯 关键理解:")
for i, (val, vec) in enumerate(zip(eigvals.T, eigvecs.T)):
    print(f"   - λ_{i+1}={val:.2f}: 方向 [{vec[0]:.2f}, {vec[1]:.2f}] 被拉长{val:.2f}倍")
print("   → 椭圆的主轴就是特征向量，半轴长度就是特征值！")