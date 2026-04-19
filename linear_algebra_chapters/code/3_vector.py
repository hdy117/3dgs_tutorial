import numpy as np
import matplotlib.pyplot as plt

# 两个二维向量
A = np.array([3.0, 1.0])
B = np.array([1.0, 2.5])

# B 在 A 方向上的投影
proj_B_on_A = (A @ B) / (A @ A) * A

plt.figure(figsize=(6, 6))
plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linewidth=1)

plt.quiver(0, 0, A[0], A[1], angles='xy', scale_units='xy', scale=1, color='tab:blue', label='A')
plt.quiver(0, 0, B[0], B[1], angles='xy', scale_units='xy', scale=1, color='tab:orange', label='B')
plt.quiver(0, 0, proj_B_on_A[0], proj_B_on_A[1], angles='xy', scale_units='xy', scale=1, color='tab:green', label='proj_A(B)')

plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.gca().set_aspect('equal')
plt.legend()
plt.title(f'dot(A, B) = {A @ B:.2f}')
plt.show()