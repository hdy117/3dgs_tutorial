import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 同一个几何向量（在标准坐标下）
v = np.array([3.0, 1.0])

# 一套旋转了 45 度的新基
th = np.deg2rad(45)
B = np.array([
    [np.cos(th), -np.sin(th)],
    [np.sin(th),  np.cos(th)],
])

# v 在新基下的坐标
c_new = np.linalg.inv(B) @ v

print('standard coords =', v)
print('rotated-basis coords =', c_new)

plt.figure(figsize=(6, 6))
plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linewidth=1)

# 原坐标基
plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='tab:blue')
plt.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='tab:blue')

# 新基
plt.quiver(0, 0, B[0,0], B[1,0], angles='xy', scale_units='xy', scale=1, color='tab:orange')
plt.quiver(0, 0, B[0,1], B[1,1], angles='xy', scale_units='xy', scale=1, color='tab:orange')

# 同一个几何向量
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='tab:green')

plt.xlim(-2, 4)
plt.ylim(-2, 4)
plt.gca().set_aspect('equal')
plt.title('Same vector, different coordinates')
plt.show()