import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [1.2, 0.6],
    [0.2, 1.0],
])

xs = np.linspace(-2, 2, 9)
ys = np.linspace(-2, 2, 9)

plt.figure(figsize=(12, 5))

# 原网格
plt.subplot(1, 2, 1)
for x in xs:
    pts = np.stack([np.full_like(ys, x), ys], axis=0)
    plt.plot(pts[0], pts[1], color='lightgray')
for y in ys:
    pts = np.stack([xs, np.full_like(xs, y)], axis=0)
    plt.plot(pts[0], pts[1], color='lightgray')
plt.title('Before')
plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# 变换后网格
plt.subplot(1, 2, 2)
for x in xs:
    pts = np.stack([np.full_like(ys, x), ys], axis=0)
    pts_t = A @ pts
    plt.plot(pts_t[0], pts_t[1], color='lightgray')
for y in ys:
    pts = np.stack([xs, np.full_like(xs, y)], axis=0)
    pts_t = A @ pts
    plt.plot(pts_t[0], pts_t[1], color='lightgray')
plt.title('After y = A x')
plt.axis('equal')
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.show()