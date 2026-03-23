# 第4章：可微分渲染管线

**学习路径**：`invention → verification`

**核心目标**：将第3章的数学公式转化为实际可计算的渲染流程

---

## 一、引言：从数学公式到实际渲染

### 1.1 第3章回顾

第3章我们推导了：
- 3D高斯参数：μ, Σ, α, c
- 投影公式：μ₂D, Σ₂D（透视投影）
- Alpha Blending：按深度排序 + 累加

### 1.2 本章任务

将这些公式组合成**可运行的渲染管线**：

```
输入：{ (μ_i, Σ_i, α_i, c_i) }，相机参数 K, R, t
输出：2D图像 C（H×W×3）
过程：
  1. 投影：μ_i → 2D坐标，Σ_i → 2D协方差
  2. 计算2D高斯值（每个像素）
  3. 按深度排序
  4. Alpha blending 累加颜色
```

---

## 二、Step 1：3D → 2D投影（透视相机）

### 2.1 3D中心投影

**标准小孔模型**：
```
x_2d = K * (R * μ + t)
其中：
  μ ∈ ℝ³ 世界坐标
  R, t 相机外参
  K ∈ ℝ³ˣ³ 内参矩阵（fx, fy, cx, cy）
```

**深度z**：
- 在相机坐标系：z = (Rμ + t)_z
- 用于排序

---

### 2.2 3D协方差投影到2D

**挑战**：透视投影是非线性的（x = X/Z）

**推导**（详见3DGS论文附录）：

令相机坐标系下的3D点：X = R * μ + t
透视投影：x = K * [X_x/X_z, X_y/X_z]ᵀ

**雅可比矩阵 J**（投影变换的线性近似）：
```
J = ∂x/∂X = [1/Z  0  -X_x/Z²]
            [0  1/Z  -X_y/Z²]  * K
```

**投影协方差**：
```
Σ_2D = J * W * Σ * Wᵀ * Jᵀ
其中 W = R（将世界坐标转到相机坐标系）
```

**简化理解**：
- 透视投影相当于在3D空间先乘以R，再用非线性x=X/Z投影
- 雅可比J近似了非线性在局部的影响
- 对于靠近中心线（小视差）的高斯，近似是准确的

**实际实现**：
```python
# 1. 转到相机坐标系
mu_cam = R @ mu + t  # (3,)
Sigma_cam = R @ Sigma @ R.T  # (3,3)

# 2. 计算雅可比
z = mu_cam[2]
J = torch.tensor([
    [K[0,0]/z, 0, -K[0,0]*mu_cam[0]/z**2],
    [0, K[1,1]/z, -K[1,1]*mu_cam[1]/z**2]
], device=mu.device)  # (2,3)

# 3. 投影协方差
Sigma_2d = J @ Sigma_cam @ J.T  # (2,2)
```

---

## 三、Step 2：2D高斯在像素网格上的评估

### 3.1 2D高斯函数

给定：
- 中心 μ_2D = (u, v) 像素坐标
- 协方差 Σ_2D = [[σ_uu, σ_uv], [σ_uv, σ_vv]]

对于任意像素 x = (i, j)，高斯值为：
```
G(x) = exp(-0.5 * (x - μ)^T Σ⁻¹ (x - μ))
```

---

### 3.2 计算优化

**问题**：直接计算逆矩阵 Σ⁻¹ 是O(4)操作，可接受但略慢

**优化方案**：

**方案A：Cholesky分解**
```python
L = torch.linalg.cholesky(Sigma_2d)  # Σ = L L^T
diff = x - mu_2D
solve = torch.triangular_solve(diff, L, upper=False)[0]
exponent = -0.5 * (solve**2).sum()
```

**方案B：特征值分解**
```python
eigvals, eigvecs = torch.linalg.eigh(Sigma_2d)
inv_diag = 1.0 / eigvals.clamp(min=1e-8)
diff = x - mu_2D
rotated = eigvecs.T @ diff
exponent = -0.5 * (rotated**2 * inv_diag).sum()
```

**数值稳定性**：
- 添加小量 ε 到对角线防止奇异：`Sigma_2d + ε * I`
- 限制协方差的最小/最大特征值（避免过扁或过圆）

---

### 3.3 渲染窗口（Tile-based）

**问题**：每个高斯理论上影响整个图像，但实际上只影响μ附近3σ范围

**优化**：
- 计算2D高斯的"包围盒"：μ ± 3 * sqrt(最大特征值)
- 只在该包围盒内评估高斯值
- 进一步：按图像分块（tile），每个tile预计算需要的高斯列表

**复杂度分析**：
- 无优化：O(H*W*N) → 不可接受（H*W=2M, N=1M → 2万亿次）
- 有窗口：每个高斯只影响 ~100像素 → 总复杂度 O(N * 100)
- 对于 N=1M → 1亿次操作，可接受

---

## 四、Step 3：深度排序

### 4.1 为什么需要排序？

**Alpha blending 公式**：
```
C = Σ (g_i.α * g_i.c * Π_{j<i} (1 - g_j.α))
```

只有从远到近累加，才能正确计算遮挡。

**交换律不成立**：
```
a₁·c₁·(1-a₂) + a₂·c₂ ≠ a₂·c₂·(1-a₁) + a₁·c₁
```
除非 a₁=0 或 a₂=0

---

### 4.2 排序依据

- 使用投影后的深度 z = (Rμ + t)_z
- 所有高斯按 z **降序**排列（远的先画）

**近似性分析**：
- 高斯有体积，不是点 → 严格说应该按"质心深度"排序
- 但对于细长高斯，中心深度是合理近似
- 实际效果：轻微渲染瑕疵，但可接受

**实现**：
```python
depths = (R @ mu.T + t[:, None]).T[:, 2]  # (N,)
sorted_indices = torch.argsort(depths, descending=True)
```

---

### 4.3 排序的代价

- O(N log N) 排序
- N=1M 时，~20M次比较 → 在GPU上很快（<1ms）
- 优化：如果高斯集合不变（推理时），排序可缓存

---

## 五、Step 4：Alpha Blending（逐tile累加）

### 5.1 伪代码

```python
# 初始化
image = torch.zeros((3, H, W))
accum_alpha = torch.zeros((1, H, W))

for g in sorted_gaussians:
    # 计算影响范围 [x_min, x_max] × [y_min, y_max]
    bbox = compute_bbox(g.mu_2d, g.Sigma_2d)
    for i in range(bbox.x_min, bbox.x_max):
        for j in range(bbox.y_min, bbox.y_max):
            # 计算高斯值
            value = g.α * exp(-0.5 * quad_form)
            # Alpha blending
            image[:,j,i] += value * g.c * (1 - accum_alpha[:,j,i])
            accum_alpha[:,j,i] += value
            # 早停
            if accum_alpha[0,j,i] >= 0.99:
                break
```

---

### 5.2 并行化策略

**Tile-based并行**：
- 每个tile独立 → 多线程并行处理不同tile
- GPU实现：每个block处理一个tile，线程处理像素
- 需要原子操作更新 accum_alpha（或使用Warp-level reduction）

**内存访问优化**：
- 高斯数据：连续读取（coalesced）
- 图像写入：同一block的线程写入相邻像素 → coalesced

---

## 六、可微分性验证

### 6.1 整个管道的可微节点

```
μ, Σ, α, c  ← 梯度 ← 最终图像 C
  ↑              ↑
投影公式        Alpha blending
  ↑              ↑
矩阵乘法        乘加运算
```

### 6.2 关键点

- 投影公式：矩阵乘法 → 可微
- 高斯评估：指数+二次型 → 可微
- Alpha blending：乘加 → 可微

### 6.3 唯一"问题"

**排序是离散操作** → 梯度不连续

**实际处理**：
- 在训练中，排序顺序相对稳定（高斯不会突然跳变）
- 或使用软化排序（如使用α值加权中心深度的连续近似）
- 论文：直接使用离散排序，梯度通过固定顺序传播

---

## 七、实现要点（CUDA视角）

### 7.1 内存布局

- 所有高斯参数存储在GPU全局内存：N × 13 float
- SoA（Structure of Arrays）更利于向量化：
  ```c
  struct Gaussians {
      float* mu;       // (N, 3)
      float* Sigma;    // (N, 6) 对称矩阵
      float* alpha;    // (N, 1)
      float* color;    // (N, 3)
  };
  ```

---

### 7.2 Kernel设计

**三阶段设计**：

**Stage 1：Projection kernel**
```cuda
__global__ void project_kernel(Gaussian g, Camera cam, Projected* out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;
    // 计算 mu_2d, Sigma_2d, depth
    out[idx].mu_2d = ...
    out[idx].Sigma_2d = ...
    out[idx].depth = ...
}
```

**Stage 2：Sort kernel**
- 使用Thrust库：`thrust::sort_by_key(depths, indices)`

**Stage 3：Render kernel**
```cuda
__global__ void render_kernel(Projected* gaussians, int N, Camera cam, uchar4* image) {
    // 每个block处理一个tile
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    // 共享内存缓存该tile涉及的高斯
    extern __shared__ Projected shared_gaussians[];
    // 预筛选 + 加载
    // 每个像素累加
}
```

---

### 7.3 内存访问模式

- 高斯数据：连续读取（coalesced）
- 图像写入：block内线程写相邻像素 → 合并访问

---

## 八、思考题（独立重推检验）

1. **透视投影推导**：自己推导雅可比矩阵 J。假设相机坐标系下点X=(X, Y, Z)，写出投影到归一化坐标 x=X/Z, y=Y/Z 的偏导数。
2. **排序的影响**：如果两个高斯深度相同但顺序颠倒，Alpha blending结果是否相同？为什么？
3. **性能分析**：假设N=1M高斯，每帧只渲染100k高斯的窗口。总复杂度是多少？如果图像分辨率为800×600，总像素数480k，哪个是瓶颈？
4. **可微性边界**：如果我们在训练时使用排序，但推理时不排序（假设顺序固定），梯度会受影响吗？为什么？

---

## 九、下一章预告

**第5章**：优化目标与损失函数 - 如何让高斯学习到正确的形状和颜色？详解重建损失（L1+SSIM）、正则化项、自适应密度控制。

---

**关键记忆点**：
- ✅ 投影：μ₂D = K(Rμ+t), Σ₂D = J·(RΣRᵀ)·Jᵀ
- ✅ 排序：按深度降序，保证Alpha blending正确
- ✅ Tile优化：每个高斯只影响局部像素
- ✅ 可微：整个管线可反向传播
- 🎯 **渲染复杂度：O(N·窗口大小)，不是O(N·H·W)**
