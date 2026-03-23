# 第4章：可微分渲染管线

**学习路径**：`invention → verification`

**核心目标**：将第3章的数学公式转化为实际可计算的渲染流程

---

## 一、渲染管线总览

### 1.1 完整流程图

```
输入: 高斯集合 {μ,Σ,α,c} + 相机
    |
    +---> Step1: 投影
    |        → μ₂D, Σ₂D, depth
    |
    +---> Step2: 按depth排序
    |
    +---> Step3: Tile预筛选
    |        → 每个tile的高斯列表
    |
    +---> Step4: Alpha Blending
             → 输出: 2D图像 C

可微路径:
投影 → 评估 → Blending → C
   ↓      ↓       ↓
  梯度反向传播到 μ,Σ,α,c
```

---

### 1.2 时间复杂度分析

```
+--------+----------------+-----------------+------------------+
| 阶段   | 操作           | 复杂度          | 典型时间(N=1M)   |
+--------+----------------+-----------------+------------------+
| 投影   | 矩阵乘法       | O(N)            | 2-5 ms           |
| 排序   | quick sort     | O(N log N)      | 5-10 ms          |
| Tile预筛选| 包围盒计算   | O(N)            | 1-2 ms           |
| 渲染   | 每像素遍历tile内高斯| O(N·100)   | 5-15 ms          |
| 总计   | -              | -               | 13-32 ms         |
+--------+----------------+-----------------+------------------+
```

**瓶颈**: 渲染阶段 (像素遍历)

---

## 二、Step 1：投影 (3D → 2D)

### 2.1 相机坐标系转换

**世界 → 相机**:
```
X_cam = R · (μ - t)  # 注意: 有些约定是 R·μ + t
实际: X_cam = R·μ + t
```

**代码**:
```python
mu_cam = torch.bmm(R, mu.unsqueeze(-1)).squeeze(-1) + T  # (N,3)
```

---

### 2.2 投影中心计算

**小孔模型**:
```
x_norm = [X_x/Z, X_y/Z]
x_pixel = K · x_norm
```

**代码**:
```python
mu_hom = torch.bmm(K, mu_cam.unsqueeze(-1)).squeeze(-1)  # (N,3)
mu_2d = mu_hom[:, :2] / mu_hom[:, 2:3]  # 除以z
```

---

### 2.3 协方差投影 (核心数学)

**透视投影的雅可比矩阵**:

对于相机坐标系点 X = (X, Y, Z):
```
x = K[0,0]·X/Z + K[0,2]
y = K[1,1]·Y/Z + K[1,2]

∂x/∂X = K[0,0]/Z
∂x/∂Y = 0
∂x/∂Z = -K[0,0]·X/Z²

∂y/∂X = 0
∂y/∂Y = K[1,1]/Z
∂y/∂Z = -K[1,1]·Y/Z²

J = [K00/Z, 0, -K00·X/Z²]
    [0, K11/Z, -K11·Y/Z²]
```

**投影协方差**:
```
Σ_2D = J · (R·Σ·Rᵀ) · Jᵀ
```

**代码**:
```python
# 1. Σ转到相机坐标系
Sigma_cam = R @ Sigma @ R.T  # (N,3,3)

# 2. 计算雅可比J
z = mu_cam[:, 2].clamp(min=1e-6)  # (N,)
J = torch.zeros((N, 2, 3), device=mu.device)
J[:, 0, 0] = K[0, 0] / z
J[:, 0, 2] = -K[0, 0] * mu_cam[:, 0] / (z**2)
J[:, 1, 1] = K[1, 1] / z
J[:, 1, 2] = -K[1, 1] * mu_cam[:, 1] / (z**2)

# 3. Σ_2D
Sigma_2d = J @ Sigma_cam @ J.transpose(1, 2)  # (N,2,2)
```

---

### 2.4 数值稳定性

**问题**:
- z接近0 → J很大 → Σ_2D爆炸
- Σ_cam接近奇异 → 数值不稳定

**解决方案**:
```python
# 1. 限制z最小值
z = mu_cam[:, 2].clamp(min=1e-6)

# 2. Σ_2D添加正则项
epsilon = 1e-8
Sigma_2d = Sigma_2d + torch.eye(2, device=mu.device)[None,:,:] * epsilon

# 3. 限制Σ的特征值范围
eigvals, eigvecs = torch.linalg.eigh(Sigma_2d)
eigvals = eigvals.clamp(min=1e-6, max=1e2)
Sigma_2d = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(1,2)
```

---

## 三、Step 2：深度排序

### 3.1 排序依据选择

```
+--------+----------------+------+------+
| 候选   | 公式           | 精度 | 速度 | 推荐 |
+--------+----------------+------+------+------+
| 中心深度| z = μ_cam[2]   | ⚠️ 近似 | ✅ 快 | ✅ 推荐 |
| 平均深度| 高斯质心深度   | ✅ 更准| ⚠️ 需计算| ⚠️ 可选 |
| 最近点 | min(μ-3σ)      | ✅ 最准| ❌ 慢 | ❌ 不推荐 |
+--------+----------------+------+------+------+
```

**为什么中心深度足够?**
- 高斯有体积，但中心是质心
- 对于细长高斯，中心和质心接近
- 轻微瑕疵可接受

---

### 3.2 排序实现

```python
# 计算深度
depths = mu_cam[:, 2]  # (N,)

# 降序排列 (远的先画)
sorted_indices = torch.argsort(depths, descending=True)

# 重排序所有张量
mu_2d = mu_2d[sorted_indices]
Sigma_2d = Sigma_2d[sorted_indices]
alpha = alpha[sorted_indices]
color = color[sorted_indices]
```

**性能**: N=1M时，排序 ~5-10 ms (GPU快速排序)

---

## 四、Step 3：Tile预筛选

### 4.1 为什么需要预筛选?

**问题**: 每个高斯理论上影响全图，但实际只影响μ附近3σ范围

**无预筛选复杂度**:
```
每像素遍历所有高斯: O(H·W·N)
H=1200, W=900, N=1M → 1.08×10¹² 次操作 ❌
```

**预筛选后**:
```
每个高斯影响 ~100像素
总操作: N × 100 = 10⁸ ✅
加速: 10,000倍!
```

---

### 4.2 包围盒计算

**2D高斯3σ范围**:
```
μ₂D ± 3·√(最大特征值)
```

**代码**:
```python
def compute_bbox(mu_2d, Sigma_2d, scale=3.0):
    """计算高斯的屏幕空间包围盒"""
    # 特征值 = 半轴长度平方
    eigvals = torch.linalg.eigvalsh(Sigma_2d)  # (N,2)
    radii = scale * torch.sqrt(eigvals.max(dim=1)[0])  # (N,)
    
    bbox_min = (mu_2d - radii[:, None]).long().clamp(min=0)
    bbox_max = (mu_2d + radii[:, None]).long().clamp(max=W)  # W: 图像宽
    
    return bbox_min, bbox_max  # (N,2)
```

---

### 4.3 Tile映射 (倒排索引)

**Tile划分**:
```
图像: H×W
Tile大小: T=16
Tile数: n_tiles_x = ceil(W/T), n_tiles_y = ceil(H/T)
```

**映射过程**:
```python
def assign_gaussians_to_tiles(bbox_min, bbox_max, tile_size=16, W=800, H=600):
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles = n_tiles_x * n_tiles_y
    
    tile_mapping = [[] for _ in range(n_tiles)]
    
    for g_idx in range(len(bbox_min)):
        x0, y0 = bbox_min[g_idx]
        x1, y1 = bbox_max[g_idx]
        
        # 该高斯影响的所有tile
        tile_x0 = x0 // tile_size
        tile_x1 = x1 // tile_size
        tile_y0 = y0 // tile_size
        tile_y1 = y1 // tile_size
        
        for ty in range(tile_y0, tile_y1+1):
            for tx in range(tile_x0, tile_x1+1):
                tile_id = ty * n_tiles_x + tx
                if 0 <= tile_id < len(tile_mapping):
                    tile_mapping[tile_id].append(g_idx)
    
    return tile_mapping
```

---

## 五、Step 4：Alpha Blending

### 5.1 算法伪代码

```
输入: 已排序高斯列表 G[0..N-1]
输出: 图像 C[H,W,3]

初始化:
  C = zeros(H,W,3)
  A = zeros(H,W)  # accumulated alpha

for g in G:
    # 获取g影响的tile列表
    for tile in g.tiles:
        for pixel in tile.pixels:
            # 计算高斯值
            diff = pixel - g.μ₂D
            exponent = -½·diffᵀ·Σ₂D⁻¹·diff
            g_val = g.α · exp(exponent)
            
            # Alpha blending
            C[pixel] += g_val · g.c · (1 - A[pixel])
            A[pixel] += g_val
            
            if A[pixel] >= 0.99:
                break  # 早停
```

---

### 5.2 2D高斯评估优化

**直接求逆** (慢):
```python
inv_Sigma = torch.linalg.inv(Sigma_2d)  # O(8)
exponent = -0.5 * (diff @ inv_Sigma @ diff)
```

**Cholesky分解** (快):
```python
L = torch.linalg.cholesky(Sigma_2d)  # Σ = LLᵀ, O(4)
solve = torch.triangular_solve(diff.unsqueeze(-1), L, upper=False)[0].squeeze()
exponent = -0.5 * (solve**2).sum()
```

**特征值分解** (最稳):
```python
eigvals, eigvecs = torch.linalg.eigh(Sigma_2d)
inv_diag = 1.0 / eigvals.clamp(min=1e-8)
rotated = eigvecs.T @ diff
exponent = -0.5 * (rotated**2 * inv_diag).sum()
```

**选择建议**:
- Cholesky: 最快，但Σ需正定 (已保证)
- 特征值: 最稳，适合Σ接近奇异的情况

---

### 5.3 早停策略

**条件**: 累计不透明度 A[pixel] ≥ 0.99

**原理**:
```
剩余透射率 T_rem = Π(1-α_i) 已画的高斯
当 T_rem < 0.01 时，后续高斯贡献可忽略
```

**效果**:
- 平均每像素只需遍历 ~20-50 高斯 (而非全部)
- 加速 2-5 倍

---

## 六、可微分性验证

### 6.1 梯度流图

```
参数 μ,Σ,α,c
    ↓
投影: μ₂D,Σ₂D
    ↓
评估: G_i(x)
    ↓
Blending: C(x)
    ↓
损失 L
    ← 反向传播 ←
```

**各操作的可微性**:

```
+----------------+------+---------+
| 操作           | 公式 | 可微?    |
+----------------+------+---------+
| 投影(矩阵乘)   | μ₂D=K(Rμ+t) | ✅ |
| 投影(雅可比)   | Σ₂D=J(RΣRᵀ)Jᵀ| ✅ |
| 高斯评估       | exp(-½ΔᵀΣ⁻¹Δ)| ✅ |
| Alpha blending | C+=α·c·(1-A) | ✅ |
| 排序           | argsort(depth)| ❌ (离散) |
+----------------+------+---------+
```

**排序问题**:
- 训练时顺序稳定 (高斯不跳变) → 梯度有效
- 或使用软化排序 (如 smooth_rank)

---

## 七、CUDA实现要点

### 7.1 内存布局

**推荐: Structure of Arrays (SoA)**

```cpp
// GPU全局内存
struct Gaussians {
    float* mu;       // (N,3) 连续存储
    float* Sigma;    // (N,6) 对称矩阵只存上三角
    float* alpha;    // (N,1)
    float* color;    // (N,3)
    int N;
};
```

**为什么SoA?**
- 读取时连续访问 (coalesced)
- 便于向量化加载 (float4)
- 易于更新部分参数

---

### 7.2 Kernel设计 (三阶段)

#### Stage 1: Projection Kernel

```cpp
__global__ void project_kernel(
    const float* mu, const float* Sigma,
    const float* R, const float* T, const float* K,
    float* mu_2d, float* Sigma_2d, float* depth,
    int N
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;
    
    // 1. mu_cam = R*mu + T
    float3 mu_cam = matvec3(R, mu[idx]) + T;
    depth[idx] = mu_cam.z;
    
    // 2. mu_2d = K * (mu_cam.xy / mu_cam.z)
    float inv_z = 1.0f / mu_cam.z;
    float2 mu_hom = make_float2(
        K[0]*mu_cam.x + K[2],
        K[4]*mu_cam.y + K[5]
    ) * inv_z;
    mu_2d[idx] = mu_hom;
    
    // 3. Sigma_cam = R * Sigma * R^T
    float3x3 Sigma_cam = matmul3(R, Sigma[idx], R.T);
    
    // 4. J矩阵
    float J[2][3] = {
        {K[0]*inv_z, 0, -K[0]*mu_cam.x*inv_z*inv_z},
        {0, K[4]*inv_z, -K[4]*mu_cam.y*inv_z*inv_z}
    };
    
    // 5. Sigma_2d = J * Sigma_cam * J^T
    float3x3 temp = matmul3(J, Sigma_cam);
    Sigma_2d[idx] = matmul2x3(temp, J.T);
}
```

---

#### Stage 2: Sort Kernel (使用Thrust)

```cpp
// 调用Thrust
thrust::sort_by_key(
    thrust::device,      // 执行策略
    depths,              // 键 (深度)
    depths+N,            // 键结束
    indices              // 输出排序索引
);
```

---

#### Stage 3: Render Kernel (Tile-based)

```cpp
__global__ void render_kernel_tiled(
    const Gaussian* gaussians,  // 已排序
    const int* tile_start,      // 每个tile的高斯起始索引
    const int* tile_count,      // 每个tile的高斯数量
    uchar4* image,
    int H, int W, int tile_size
) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_id = tile_y * gridDim.x + tile_x;
    
    // 该tile的像素范围
    int x0 = tile_x * tile_size;
    int x1 = min(x0 + tile_size, W);
    int y0 = tile_y * tile_size;
    int y1 = min(y0 + tile_size, H);
    
    // Shared memory缓存该tile的高斯
    extern __shared__ Gaussian shared_gaussians[];
    int shared_count = 0;
    
    // 加载该tile涉及的所有高斯
    for (int i = threadIdx.x; i < tile_count[tile_id]; i += blockDim.x) {
        int g_idx = tile_start[tile_id] + i;
        shared_gaussians[i] = gaussians[g_idx];
    }
    __syncthreads();
    shared_count = tile_count[tile_id];
    
    // 每个线程处理一个像素
    int px = x0 + threadIdx.x % tile_size;
    int py = y0 + threadIdx.x / tile_size;
    if (px >= x1 || py >= y1) return;
    
    float3 color = make_float3(0,0,0);
    float alpha_acc = 0.0f;
    
    for (int i = 0; i < shared_count; i++) {
        Gaussian& g = shared_gaussians[i];
        
        // 2D高斯评估
        float2 diff = make_float2(px - g.mu_2d.x, py - g.mu_2d.y);
        float exponent = evaluate_gaussian_2d(diff, g.Sigma_2d);
        float g_val = g.alpha * expf(exponent);
        
        // Alpha blending
        color.x += g_val * g.color.x * (1 - alpha_acc);
        color.y += g_val * g.color.y * (1 - alpha_acc);
        color.z += g_val * g.color.z * (1 - alpha_acc);
        alpha_acc += g_val;
        
        if (alpha_acc >= 0.99f) break;
    }
    
    // 写入输出
    int pixel_idx = py * W + px;
    image[pixel_idx] = make_uchar4(
        color.x * 255,
        color.y * 255,
        color.z * 255,
        255
    );
}
```

---

## 八、思考题

1. **性能瓶颈**: 如果N=3M高斯，Tile预筛选能减少多少计算? 估算具体数字
2. **排序稳定性**: 训练中高斯参数在变，深度顺序会变吗? 什么时候需要重排?
3. **数值精度**: float16在投影计算中哪里可能不稳定? 如何检测?
4. **并行度**: 如果block大小=256，每个tile高斯的加载如何设计warp调度?

---

## 九、下一章预告

**第5章**: 优化目标与损失函数 - 如何设计损失让高斯学到正确的形状和颜色? 详解L1+SSIM、正则化、密度控制。

---

**关键记忆点**:
- ✅ 投影: μ₂D = K(Rμ+t), Σ₂D = J·(RΣRᵀ)·Jᵀ
- ✅ 排序: 按深度降序，保证Alpha blending
- ✅ Tile优化: 预筛选减少99%计算
- ✅ 早停: 累计α≥0.99停止
- 🎯 **渲染复杂度**: O(N·窗口大小) ≈ O(N·100)，不是O(N·H·W)