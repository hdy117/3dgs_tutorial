# 第4章：从数学到图像——可微分渲染的完整推导

**学习路径**：`problem → invention → verification`

**本章核心问题**：第3章我们推导了3D高斯投影到2D的公式，但怎么把它变成一个**实际可计算**的渲染管线？而且，怎么让整个流程**可微分**，让梯度能流回高斯参数？

---

## 符号约定表

```
变量名     | 维度           | 含义                         | 单位
-----------|----------------|------------------------------|------
μ          | (N,3)          | 3D高斯中心位置               | 世界坐标
Σ          | (N,3,3)        | 3D协方差矩阵（对称正定）     | 长度²
α          | (N,1)          | 不透明度                     | [0,1]
c          | (N,3)          | 颜色（RGB）                  | [0,1]
K          | (3,3)          | 相机内参矩阵                 | 像素
R          | (3,3)          | 旋转矩阵（世界→相机）        | 无单位
T          | (3,)           | 平移向量（世界→相机）        | 世界坐标
μ_cam      | (N,3)          | 相机坐标系下的3D中心         | 相机坐标
μ_2d       | (N,2)          | 图像平面投影中心（像素）     | 像素
Σ_cam      | (N,3,3)        | 相机坐标系协方差             | 长度²
Σ_2d       | (N,2,2)        | 图像平面协方差               | 像素²
J          | (N,2,3)        | 透视投影雅可比矩阵           | 1/长度
z          | (N,)           | 深度值（相机坐标系Z）        | 世界坐标
depth      | (N,)           | 深度排序依据                 | 世界坐标
N          | 标量           | 高斯数量                     | 无单位
H, W       | 标量           | 图像高、宽                   | 像素
tile_size  | 标量（默认16） | Tile大小                    | 像素
```

---

## 一、问题的本质：我们要构建什么？

### 1.1 输入与输出

**输入**是什么？你有一堆3D高斯，每个高斯有：

```
G_i = {μ_i, Σ_i, α_i, c_i}
  μ_i: 3D位置 (x,y,z)  [维度: (3,), 单位: 世界坐标]
  Σ_i: 3×3协方差矩阵（对称正定） [维度: (3,3), 单位: 长度²]
  α_i: 透明度 [0,1] [维度: (1,), 无单位]
  c_i: 颜色 (RGB) [维度: (3,), 范围: [0,1]]
```

**输出**是什么？一张2D图像：

```
C[x,y] ∈ [0,1]³  # 每个像素的颜色 [维度: (H,W,3)]
```

**问题**：从 `{μ_i, Σ_i, α_i, c_i}` 到 `C[x,y]`，中间经过什么步骤？

---

## 一、问题的本质：我们要构建什么？

### 1.1 输入与输出

**输入**是什么？你有一堆3D高斯，每个高斯有：

```
G_i = {μ_i, Σ_i, α_i, c_i}
  μ_i: 3D位置 (x,y,z)
  Σ_i: 3×3协方差矩阵（对称正定）
  α_i: 透明度 [0,1]
  c_i: 颜色 (RGB)
```

**输出**是什么？一张2D图像：

```
C[x,y] ∈ [0,1]³  # 每个像素的颜色
```

**问题**：从 `{μ_i, Σ_i, α_i, c_i}` 到 `C[x,y]`，中间经过什么步骤？

---

### 1.2 约束条件：不能随便设计

你可能会想：这不简单吗？对每个像素，遍历所有高斯，算贡献，加起来。

**但这里有硬约束**：

1. **可微分**：必须能从 C 反向传播梯度到 μ, Σ, α, c
2. **可扩展**：N=1M 高斯时不能太慢
3. **质量**：不能有锯齿、空洞
4. **稀疏性**：只能计算"相关"的高斯，不能每个像素遍历所有高斯

**这些约束同时存在，缺一不可。** 这就是你的设计难题。

---

## 二、Axioms：从第一性原理出发

让我们从**不可约的基础事实**开始，一步步推导出完整管线。

**公理1：透视投影是必须的**
- 相机模型固定：小孔成像
- 3D点 (X,Y,Z) → 2D像素 (x,y) = (X/Z, Y/Z)（归一化坐标）
- 加上内参矩阵 K 得到像素坐标

**公理2：渲染需要处理半透明叠加**
- 高斯不是实体表面，是"云"一样的半透明物质
- Alpha blending 是标准做法：`C += α·color·(1-A_accum)`
- 必须从后往前混合（深度排序）

**公理3：计算必须稀疏**
- 每个高斯只影响它附近的像素（μ ± 3σ）
- 不能对每个像素遍历所有高斯（O(H·W·N) 不可接受）

**公理4：所有操作必须可微**
- 投影、排序、混合都必须是可微函数
- 梯度才能流回参数进行优化

**公理5：质量要求连续**
- 不能有硬边界（如立方体投影的六边形边缘）
- 需要连续的概率分布

---

## 三、Contradictions：约束之间的张力

现在看看这些公理之间有什么矛盾？

**矛盾1：稀疏性 vs 全局排序**
- 稀疏性要求：只处理每个像素附近的有限高斯
- 但排序要求：必须知道所有高斯的深度，从后往前混合
- **问题**：如果我只选每个像素附近的高斯，怎么保证全局深度顺序？

**矛盾2：可微分 vs 排序**
- Alpha blending 需要从后往前混合
- 但 `argsort(depth)` 是离散操作，梯度是什么？
- **问题**：排序不可微，怎么反向传播？

**矛盾3：快速投影 vs 协方差计算**
- 每个高斯投影需要：μ_2d = K·(R·μ + t)，Σ_2d = J·(R·Σ·Rᵀ)·Jᵀ
- Σ_2d 涉及矩阵乘法和雅可比计算
- 如果 N=1M，每个都算 Σ_2d = O(N) 矩阵运算，会不会慢？

**矛盾4：稀疏筛选 vs 边界情况**
- 高斯影响范围是 μ ± 3√(max_eigenvalue(Σ_2d))
- 但 Σ_2d 还没算，怎么知道高斯影响多大？
- **问题**：先算 Σ_2d 再筛选？还是先筛选再算 Σ_2d？

这些矛盾就是你要解决的核心设计问题。每一步选择都在trade-off。

---

## 四、Solution Path：逐步化解矛盾

### 4.1 矛盾1的解决：两阶段渲染

**问题**：既要稀疏性（只处理相关高斯），又要全局排序（保证混合正确）。

**洞察**：排序只需要**在屏幕空间**正确，不需要全局3D空间。

**解决方案**：两阶段设计

```
阶段1: 投影 + 全局排序
  所有高斯 N 个 → 计算 μ_2d, depth → argsort → 得到全局2D深度顺序

阶段2: Tile-based 渲染
  将屏幕划分为 16×16 的tile
  对每个tile：
    找出所有影响该tile的高斯（由 μ_2d 和 Σ_2d 的包围盒决定）
    按全局顺序只遍历这些高斯进行混合
```

**为什么可行？**
- 阶段1是 O(N)，但只做一次
- 阶段2每个tile独立，只处理"相关"高斯
- 全局排序在阶段1完成，阶段2直接使用

**需要细化吗？** Tile映射的具体实现（倒排索引）想展开吗？

---

### 4.2 矛盾2的解决：训练 vs 推理的不同策略

**问题**：排序不可微，但训练需要梯度。

**关键洞察**：
- **训练**：梯度有效是因为高斯参数变化缓慢，深度顺序基本稳定
- **推理**：排序只做一次，不影响可微性

**训练时的技巧**：
- 每N次迭代（比如每1000步）才重新排序一次
- 期间顺序固定，梯度有效
- 如果顺序变化太大（高斯移动很多），说明需要重排

**更高级的方案**（可选）：
- 软化排序：用可微排序函数（如 softsort）代替 argsort
- 代价：速度稍慢，但梯度更稳定

**你需要哪种？** 想深入了解软化排序的实现吗？

---

### 4.3 矛盾3的解决：协方差投影优化

**问题**：Σ_2d = J·(R·Σ·Rᵀ)·Jᵀ 涉及多次矩阵乘法，N=1M时会慢。

**优化策略**：

**策略1：预计算旋转部分**
```
R·Σ·Rᵀ 只依赖于 R 和 Σ，不依赖于 J（相机内参）
如果相机不变（静态场景），可以缓存
```

**策略2：避免显式矩阵乘法**
```
协方差投影可以展开为：
Σ_2d[0,0] = (K00²/z²)·Σ_cam[0,0] + 交叉项
...
可以推导出简化公式，减少乘法次数
```

**策略3：数值稳定性优先**
```
虽然慢一点，但 Cholesky 分解比直接求逆更稳定
实际3DGS实现：用 Cholesky 或特征值分解
```

**性能数据**（N=1M，RTX 4090）：
- 投影（含 Σ_2d）: 2-5 ms ✅ 可接受
- 排序: 5-10 ms
- Tile筛选: 1-2 ms
- 渲染: 5-15 ms
- **总计**: 13-32 ms（30-75 FPS）🔥

**需要展开**：协方差投影的详细推导（雅可比矩阵如何构造）？

---

### 4.4 矛盾4的解决：包围盒的近似计算

**问题**：要筛选tile，需要知道每个高斯影响多大（包围盒），但包围盒依赖 Σ_2d。

**解决方案**：两阶段投影

```
阶段A: 计算 Σ_2d（所有高斯）
  此时还没有排序，但需要全部计算

阶段B: 用 Σ_2d 计算包围盒
  bbox = μ_2d ± 3·√(max_eigenvalue(Σ_2d))

阶段C: Tile映射
  用 bbox 决定每个高斯影响哪些tile
```

**为什么这是O(N)？**
- 阶段A: O(N) 矩阵运算（前面说了2-5ms）
- 阶段B: O(N) 特征值计算（但2×2矩阵很快）
- 阶段C: O(N) 遍历每个高斯的包围盒

**总筛选时间**: 1-2 ms ✅

**关键点**：筛选在排序之后，但 Σ_2d 计算在排序之前。顺序是：
1. 所有高斯投影（μ_2d, Σ_2d, depth）
2. 排序
3. 用 Σ_2d 算包围盒
4. Tile映射
5. 渲染

**需要细化**：2×2矩阵特征值的快速计算（避免通用eigh）？

---

## 五、完整管线：从输入到输出

现在把各阶段串联起来，看完整流程。

### 5.1 阶段1：投影（所有高斯）

**输入**：N个高斯 {μ_i, Σ_i, α_i, c_i}，相机 {K, R, T}

**输出**：N个2D高斯 {μ_2d_i, Σ_2d_i, depth_i, α_i, c_i}

**计算**：

```python
# 变量说明:
# mu: (N,3) 3D高斯中心（世界坐标）
# Sigma: (N,3,3) 3D协方差矩阵
# R: (3,3) 相机旋转矩阵（世界→相机）
# T: (3,) 相机平移向量（世界→相机）
# K: (3,3) 相机内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]
# N: 高斯数量

# 1. 世界坐标系 → 相机坐标系
# mu_cam = R·μ + T，广播T到(N,3)
mu_cam = (R @ mu.T).T + T  # 输出: (N,3)，相机坐标系下的3D位置

# 2. 相机坐标系 → 像素坐标（透视投影）
# 齐次坐标: [X·Z, Y·Z, Z]ᵀ = K·[X, Y, Z]ᵀ
mu_hom = (K @ mu_cam.T).T  # (N,3)，齐次坐标前两维 = x·z, y·z
mu_2d = mu_hom[:, :2] / mu_hom[:, 2:3]  # (N,2)，像素坐标 (x, y) = (X·Z/Z, Y·Z/Z)
depth = mu_hom[:, 2]  # (N,)，深度值Z（用于排序）

# 3. 3D协方差转换到相机坐标系
# Σ_cam = R·Σ·Rᵀ，保持协方差性质
Sigma_cam = R @ Sigma @ R.T  # 输出: (N,3,3)

# 4. 计算透视投影的雅可比矩阵 J = ∂(x_2d)/∂(x_c)
# x = fx·X/Z + cx, y = fy·Y/Z + cy
# ∂x/∂X = fx/Z, ∂x/∂Y = 0, ∂x/∂Z = -fx·X/Z²
# ∂y/∂X = 0, ∂y/∂Y = fy/Z, ∂y/∂Z = -fy·Y/Z²
z = mu_cam[:, 2].clamp(min=1e-6)  # (N,)，深度裁剪，避免除零错误（数值稳定）
J = torch.zeros((N, 2, 3), device=mu.device)  # (N,2,3) 雅可比矩阵
J[:, 0, 0] = K[0, 0] / z  # ∂x/∂X = fx / z
J[:, 0, 2] = -K[0, 0] * mu_cam[:, 0] / (z**2)  # ∂x/∂Z = -fx·X/z²
J[:, 1, 1] = K[1, 1] / z  # ∂y/∂Y = fy / z
J[:, 1, 2] = -K[1, 1] * mu_cam[:, 1] / (z**2)  # ∂y/∂Z = -fy·Y/z²

# 5. 2D协方差投影（核心公式）
# Σ_2d = J · Σ_cam · Jᵀ，利用协方差变换性质
Sigma_2d = J @ Sigma_cam @ J.transpose(1, 2)  # 输出: (N,2,2)

# 6. 数值稳定化（避免奇异矩阵导致求逆爆炸）
# 添加一个小的对角正则项，确保所有特征值 > 1e-8
Sigma_2d = Sigma_2d + torch.eye(2, device=mu.device)[None,:,:] * 1e-8
```

**复杂度**：O(N)，2-5ms（N=1M）

---

### 5.2 阶段2：深度排序

```python
sorted_indices = torch.argsort(depth, descending=True)

mu_2d = mu_2d[sorted_indices]
Sigma_2d = Sigma_2d[sorted_indices]
alpha = alpha[sorted_indices]
color = color[sorted_indices]
```

**复杂度**：O(N log N)，5-10ms

---

### 5.3 阶段3：Tile预筛选

**目标**：建立 `tile_id → [gaussian_indices]` 的映射

```python
# 1. 计算包围盒
eigvals = torch.linalg.eigvalsh(Sigma_2d)  # (N,2)
radii = 3.0 * torch.sqrt(eigvals.max(dim=1)[0])  # (N,)
bbox_min = (mu_2d - radii[:, None]).long().clamp(min=0)
bbox_max = (mu_2d + radii[:, None]).long().clamp(max=W)

# 2. Tile划分
tile_size = 16
n_tiles_x = (W + tile_size - 1) // tile_size
n_tiles_y = (H + tile_size - 1) // tile_size
n_tiles = n_tiles_x * n_tiles_y
tile_mapping = [[] for _ in range(n_tiles)]

# 3. 分配高斯到tile
for g in range(N):
    x0, y0 = bbox_min[g]
    x1, y1 = bbox_max[g]
    tile_x0, tile_x1 = x0 // tile_size, x1 // tile_size
    tile_y0, tile_y1 = y0 // tile_size, y1 // tile_size
    for ty in range(tile_y0, tile_y1+1):
        for tx in range(tile_x0, tile_x1+1):
            tile_id = ty * n_tiles_x + tx
            if 0 <= tile_id < n_tiles:
                tile_mapping[tile_id].append(g)

# 4. 构建连续数组（便于GPU kernel）
tile_start = []
tile_count = []
all_gaussians = []
for tile_id in range(n_tiles):
    tile_start.append(len(all_gaussians))
    tile_count.append(len(tile_mapping[tile_id]))
    all_gaussians.extend(tile_mapping[tile_id])
```

**复杂度**：O(N + total_overlap)，1-2ms

**关键洞察**：总重叠数 ≈ N × (平均每个高斯影响的tile数) ≈ N × 10-20 ✅

---

### 5.4 阶段4：Alpha Blending（渲染）

**伪代码**（带变量注释）：

```
# 初始化输出缓冲
C = zeros(H, W, 3)  # 输出图像 C[x,y] ∈ [0,1]³，维度 (H,W,3)
A = zeros(H, W)      # 累计不透明度 A[x,y] ∈ [0,1]，维度 (H,W)

# Tile遍历（每个tile独立）
for tile_y in tiles_y:
  for tile_x in tiles_x:
    tile_id = tile_y * n_tiles_x + tile_x
    gaussians = tile_mapping[tile_id]  # 该tile内的高斯索引列表（已按深度排序）
    
    # 遍历tile内每个像素
    for pixel in tile:
      x, y = pixel坐标  # (int, int)，像素坐标
      T = 1.0  # 透射率（剩余透明度），标量 ∈ [0,1]，初始=1（全透射）
      
      # 遍历影响该像素的高斯（从后到前，远→近）
      for g in gaussians:
        # 2D高斯评估（计算该高斯在该像素的贡献）
        diff = (x - mu_2d[g,0], y - mu_2d[g,1])  # (2,)，像素到高斯中心的偏移
        # exponent = -½·Δᵀ·Σ⁻¹·Δ，标量（负值，绝对值大表示远离中心）
        exponent = -0.5 * (diff @ inv(Sigma_2d[g]) @ diff)
        g_val = alpha[g] * exp(exponent)  # 高斯值 = α·exp(exponent) ∈ [0,α]
        
        # Alpha blending（前向混合公式）
        # 当前高斯贡献 = 当前透射率 × 高斯值 × 颜色
        C[x,y] += T * g_val * color[g]  # color[g] ∈ [0,1]³
        A[x,y] += g_val  # 累计不透明度增加
        T *= (1 - g_val)  # 更新透射率：光线被吸收的比例
        
        if T < 0.01:  # 早停条件：剩余透射率<1%时，后续高斯贡献可忽略
          break
```

**变量含义总结**:
- `C`: 最终像素颜色（RGB）
- `A`: 累计不透明度（已吸收光线比例）
- `T`: 剩余透射率（未吸收光线比例），满足 T = Π(1-α_i)（已处理高斯）
- `g_val`: 单个高斯的有效贡献值（∈ [0,α]）
- `diff`: 像素位置与高斯中心的2D偏移（像素单位）
- `exponent`: 2D高斯指数项（负值，绝对值>3时贡献<5%）

**早停原理**:
- 当 T < 0.01 时，意味着99%的光线已被吸收
- 后续高斯即使有贡献，也小于当前像素的1%
- 视觉上不可见，提前终止节省计算

**复杂度分析**:
- 每个像素平均遍历 k ≈ 20-50 个高斯（早停效果）
- 总操作数 ≈ H·W·k = 1200×900×50 = 54M 次高斯评估
- 实际时间: 5-15ms（RTX 4090）
```

**复杂度**：每个像素约 20-50 次高斯评估（早停）
总操作数 ≈ H·W·50 = 1200×900×50 = 54M，5-15ms ✅

---

## 六、可微分性验证

### 6.1 梯度流路径

```
损失 L
  ↓ ∂L/∂C
C ← 混合 ← g_val ← 2D高斯 ← Σ_2d, μ_2d ← 投影 ← Σ, μ
  ↑               ↑              ↑
  └─── 混合可微 ───┘              └─── 矩阵运算可微 ───┘
```

**每个操作的可微性**：

| 操作 | 公式 | 可微? | 梯度路径 |
|------|------|-------|----------|
| 投影(μ_2d) | K·(R·μ+T) | ✅ | ∂μ_2d/∂μ = K·R |
| 投影(Σ_2d) | J·(R·Σ·Rᵀ)·Jᵀ | ✅ | 矩阵求导 |
| 高斯评估 | exp(-½ΔᵀΣ⁻¹Δ) | ✅ | 链式法则 |
| Alpha blending | C += T·g_val·c | ✅ | 线性 |
| 排序 | argsort | ❌ | 但训练时稳定 |

**排序问题**：argsort 不可微，但：
- 训练时每K步才重排，期间顺序固定
- 顺序变化缓慢时，梯度仍有效
- 或使用软化排序（soft-rank）作为替代

**需要展开**：Σ_2d 对 Σ 的梯度推导（矩阵求导细节）？

---

### 6.2 数值稳定性检查

**哪些地方容易炸？**

1. **z接近0**：J 矩阵很大
   - 解决：`z = mu_cam[:,2].clamp(min=1e-6)`

2. **Σ_2d 接近奇异**：特征值很小，求逆爆炸
   - 解决：`Sigma_2d += I·1e-8`
   - 或使用 Cholesky：`L = cholesky(Sigma_2d)`，然后 `solve(L, diff)`

3. **指数溢出**：`exp(-½ΔᵀΣ⁻¹Δ)` 当 Δ 很大时，exponent 负很大 → 下溢
   - 解决：用 `log1p` 或分块计算
   - 实际上，早停策略已经过滤了远距离高斯

**实际实现**：3DGS官方代码使用了 Cholesky + 早停，非常稳定。

---

## 七、性能优化：为什么能到30-75 FPS？

### 7.1 复杂度分析

**原始复杂度（无优化）**：
```
每像素遍历所有高斯: O(H·W·N)
H=1200, W=900, N=1M → 1.08×10¹² 次操作 ❌
```

**优化后**：
```
阶段1 (投影): O(N) = 1M
阶段2 (排序): O(N log N) ≈ 20M
阶段3 (Tile筛选): O(N) = 1M + 重叠 ≈ 10M
阶段4 (渲染): O(H·W·k) where k=平均每个像素遍历的高斯数 ≈ 50
               = 1200×900×50 = 54M

总计: ~85M 次操作 ✅
加速比: 1.08×10¹² / 85×10⁶ ≈ 12,700 倍!
```

**关键优化**：Tile预筛选将复杂度从 O(H·W·N) 降到 O(H·W·k)，其中 k << N。

---

### 7.2 Tile预筛选的威力

**为什么能减少99%计算？**

- 每个高斯只影响约 100 像素（μ ± 3σ）
- 总像素数 H·W = 1,080,000
- 每个像素平均被多少高斯影响？
  ```
  总影响次数 = N × 100 = 1M × 100 = 100M
  平均每像素 = 100M / 1.08M ≈ 93 高斯
  ```
- 而原始方案是每个像素遍历 1M 高斯
- **减少倍数**: 1M / 93 ≈ 10,700 倍!

** Tile预筛选是3DGS速度的核心**，没有它，实时渲染不可能。

**需要展开**：如何用CUDA并行实现Tile映射和渲染？

---

## 八、完整管线的第一性原理总结

### 8.1 从公理到实现的推导链

```
Axioms:
  1. 透视投影必须
  2. 半透明需要从后往前混合
  3. 计算必须稀疏
  4. 所有操作可微
  5. 质量需要连续

→ Contradictions:
  - 稀疏性 vs 全局排序
  - 可微 vs 排序
  - 快速投影 vs 协方差计算
  - 筛选 vs 边界未知

→ Solution Path:
  1. 两阶段：投影排序（全局）→ Tile渲染（局部）
  2. 训练：固定排序区间；推理：一次排序
  3. 协方差优化：预计算、简化公式、数值稳定
  4. 两阶段筛选：先算Σ_2d，再算包围盒

→ 完整管线:
  投影 → 排序 → Tile映射 → Alpha Blending

→ 性能:
  O(N) + O(N log N) + O(N) + O(H·W·k)
  ≈ 85M ops (N=1M) → 13-32 ms
```

**这就是"从问题到方案"的完整推导**。每一步选择都是权衡的结果。

---

### 8.2 为什么这个管线是"唯一合理"的？

**你能想到其他设计吗？**

**方案A：先Tile筛选，再投影**
- 问题：不知道 Σ_2d，无法算包围盒
- 除非在3D空间筛选，但3D投影后才知2D影响范围 ❌

**方案B：每帧动态排序，不缓存**
- 问题：排序 O(N log N) 每帧，如果N=1M，5-10ms，可接受
- 但训练时梯度需要稳定顺序，动态排序梯度混乱 ❌

**方案C：不用Tile，用空间数据结构（如BVH）**
- 可能，但构建BVH本身 O(N log N)
- Tile预筛选更简单，且GPU友好（tile是天然并行单元）✅

**方案D：ray marching（像NeRF）**
- 质量好，但速度慢（每像素 dozens/hundreds samples）
- 违反"快速投影"公理 ❌

**结论**：当前管线是约束下的最优解。

---

## 九、思考题（第一性原理式）

**1. 你能想到比Tile预筛选更快的筛选方法吗？**
- 提示：思考"高斯影响范围的统计特性"
- 如果高斯分布很稀疏（大部分在画面外），筛选能否进一步优化？

**2. 排序的梯度问题真的不可解吗？**
- 尝试：如果深度差很小（高斯几乎同深度），梯度应该"混合"
- 如果深度差很大，顺序几乎不变
- 你能设计一个"可微排序"吗？参考 `torch.sort` 的反向传播

**3. 如果内存不够（N=10M高斯），管线哪部分会爆？**
- 投影：O(N) 内存，10M×13 floats ≈ 520 MB ✅ 可能
- Tile映射：每个高斯影响10-20 tiles，总映射条目 ≈ 100-200M，需要4字节索引 → 400-800 MB ❌
- 渲染：每个tile的高斯列表可能很大
- **问题**：内存瓶颈在Tile映射，如何流式处理？

**4. 如果场景动态（高斯移动），Tile映射需要每帧重建吗？**
- Tile映射 O(N)，每帧重建 1-2ms，可接受
- 但内存带宽：每帧重写 tile_mapping 数组，可能成为瓶颈
- 你能想到增量更新Tile映射的方法吗？（高斯移动小，只更新受影响tile）

---

## 十、本章核心记忆点

✅ **渲染管线四阶段**：投影 → 排序 → Tile筛选 → Alpha Blending

✅ **复杂度**：O(N) + O(N log N) + O(N) + O(H·W·k) ≈ 85M ops → 13-32ms

✅ **Tile预筛选**：将计算从 O(H·W·N) 降到 O(H·W·k)，加速10,000倍

✅ **可微分**：除排序外全可微；排序在训练时通过稳定策略解决

✅ **数值稳定性**：z clamp、Σ正则化、Cholesky分解

✅ **早停**：累计α≥0.99停止，每像素只遍历20-50高斯

**下一章**：我们将深入优化目标——L1+SSIM损失如何设计？为什么需要额外的正则化？densify & prune 如何实现可微的"结构学习"？

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