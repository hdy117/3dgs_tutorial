# 第8章：Feed-Forward推理 - 实时渲染

**学习路径**：`verification → example`

**核心目标**：训练完成后，如何实现实时渲染（<10ms/帧）

---

## 一、引言：从训练到部署

### 1.1 训练 vs 推理对比

| 阶段 | 训练 | 推理 |
|------|------|------|
| 参数 | 可更新 | 固定 |
| 密度控制 | 每N步densify/prune | 无 |
| 梯度 | 需要保留 | 不需要 |
| 排序 | 每帧计算 | 可缓存（相机移动慢） |
| 渲染 | 完整管线 | 相同，但可极致优化 |
| 目标速度 | ~200ms/帧（含反向） | **< 10ms/帧**（60 FPS） |

### 1.2 速度差距

- 训练：~200 ms/frame（含反向传播）
- 推理：**< 10 ms/frame**（实时60 FPS）
- **加速比：20倍以上**

**核心优化方向**：
1. 去掉反向传播
2. 预排序缓存
3. Tile极致优化
4. 降低精度（float16）
5. 融合kernel

---

## 二、推理流程（与训练对比）

### 2.1 核心管线不变

```
输入：{ (μ, Σ, α, c) } + 相机
1. 投影所有高斯 → μ_2D, Σ_2D, depth
2. 按depth排序
3. Alpha blending 累加
4. 输出图像
```

### 2.2 可以去掉的部分

- ❌ 反向传播（no autograd）
- ❌ 梯度计算
- ❌ 优化器step
- ❌ 密度控制（无densify/prune）
- ❌ 损失计算

### 2.3 可以简化的部分

- ✅ 梯度保留：推理时可关闭梯度（`torch.no_grad()`）
- ✅ 排序策略：可缓存深度顺序
- ✅ 精度：可降低到float16
- ✅ 随机性：训练可能有随机采样，推理需确定性

---

## 三、推理优化策略

### 3.1 预排序缓存

**观察**：如果相机移动小（如游戏帧间位移 < 0.1m），高斯深度顺序几乎不变

**优化**：
- 初始帧：计算所有高斯的深度顺序，存储索引数组 `sorted_indices`
- 后续帧：复用 `sorted_indices`，除非相机移动超过阈值
- 若相机位移大，重新排序

**阈值判断**：
```python
def need_resort(prev_camera, curr_camera, threshold=0.1):
    """相机位移超过阈值则重排"""
    delta_t = torch.norm(curr_camera.T - prev_camera.T)
    return delta_t > threshold
```

**速度收益**：
- 排序 O(N log N) → 免去，节省 ~5-10 ms（N=1M）

---

### 3.2 Tile-based 并行优化（极致版）

**训练时已有tile优化，推理需极致化**：

#### 3.2.1 Tile大小选择

| Tile大小 | 内存占用 | 并行度 | 适用场景 |
|---------|----------|--------|----------|
| 8×8 | 低 | 高 | 小高斯数量 |
| 16×16 | 中 | 中 | **推荐** |
| 32×32 | 高 | 低 | 大高斯数量 |

**推荐**：16×16（平衡）

---

#### 3.2.2 高斯预筛选（Pre-filter）

**问题**：如何快速知道哪些高斯影响某个tile？

**方案**：投影时同时计算包围盒

```python
def compute_bbox(mu_2d, Sigma_2d, scale=3.0):
    """计算高斯在屏幕上的包围盒"""
    # 协方差的特征值 → 半轴长度
    eigvals = torch.linalg.eigvalsh(Sigma_2d)  # (N, 2)
    radii = scale * torch.sqrt(eigvals.max(dim=1)[0])  # (N,)
    bbox_min = (mu_2d - radii[:, None]).long().clamp(min=0)
    bbox_max = (mu_2d + radii[:, None]).long().clamp(max=W)  # W: 图像宽
    return bbox_min, bbox_max  # (N, 2)
```

**预筛选流程**：
1. 投影所有高斯，得到 mu_2d, Sigma_2d, depth
2. 计算每个高斯的包围盒 (x_min, y_min, x_max, y_max)
3. 根据包围盒，将高斯分配给对应的tile（倒排索引）
4. 每个tile只加载其关联的高斯

---

#### 3.2.3 CUDA Kernel设计（简化）

```cpp
// 假设：已排序的高斯列表，每个tile的高斯列表已预计算

__global__ void render_kernel(
    Gaussian* gaussians,  // 已排序
    int N,
    Tile* tiles,         // 每个tile的高斯索引列表
    int num_tiles,
    uchar4* image
) {
    int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_id >= num_tiles) return;

    Tile tile = tiles[tile_id];
    int x0 = tile.x * TILE_SIZE;
    int y0 = tile.y * TILE_SIZE;

    __shared__ Gaussian shared_gaussians[MAX_GAUSSIANS_PER_TILE];
    __shared__ int shared_count;

    // 加载该tile的高斯到shared memory
    if (threadIdx.x < tile.gaussian_count) {
        shared_gaussians[threadIdx.x] = gaussians[tile.gaussian_indices[threadIdx.x]];
    }
    __syncthreads();
    shared_count = tile.gaussian_count;

    // 每个线程处理一个像素（tile内）
    int px = x0 + (threadIdx.x % TILE_SIZE);
    int py = y0 + (threadIdx.x / TILE_SIZE);
    if (px >= W || py >= H) return;

    float3 color = make_float3(0, 0, 0);
    float alpha_acc = 0.0f;

    // 遍历该tile的所有高斯
    for (int i = 0; i < shared_count; i++) {
        Gaussian& g = shared_gaussians[i];
        // 计算高斯值
        float g_val = evaluate_gaussian_2d(g.mu_2d, g.Sigma_2d, px, py);
        g_val *= g.alpha;

        // Alpha blending
        color.x += g_val * g.color.x * (1 - alpha_acc);
        color.y += g_val * g.color.y * (1 - alpha_acc);
        color.z += g_val * g.color.z * (1 - alpha_acc);
        alpha_acc += g_val;

        if (alpha_acc >= 0.99f) break;  // 早停
    }

    // 写入输出
    image[py * W + px] = make_uchar4(
        color.x * 255,
        color.y * 255,
        color.z * 255,
        255
    );
}
```

---

### 3.3 精度优化：float16

**训练**：float32（或混合精度float16）
**推理**：
- 可安全使用 **float16**（半精度），速度翻倍（Tensor Core）
- 但注意：
  - 协方差矩阵求逆可能不稳定（float16范围小）
  - 实际：投影用 float32，评估用 float16
  - 或：全部用 float32，但用Tensor Core加速（NVIDIA GPU支持float32 Tensor Core）

**实现**：
```python
# 转换高斯到half
gaussians.half()

with torch.no_grad(), torch.cuda.amp.autocast():
    rendered = render(gaussians, camera)
```

---

### 3.4 动态分辨率适配

**场景**：渲染到不同分辨率（如1080p → 4K）

**问题**：高斯尺度不变，但像素更多 → 需要更多高斯才能填满

**优化**：
- 按分辨率缩放高斯尺度：
  ```python
  scale_factor = new_res / train_res
  gaussians.mu *= scale_factor  # 位置不变
  gaussians.Sigma *= scale_factor**2  # 协方差缩放平方
  ```
- 或在训练时用多分辨率（随机crop），让高斯适应不同尺度

---

## 四、性能基准

### 4.1 典型配置（RTX 4090）

| 场景 | 高斯数量 | 分辨率 | FPS | 延迟 |
|------|----------|--------|-----|------|
| 小型室内 | 0.5M | 800×600 | 150+ | < 7 ms |
| 中型场景 | 1.5M | 1200×900 | 80-100 | 10-12 ms |
| 大型外景 | 3M | 1920×1080 | 40-60 | 16-25 ms |

**瓶颈分析**：
- N < 1M：内存带宽（读写高斯数据）
- N > 2M：计算（投影 + 评估）
- 分辨率：像素数增加 → 每个像素处理时间增加

---

### 4.2 优化手段对比

| 优化 | 速度提升 | 实现难度 |
|------|----------|----------|
| Tile-based 并行 | 5-10× | 中 |
| 预排序缓存 | 1.2-1.5× | 低 |
| float16 精度 | 1.5-2× | 低 |
| 融合kernel | 1.1-1.3× | 高 |
| **总计** | **可达 20-50×** | 高 |

---

## 五、与训练pipeline的差异总结

### 5.1 不需要的部分

- ❌ 反向传播
- ❌ 梯度计算
- ❌ 优化器step
- ❌ 密度控制
- ❌ 损失计算

### 5.2 可以简化的部分

- ✅ 关闭autograd：`torch.no_grad()`
- ✅ 缓存排序结果
- ✅ 使用float16
- ✅ 确定性渲染（无随机性）

---

## 六、实时渲染应用

### 6.1 AR/VR集成

**需求**：
- 低延迟（< 20ms 端到端）
- 高帧率（90/120 FPS）
- 6DoF 头部追踪

**方案**：
- 预训练高斯场景（离线）
- 运行时：根据头部位姿实时渲染
- 压缩高斯数据（量化、压缩）以适应移动端内存

---

### 6.2 自动驾驶仿真

**需求**：
- 快速生成新视角（用于传感器模拟：相机、LiDAR）
- 可编辑（添加/删除物体）

**方案**：
- 使用3DGS表示静态场景
- 动态物体用单独的3DGS或传统模型
- 渲染时融合

---

## 七、推理代码框架

```python
class GaussianSplattingInference:
    def __init__(self, ply_path):
        # 加载训练好的高斯
        self.gaussians = load_ply(ply_path)
        self.gaussians = self.gaussians.half().cuda()  # 转half
        self.sorted_indices = None
        self.last_camera = None

    def render(self, camera, need_sort=True):
        """
        camera: dict{R, T, K, height, width}
        need_sort: 是否重新排序（相机移动大时设为True）
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            # 1. 检查是否需要重排
            if need_sort and self.last_camera is not None:
                if self.need_resort(self.last_camera, camera):
                    self.sorted_indices = None
            self.last_camera = camera

            # 2. 投影（可缓存mu_2d, Sigma_2d）
            mu_2d, Sigma_2d, depth = project_gaussians(
                self.gaussians.mu, self.gaussians.Sigma,
                camera.R, camera.T, camera.K
            )

            # 3. 排序（如需要）
            if self.sorted_indices is None:
                self.sorted_indices = torch.argsort(depth, descending=True)

            # 4. 预计算包围盒 + tile分配
            bbox_min, bbox_max = compute_bbox(mu_2d, Sigma_2d)
            tile_mapping = assign_gaussians_to_tiles(bbox_min, bbox_max)

            # 5. Tile-based渲染
            image = render_tiled(
                self.gaussians, self.sorted_indices,
                mu_2d, Sigma_2d, tile_mapping,
                camera.height, camera.width
            )

            return image.clamp(0, 1)  # 确保在[0,1]

    def need_resort(self, prev_cam, curr_cam, threshold=0.1):
        delta_t = torch.norm(curr_cam['T'] - prev_cam['T'])
        return delta_t > threshold
```

---

## 八、思考题（独立重推检验）

1. **缓存的意义**：如果相机每帧随机移动，预排序缓存还有效吗？如何设计fallback策略？
2. **精度与速度**：float16 在3DGS渲染中哪里可能不稳定？如何检测和修复？
3. **内存瓶颈**：假设N=3M高斯，每个高斯13 float（52字节），总内存 ~156MB。如果tile缓存1000高斯，shared memory需要多少？够吗（GPU shared memory通常100KB左右）？
4. **扩展性**：如果要渲染8K分辨率（7680×4320），高斯数量需要增加多少？为什么？

---

## 九、下一章预告

**第9章**：扩展与变体 - 3DGS的后续发展：动态场景（4D Gaussian）、压缩传输、几何质量提升、与NeRF结合。

---

**关键记忆点**：
- ✅ 推理核心：去掉梯度 + 缓存排序 + tile优化 + float16
- ✅ 预排序缓存：相机移动小时复用深度顺序
- ✅ Tile大小：16×16像素（推荐）
- ✅ 性能：1.5M高斯 @ 1080p → 80-100 FPS（RTX 4090）
- 🎯 **目标**：< 10ms/帧（60+ FPS）
