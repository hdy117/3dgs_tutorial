# 第8章：Feed-Forward推理 - 实时渲染

**学习路径**：`verification → example`

---

## 引言：从训练到部署

**训练**：每帧都要计算梯度、更新参数、调整密度
**推理**：参数固定，只做前向投影 + Alpha Blending

**速度差异**：
- 训练：~200 ms/frame（含反向传播）
- 推理：**< 10 ms/frame**（实时60 FPS）

本章揭示推理pipeline的优化技巧。

---

## 1. 推理流程（与训练对比）

### 训练 vs 推理

| 阶段 | 训练 | 推理 |
|------|------|------|
| 参数 | 可更新 | 固定 |
| 密度控制 | 每N步densify/prune | 无 |
| 梯度 | 需要保留 | 不需要 |
| 排序 | 每帧计算 | 可缓存（如果相机移动慢） |
| 渲染 | 完整管线 | 相同，但可极致优化 |

**核心管线不变**：
```
输入：{ (μ, Σ, α, c) } + 相机
1. 投影所有高斯 → μ_2D, Σ_2D, depth
2. 按depth排序
3. Alpha blending 累加
4. 输出图像
```

---

## 2. 推理优化策略

### 2.1 预排序缓存

**观察**：如果相机移动小（如游戏帧间位移 < 0.1m），高斯深度顺序几乎不变

**优化**：
- 初始帧：计算所有高斯的深度顺序，存储索引数组 `sorted_indices`
- 后续帧：复用 `sorted_indices`，除非相机移动超过阈值
- 若相机位移大，重新排序

**速度收益**：
- 排序 O(N log N) → 免去，节省 ~5-10 ms（N=1M）

---

### 2.2 Tile-based 并行优化

**训练时已有tile优化，推理需极致化**：

**CUDA实现细节**：
- Tile大小：16×16 或 32×32 像素
- 每个block处理一个tile
- 共享内存缓存该tile涉及的高斯（通常 < 1000个）
- 线程内循环高斯，原子操作累加颜色

**内存访问优化**：
- 高斯数据：连续读取（coalesced）
- 图像写入：block内线程写相邻像素 → 合并访问

---

### 2.3 投影与评估的融合

**训练做法**：先计算所有高斯的 μ_2D 和 Σ_2D，存储数组，再渲染

**推理优化**：
- 不存储中间结果，在渲染kernel内直接计算投影并评估
- 减少全局内存读写（一次遍历完成）

但要注意：如果做tile预筛选，仍需知道每个高斯的影响范围 → 需要预先计算包围盒

**折中方案**：
1. 第一遍：只计算 μ_2D 和包围盒，不计算 Σ_2D
2. 根据包围盒，将高斯分配到tile
3. 第二遍：对每个tile，加载其高斯，计算 Σ_2D 和像素值

这是训练的标准做法，推理沿用。

---

### 2.4 数值精度

**训练**：float32（或混合精度float16）
**推理**：
- 可安全使用 **float16**（半精度），速度翻倍（Tensor Core）
- 但注意：
  - 协方差矩阵求逆可能不稳定（float16范围小）
  - 实际：投影用 float32，评估用 float16
  - 或：全部用 float32，但用Tensor Core加速（NVIDIA GPU支持float32 Tensor Core）

---

### 2.5 动态分辨率适配

**场景**：渲染到不同分辨率（如1080p → 4K）

**问题**：高斯尺度不变，但像素更多 → 需要更多高斯才能填满

**优化**：
- 按分辨率缩放高斯尺度：
  ```
  new_scale = old_scale * (new_res / train_res)
  ```
- 或在训练时用多分辨率（随机crop），让高斯适应不同尺度

---

## 3. 性能基准

### 3.1 典型配置（RTX 4090）

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

### 3.2 优化手段对比

| 优化 | 速度提升 | 实现难度 |
|------|----------|----------|
| Tile-based 并行 | 5-10× | 中 |
| 预排序缓存 | 1.2-1.5× | 低 |
| float16 精度 | 1.5-2× | 低 |
| 融合kernel | 1.1-1.3× | 高 |
| **总计** | **可达 20-50×** | 高 |

---

## 4. 与训练pipeline的差异总结

### 4.1 不需要的部分

- ❌ 反向传播（no autograd）
- ❌ 梯度计算
- ❌ 优化器step
- ❌ 密度控制（无densify/prune）
- ❌ 损失计算

### 4.2 可以简化的部分

- ✅ 梯度保留：推理时可关闭梯度（`torch.no_grad()`）
- ✅ 排序策略：可缓存深度顺序
- ✅ 精度：可降低到float16
- ✅ 随机性：训练可能有随机采样，推理需确定性

---

## 5. 实时渲染应用

### 5.1 AR/VR集成

**需求**：
- 低延迟（< 20ms 端到端）
- 高帧率（90/120 FPS）
- 6DoF 头部追踪

**方案**：
- 预训练高斯场景（离线）
- 运行时：根据头部位姿实时渲染
- 压缩高斯数据（量化、压缩）以适应移动端内存

---

### 5.2 自动驾驶仿真

**需求**：
- 快速生成新视角（用于传感器模拟：相机、LiDAR）
- 可编辑（添加/删除物体）

**方案**：
- 使用3DGS表示静态场景
- 动态物体用单独的3DGS或传统模型
- 渲染时融合

---

## 6. 推理代码框架（CUDA思路）

```cpp
// 伪代码
__global__ void render_kernel(
    Gaussian* gaussians,  // 已排序
    int N,
    Camera camera,
    uchar4* image  // 输出
) {
    // 每个block处理一个tile
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_size = 16;

    // 共享内存：缓存该tile涉及的高斯
    extern __shared__ Gaussian shared_gaussians[];
    int shared_count = 0;

    // 1. 预筛选：将影响该tile的高斯加载到shared
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        Gaussian& g = gaussians[i];
        Rect bbox = project_bbox(g.Sigma, g.mu, camera);
        if (bbox.intersects(tile_x, tile_y)) {
            int idx = atomicAdd(&shared_count, 1);
            shared_gaussians[idx] = g;
        }
    }
    __syncthreads();

    // 2. 每个像素计算
    for (int py = 0; py < tile_size; py++) {
        for (int px = 0; px < tile_size; px++) {
            int x = tile_x * tile_size + px;
            int y = tile_y * tile_size + py;

            float3 color = {0,0,0};
            float alpha_acc = 0;

            for (int i = 0; i < shared_count; i++) {
                Gaussian& g = shared_gaussians[i];
                // 计算高斯值
                float g_val = evaluate_gaussian_2d(g.mu_2d, g.Sigma_2d, x, y);
                g_val *= g.alpha;

                // Alpha blending（从后往前，但顺序已排好）
                color += g_val * g.color * (1 - alpha_acc);
                alpha_acc += g_val;

                if (alpha_acc >= 0.99) break;  // 早停
            }

            image[y*width + x] = make_uchar4(
                color.x * 255,
                color.y * 255,
                color.z * 255,
                255
            );
        }
    }
}
```

---

## 思考题（独立重推检验）

1. **缓存的意义**：如果相机每帧随机移动，预排序缓存还有效吗？如何设计fallback策略？
2. **精度与速度**：float16 在3DGS渲染中哪里可能不稳定？如何检测和修复？
3. **内存瓶颈**：假设N=3M高斯，每个高斯13 float（52字节），总内存 ~156MB。如果tile缓存1000高斯，shared memory需要多少？够吗（GPU shared memory通常100KB左右）？
4. **扩展性**：如果要渲染8K分辨率（7680×4320），高斯数量需要增加多少？为什么？

---

**下一章**：扩展与变体 - 3DGS的后续发展（动态场景、压缩、与其他方法的结合）。
