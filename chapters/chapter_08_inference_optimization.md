# 第 8 章：为什么训练完的模型跑不动？——实时推理优化的必然路径

**学习路径**：`problem → starting point → invention → verification → example`

---

## 问题：训练好的高斯，为什么跑不动？

你刚训练完一个场景，300 万高斯，效果很漂亮。现在你想做点什么实际的东西——VR 体验、实时可视化、或者只是个流畅的网页演示。

你把相机往右转一点，渲染一帧……150ms。再转一下……又 150ms。

**6 FPS**。这甚至不如幻灯片。

你盯着这个数字发呆：**明明训练时也是这些高斯，为什么推理这么慢？**

等等，好像哪里不对。训练时每帧确实要 200-300ms，但那包括了反向传播、优化器更新、密度控制……这些东西在推理时根本不需要啊？

**问题的本质浮现了：我们把"训练代码"当成了"推理代码"。**

---

## 起点：推理到底需要什么？

让我们退一步，从第一性原理思考：**实时渲染的硬约束是什么？**

### 需求 1：延迟 < 10ms（物理限制）

人眼的运动模糊阈值是 16ms（60Hz）。如果你的渲染超过这个时间：
- VR 里会晕眩（头部转动和画面不同步）
- AR 里虚拟物体会"漂移"
- 游戏操作会有明显的输入延迟

**10ms 不是目标，是底线。**

### 需求 2：确定性输出

训练时你可以有随机性——dropout、随机采样、数据增强。但推理时：
- **同样的相机位姿必须产生完全相同的图像**
- 否则 VR 里画面会闪烁（相邻帧不一致）
- AR 里虚拟物体会抖动

### 需求 3：内存带宽友好

你训练了 300 万高斯，每个约 50 字节（μ、Σ、α、SH 系数）。总共**150MB**。

每渲染一帧都要读一遍这 150MB。RTX 4090 的内存带宽是 1TB/s —— 听起来很快？但实际测试只有 50-80 FPS。为什么理论极限和现实差距这么大？

---

## 发明：从矛盾中逼出优化策略

### 矛盾一："每帧都重算"vs"相机移动很小"

**你观察到一个奇怪的现象**：

训练时，你把所有高斯按深度排序（近大远小），然后逐帧渲染。但推理时，如果相机只是微微转动——比如 VR 头显的头部追踪，每秒更新 90 次，每次位移可能只有几毫米——**你真的需要每帧都重新排序吗？**

300 万高斯排序是 O(N log N)。实测要 5-10ms。这占了你 10ms 预算的一半！

**但关键是：如果相机移动很小，深度顺序几乎不变。**

于是你发明了一个**启发式缓存策略**：
```python
class InferenceState:
    def __init__(self):
        self.sorted_indices = None      # 缓存的排序结果
        self.last_camera_pose = None     # 上次相机位姿
    
    def render(self, camera_pose):
        if self._need_resort(camera_pose):
            # 只在大移动时才重算
            depth = project_and_get_depth(gaussians, camera_pose)
            self.sorted_indices = torch.argsort(depth, descending=True)
        
        return render_with_cached_order(self.sorted_indices, camera_pose)
    
    def _need_resort(self, curr_pose, trans_thresh=0.1, rot_thresh=5):
        """平移>10cm 或旋转>5°时失效"""
        if self.last_camera_pose is None:
            return True
        
        delta_t = torch.norm(curr_pose.T - self.last_camera_pose.T)
        R_rel = curr_pose.R @ self.last_camera_pose.R.T
        angle = torch.acos((torch.trace(R_rel) - 1) / 2) * 180 / np.pi
        
        return delta_t > trans_thresh or angle > rot_thresh
```

**效果验证**：相机缓慢移动时，排序从每帧 5ms → 几乎 0（缓存命中）。加速 1.5-2×。

---

### 矛盾二："float32 很安全"vs"带宽是瓶颈"

训练时用 float32 是为了数值稳定——反向传播对精度敏感。但推理呢？

**你意识到一个事实**：最终输出是 8 位图像（0-255）。中间计算用 float32 是不是有点过度了？

float16（半精度）的优势：
- **内存减半**：150MB → 75MB，带宽压力÷2
- **计算加速**：Tensor Core 对 FP16 的 throughput 是 FP32 的 2×
- **质量损失**：实测 PSNR < 0.5dB（肉眼不可见）

但有个坑：**协方差矩阵求逆在 float16 下可能不稳定**。

你尝试了三种方案：

| 方案 | 方法 | 效果 |
|------|------|------|
| 全 float16 | 所有计算半精度 | Σ⁻¹偶尔发散，渲染有噪点 |
| 混合精度 | 投影用 FP32，评估用 FP16 | ✅ 稳定，加速明显 |
| epsilon 修正 | Sigma += I·1e-8（float16） | ⚠️ 需要调参 |

**最终选择混合精度**：
```python
gaussians = gaussians.half()  # 存储用 FP16

with torch.no_grad(), torch.cuda.amp.autocast():
    rendered = render(gaussians, camera)
```

**验证**：内存带宽从瓶颈→非瓶颈，FPS 提升 1.5-2×。

---

### 矛盾三："Tile-based 渲染很快"vs"每帧都重建 tile mapping 很慢"

第 7 章你发明了 Tile-based 渲染——把画面分成 16×16 的瓦片，每个瓦片只渲染能看见的高斯。这个想法本身很聪明，但有个隐藏成本：

**每帧都要计算「哪些高斯在哪些 tile 里」**。对 300 万高斯来说，这个映射计算要 1-2ms。

你再次观察：**如果相机移动很小，tile mapping 也会变化吗？**

大部分高斯的投影中心偏移只有几个像素——它们还在原来的 tile 里！只有边缘的高斯可能跨 tile 边界。

**于是你发明了 Tile 缓存**：
```python
class TileCache:
    def __init__(self):
        self.tile_mapping = None  # {tile_id: [gaussian_ids]}
    
    def get_tile_gaussians(self, mu_2d, camera_pose):
        if self._need_rebuild(camera_pose):
            # 重新计算包围盒→tile 映射
            self.tile_mapping = build_tile_mapping(mu_2d)
        
        return self.tile_mapping
    
    def _need_rebuild(self, curr_pose):
        """和排序缓存共享失效条件"""
        # ... 类似 need_resort 的逻辑
```

**关键洞察**：Tile 缓存可以和深度排序缓存**共享同一个失效条件**。相机移动大时，两者一起失效；移动小时，一起复用。

---

### 矛盾四："多个 kernel 好调试"vs"kernel 启动开销很大"

现在的渲染流程是：
1. Kernel A: 投影（μ, Σ → mu_2d, Sigma_2d, depth）
2. Kernel B: 排序（depth → sorted_indices）
3. Kernel C: Tile 映射（mu_2d → tile_mapping）
4. Kernel D: Alpha blending（tile_mapping → pixels）

每个 kernel 启动有~10μs 开销。更糟的是：**数据在 global memory 反复读写**。

```
投影写 Sigma_2d (global) → Tile 映射读 Sigma_2d (global) → Alpha blend 再读一次 (global)
```

**你意识到：同一个数据被读了三遍！**

CUDA 的 shared memory（片上 SRAM）带宽是 global memory 的 10-50 倍。如果能把中间结果留在 shared memory，速度会快多少？

这就是**Kernel Fusion**的思路——把多个 kernel 合并成一个：
```cpp
__global__ void fused_render_kernel(...) {
    // Shared memory（所有线程共享）
    __shared__ float2 mu_2d_shared[GRID_SIZE];
    __shared__ float4 Sigma_2d_shared[GRID_SIZE];
    
    // Step 1: 投影（读 global → 写 shared）
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    mu_2d_shared[idx] = project(gaussians[idx], camera);
    __syncthreads();
    
    // Step 2: 块内排序（在 shared memory 里操作）
    block_sort(mu_2d_shared, depth_shared);
    __syncthreads();
    
    // Step 3: Alpha blending（直接从 shared 读，不写 global）
    float color = blend_gaussians_in_tile(...);
    
    // 只写最终结果到 global
    output_image[tile_id][pixel_idx] = color;
}
```

**效果验证**：减少 global memory 访问次数，加速 1.2-1.3×。但调试难度剧增——一个 kernel 里要处理所有逻辑，出错很难定位。

---

## 验证：性能金字塔与瓶颈分析

### 优化效果的层级

把所有优化按"必须性"排序：

```
基础层（不做到无法实时）:
  ✅ torch.no_grad()        → 1.5× (去掉 autograd graph)
  ✅ 无密度控制             → 5×   (densify/prune 是训练特有)
  
中间层（强烈建议）:
  ✅ 预排序缓存             → 1.5-2× (相机移动小时)
  ✅ float16 混合精度       → 1.5-2× (带宽÷2, Tensor Core)
  ✅ Tile 映射缓存          → 1.2×   (配合排序缓存)

高级层（调优用）:
  🔧 Kernel Fusion         → 1.2-1.3× (减少 global memory 访问)
  🔧 SoA 内存布局          → 1.5-2×  (coalesced access)
```

**总加速比**：基础 + 中间层 ≈ **20-50×**。训练时 150ms → 推理时 3-7ms（80-300 FPS）。

### RTX 4090 实测数据

| 场景 | 高斯数量 | 分辨率 | 未优化 | 优化后 | FPS |
|------|---------|--------|--------|--------|-----|
| 小室内 | 0.5M | 800×600 | 50ms | 5ms | 200 |
| 中场景 | 1.5M | 1200×900 | 150ms | 12ms | 83 |
| 大外景 | 3M | 1920×1080 | 400ms | 20ms | 50 |

### 瓶颈在哪里？（用 Nsight Compute 分析）

**N < 1M**：memory-bound。GPU 在等数据从 VRAM 读入，计算单元空闲。
- 优化方向：float16、SoA 布局、kernel fusion

**N > 2M**：compute-bound。投影 + 高斯评估的计算量太大。
- 优化方向：SIMD 向量化、降低 SH 阶数、early ray termination

**分辨率增加**：线性变慢。像素×4（1080p→4K），时间也×4。
- 这不是 bug，是物理限制——每个像素都要独立计算 alpha blending。

---

## 应用思考题（验证你的理解）

### 思考 1：缓存的"雪崩效应"

如果 VR 用户突然转头（比如被吓到了），相机位姿从 (0,0,0) 跳到 (0.5m, 30°旋转)。你的排序缓存和 Tile 缓存同时失效，需要重算。

**问题**：
- 重排 300 万高斯要 10ms，这一帧就会掉到 100FPS → 60FPS
- 如何设计**渐进式更新**避免卡顿？
- 或者：**后台异步排序**（当前帧用旧缓存，下一帧用新结果）的代价是什么？

### 思考 2：移动端的"降维打击"

手机 GPU（如 Adreno 740）算力只有 RTX 4090 的 1/40。你想在 Quest 3 上跑实时 3DGS，该怎么做？

**约束条件**：
- 延迟 < 20ms（Quest 刷新率 120Hz）
- 内存带宽 ~50GB/s（RTX 4090 是 1000GB/s）
- VRAM < 8GB（包括系统 + 应用）

**你的方案**：
```
N: 3M → ? (降到多少能跑？)
分辨率：1080p → ? 
精度：float16 → ? (还能更低吗？INT8?)
Tile 大小：16×16 → ? (更小的 tile 减少每瓦片计算量？)
```

### 思考 3：内存带宽的"物理极限"

RTX 4090: 1TB/s = 8e12 bytes/s  
3M 高斯 × 50B = 150MB  

如果每帧读一遍，**理论 FPS 上限** = 8e12 / 150e6 = **53,333 FPS**。

但实际只有 50-200 FPS。为什么差了 200-1000 倍？

用 Nsight Compute 分析，你会发现：
- kernel 启动开销（每个~10μs）
- shared memory spill（溢出到 global）
- warp divergence（不同线程走不同分支）
- **真正的带宽利用率可能只有 30-50%**

你能通过 SoA 布局、向量化加载（float4）、kernel fusion 把利用率提到 80%+吗？

---

## 本章核心记忆点

✅ **问题本质**：推理不是"训练代码去掉反向传播"，而是完全不同的优化目标（延迟 vs 精度）

✅ **三个硬约束**：
- <10ms 延迟（人眼运动模糊阈值）
- 确定性输出（避免闪烁/抖动）
- 带宽友好（每帧都要读全部高斯）

✅ **核心发明**：从"相机移动很小"这个观察出发，发明了缓存策略（排序+Tile）。这不是凭空想出来的，是被性能瓶颈逼出来的。

✅ **验证方法**：用 Nsight Compute 找真实瓶颈（memory-bound vs compute-bound），而不是瞎优化。

---

**下一章预告**：3DGS 的极限在哪里？我们能表示多复杂的场景？动态物体怎么处理？第 9 章讨论**扩展与变体**——4D Gaussian、压缩传输、几何增强，以及如何把 3DGS 变成真正的产品而非 demo。
