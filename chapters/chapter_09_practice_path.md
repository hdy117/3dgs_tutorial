# 第 9 章：从零到可运行——为什么"调库跑通"不等于理解？

**学习路径**: `problem → invention → verification`

---

## 问题：你调通了官方代码，但真的懂吗？

你在 GitHub 上 clone 了 `matthias-research/3d-gaussian-splatting`, 运行`python train.py -s data/chair`, 看到 PSNR 从 10 飙升到 32。很酷对吧？

然后面试官问你:**"如果我把 densify_interval 从 1000 改成 100，会发生什么？**"

你卡住了。因为你只知道它在哪行代码，不知道**为什么要设计成 1000**。

或者你想加个新特性——比如动态场景支持。你打开 `shaders/` 文件夹，看到满屏的 CUDA kernel，完全无从下手。**因为你的知识是黑盒式的：知道"怎么跑",不知道"为什么这样"**。

**这就是调库和实现的本质区别**。

---

## 起点：从零实现 vs 调库

### 两种路径对比

```
路径 A: 调库
  clone → run → PSNR↑ → "懂了"
  ✅ 30 分钟搞定 demo
  ❌ 改不了，不懂原理，换个场景就懵

路径 B: 从零实现
  设计数据格式 → 写投影公式 → 调渲染 bug → loss 不降 → fix 梯度
  ❌ 要花 2-3 周
  ✅ 每一行代码都知道为什么存在
```

**选择**:我们要的是理解，不是黑盒。

---

### 你会学到什么？（不是"学会 3DGS",而是更深的东西）

1. **调试复杂系统的能力**——渲染全黑、loss NaN、梯度消失，每个问题都在逼你理解底层机制
2. **从数学公式到代码的映射**——`J·Σ·Jᵀ`怎么变成 `torch.matmul()`？特征值分解怎么写稳定？
3. **性能优化的直觉**——为什么 Tile-based 比朴素版快 50×？缓存策略什么时候失效？
4. **增量验证的工程哲学**——每写一个函数立即测试，而不是写完再调试

---

## 发明：如何设计一个"能学到东西"的实现路径？

### 矛盾一："完整功能"vs"学习曲线陡峭"

**问题**:官方的 3DGS 有:
- COLMAP 解析 + 数据加载
- GaussianModel + 参数初始化
- 可微渲染（CUDA kernel）
- 密度控制（split/clone/prune）
- 训练循环 + 学习率调度
- 评估脚本

新手一次性啃全 → **overwhelmed**,三天后放弃。

**你的方案**:分阶段 MVP,每阶段只学一个核心概念:

```
Phase 1 (Week 1): "看到东西"
  ✅ 数据加载（COLMAP）
  ✅ 高斯初始化
  ✅ 慢速渲染 O(N·H·W) —— 不求快，只求数学正确
  ✅ L1 损失 + Adam 优化
  ❌ 不要 densify, 不要 SSIM, 不要 Tile

Phase 2 (Week 2): "让它工作"
  ✅ Tile 预筛选（速度↑50×）
  ✅ 密度控制（N 增长，质量↑）
  ✅ SSIM + L1 组合损失
  ✅ 完整训练循环（30k 步）

Phase 3 (Week 3): "榨干性能"
  ✅ float16 量化
  ✅ 缓存策略
  ✅ torch.compile / CUDA
```

**关键**:每阶段都有明确的"完成标准",不模糊。

---

### 矛盾二:"数学正确"vs"数值稳定"

**问题**:投影公式 `Sigma_2d = J·(R·Σ·Rᵀ)·Jᵀ` 理论上是完美的。但代码里：

```python
# 你会遇到的真实 bug:
z = mu_cam[:, 2]           # 某些高斯在相机后面！z < 0
J = torch.diag([f/z, f/z]) # z → 0 时 J 爆炸
Sigma_2d = J @ Sigma_cam @ J.T  # 非正定，求逆失败
```

**理论**:公式没错  
**现实**:浮点数精度 + 边界情况=灾难

**你被迫发明"数值稳定三原则"**:

1. **Clamp 分母**: `z = z.clamp(min=1e-6)` —— 防止除零
2. **加正则项**: `Sigma_2d += I * 1e-8` —— 保证正定性
3. **裁剪特征值**: `torch.linalg.eigvalsh(Sigma_2d).clamp(min=1e-8)` —— 求逆前检查

这不是数学，是**工程经验**。只有踩过坑才知道。

---

### 矛盾三:"PyTorch 慢"vs"CUDA 难学"

**问题**:
- PyTorch 版渲染：150ms/帧（但可读、可调试）
- CUDA kernel: 3ms/帧（但需要 GPU 编程经验，一个 typo 跑半天）

**你看到的选择**:

| 策略 | 优点 | 缺点 |
|------|------|------|
| 一开始就写 CUDA | 快，"专业" | bug 难调，放弃率高 |
| 先用 PyTorch,后转 CUDA | 易上手，理解深 | 前期慢（但可接受） |
| 永远不调用 CUDA | 简单 | 性能差，学不到底层优化 |

**你的方案**:渐进式:
```
Week 1-2: 纯 PyTorch (接受 150ms/帧)
Week 3: torch.compile() 试试（零成本加速）
后续：读官方 CUDA kernel,选择性重写瓶颈部分
```

**记住**:速度是后天可以优化的，但正确性是基础。先把慢速版跑通，再谈优化。

---

## 验证：如何知道自己"真的懂了"?

### 不是"PSNR>30",而是能回答这些问题:

1. **如果我把所有高斯的 scale_factor×2, 训练会怎么变？**
   - 答案：一开始渲染全黑（高斯太大覆盖整个画面），需要调整 alpha 或降低 scale

2. **为什么 densify_interval=1000，不能是 10?**
   - 答案：densify 要基于梯度累积判断。每步都 densify → N 爆炸式增长，内存溢出；间隔太长→早期收敛慢

3. **Tile-based 渲染在什么情况下会失效？**
   - 答案：相机快速移动时，tile mapping 缓存需要重建；极端情况（高斯包围盒>tile 尺寸）预筛选收益变小

4. **为什么用 MS-SSIM 而不是简单 MSE?**
   - 答案：MSE 对高频细节不敏感（模糊图也能 low loss），SSIM 模拟人眼感知，逼模型学锐利边缘

**如果你能回答这些问题——你不是在调库，你在理解系统。**

---

## 实战路径：3 周从零到可运行

### Week 1: "让它动"（目标：看到模糊图像）

#### Day 1-2: 数据加载（你第一次接触 COLMAP 的恐惧）

**你要解决的问题**:COLMAP 输出的是二进制文件（`.bin`），怎么解析？

```python
# colmap_loader.py (你的第一个函数)
def read_cameras_bin(path):
    """解析 cameras.bin → {camera_id: {id, model, width, height, params}}"""
    with open(path, 'rb') as f:
        # COLMAP 二进制格式：num_cameras(int64), 然后循环读取...
        num_cameras = np.fromfile(f, dtype=np.int64, count=1)[0]
        cameras = {}
        for _ in range(num_cameras):
            camera_id = np.fromfile(f, dtype=np.int64, count=1)[0]
            # ... 解析 model type, width, height, params
    return cameras

def read_images_bin(path, cameras):
    """解析 images.bin → {image_id: {name, R(3x3), T(3,), camera_id}}"""
    # 类似逻辑...
```

**你会卡住的地方**:COLMAP 文档没说清楚二进制格式的细节。你需要：
- Google "COLMAP binary format specification"
- 看官方代码 `colmap/extraction/database.cc`（C++源码）
- 或者直接借用别人的实现（如 nerf-pytorch 的 colmap_loader）

**验证**:
```python
cameras = read_cameras_bin('data/chair/sparse/0/cameras.bin')
images = read_images_bin('data/chair/sparse/0/images.bin', cameras)
print(f"Loaded {len(images)} images, {len(cameras)} cameras")
# 输出：Loaded 100 images, 1 cameras ✅
```

---

#### Day 3-4: 慢速渲染（你第一次写投影公式的挣扎）

**你要解决的问题**:数学公式`mu_2d = K @ (R @ mu + T)`怎么变成代码？还要处理协方差变换。

```python
def project_gaussian(mu, Sigma, R, T, K):
    """
    mu: (N, 3) 世界坐标
    Sigma: (N, 3, 3) 协方差（对称正定）
    R, T: 相机外参（世界→相机）
    K: 内参矩阵
    
    返回：mu_2d(N,2), Sigma_2d(N,2,2), depth(N,)
    """
    # Step 1: 变换到相机坐标系
    mu_cam = torch.matmul(R, mu.unsqueeze(-1)).squeeze(-1) + T  # (N, 3)
    
    # Step 2: 投影（关键！数值稳定）
    z = mu_cam[:, 2].clamp(min=1e-6)  # ← clamp，防止除零
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    mu_2d = torch.stack([
        fx * mu_cam[:, 0] / z + cx,
        fy * mu_cam[:, 1] / z + cy
    ], dim=-1)
    
    # Step 3: 协方差投影（ Jacobian 变换）
    # J = [f/z, 0; 0, f/z] 的简化版
    J = torch.zeros((N, 2, 3), device=mu.device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * mu_cam[:, 0] / (z ** 2)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * mu_cam[:, 1] / (z ** 2)
    
    Sigma_cam = torch.matmul(R, Sigma.unsqueeze(0)).squeeze(0) @ R.transpose(-1, -2)
    Sigma_2d = torch.matmul(J, Sigma_cam) @ J.transpose(-1, -2) + torch.eye(2, device=mu.device) * 1e-8
    
    return mu_2d, Sigma_2d, z
```

**你会遇到的 bug**:
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied` → 检查 unsqueeze/squeeze
- 渲染全黑 → `z.clamp()`没加，某些高斯在相机后面
- 图像偏移 → K 矩阵的 cx, cy 顺序反了

**调试技巧**:打印中间值!
```python
print(f"mu_cam range: [{mu_cam.min():.2f}, {mu_cam.max():.2f}]")
print(f"z range: [{z.min():.4f}, {z.max():.2f}]")  # z 应该>0
print(f"mu_2d range: x[{mu_2d[:,0].min():.1f}-{mu_2d[:,0].max():.1f}], y[...]"
      # mu_2d 应该在 [0, W]×[0, H] 范围内
```

---

#### Day 5-6: 训练循环（你第一次看到 loss 下降的快感）

**最小可行训练循环**:
```python
# train.py (Week 1 版本)
gaussians = GaussianModel(init_points)  # 从 COLMAP 点云初始化
optimizer = torch.optim.Adam([
    {'params': gaussians.mu, 'lr': 1.6e-4},
    {'params': gaussians.Sigma, 'lr': 1e-3},
    {'params': gaussians.alpha, 'lr': 5e-2},
    {'params': gaussians.color, 'lr': 5e-3}
], betas=(0.9, 0.99))

for step in range(1000):
    # 1. 采样一帧
    idx = step % len(dataset)
    gt_image, camera = dataset[idx]
    
    # 2. 渲染（慢速版，不求快）
    rendered = render_slow(gaussians, camera, H=gt_image.shape[1], W=gt_image.shape[2])
    
    # 3. 计算损失（只用 L1，先不加 SSIM）
    loss = torch.nn.functional.l1_loss(rendered, gt_image)
    
    # 4. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")
        save_image(rendered, f"output/step_{step}.png")
```

**如果 loss 不下降**:
- 检查梯度：`print(gaussians.mu.grad.norm())` → 如果是 0，渲染图有问题
- 检查学习率：太大→NaN，太小→不动（尝试×10或÷10）
- 检查初始化：scale_factor 可能不对（调大 10 倍试试）

**你会看到**:loss 从~1.0(随机噪声) 降到~0.3(模糊轮廓)。虽然不清晰，但**它在学**!

---

#### Day 7: Week 1 验收

**通过标准**:
- ✅ 训练 1000 步后 PSNR > 15（从 0 开始）
- ✅ 渲染图有大致结构（不是噪声或全黑）
- ✅ 你能解释每一行代码的作用

**没通过？不要急**。回到 Day 4，调 scale_factor、检查投影公式。这周的目标是"看到东西",不是完美质量。

---

### Week 2: "让它快 + 让它好"（目标：PSNR>25, <100ms）

#### Day 1-2: Tile-based 渲染（第一次性能优化）

**你面临的问题**:慢速渲染`O(N·H·W)`在 N=50k,H=800,W=600 时要 5 秒/帧。训练 30k 步需要...好几天？

**你需要发明 Tile-based 加速**:

```python
def assign_gaussians_to_tiles(mu_2d, Sigma_2d, tile_size, W, H):
    """计算每个高斯的包围盒→分配到高斯可见的 tiles"""
    # 1. 计算 3σ包围盒（覆盖 99.7% 概率）
    std = torch.sqrt(torch.diag(Sigma_2d))  # (N, 2)
    bbox_min = mu_2d - 3 * std
    bbox_max = mu_2d + 3 * std
    
    # 2. 映射到 tile 索引
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size
    
    tile_min_x = (bbox_min[:, 0] / tile_size).floor().long().clamp(0, n_tiles_x-1)
    tile_min_y = (bbox_min[:, 1] / tile_size).floor().long().clamp(0, n_tiles_y-1)
    tile_max_x = (bbox_max[:, 0] / tile_size).ceil().long().clamp(0, n_tiles_x-1)
    tile_max_y = (bbox_max[:, 1] / tile_size).ceil().long().clamp(0, n_tiles_y-1)
    
    # 3. 构建倒排索引：{tile_id: [gaussian_ids]}
    tile_mapping = defaultdict(list)
    for i in range(len(mu_2d)):
        for tx in range(tile_min_x[i], tile_max_x[i]+1):
            for ty in range(tile_min_y[i], tile_max_y[i]+1):
                tile_id = ty * n_tiles_x + tx
                tile_mapping[tile_id].append(i)
    
    return tile_mapping
```

**关键洞察**:每个像素只渲染"能看见它的高斯",而不是全部 N 个。平均每个 tile 只有几十到几百个高斯（不是几十万）。

**验证加速比**:
```python
import time

# 慢速版
start = time.time()
img1 = render_slow(gaussians, camera, H, W)
print(f"Slow: {time.time()-start:.3f}s")  # ~5s

# Tile 版
start = time.time()
img2 = render_tiled(gaussians, camera, H, W)
print(f"Tiled: {time.time()-start:.3f}s")  # ~100ms

# 检查正确性
assert torch.allclose(img1, img2, atol=1e-3)  # 应该几乎相同
```

**预期**:50×加速（5s → 100ms）。

---

#### Day 3-4: 密度控制（第一次"动态调整模型结构"）

**你面临的问题**:固定 N 的训练，PSNR 卡在 20 上不去。某些区域模糊，某些区域噪声多——因为高斯分布不合理。

**你需要发明 densify/prune 策略**:

```python
def densify_and_prune(gaussians, radii, grads_mu, grads_Sigma, optimizer):
    """
    radii: (N,) 每个高斯影响半径（像素单位）
    grads_mu: (N,) |∇μ| 梯度幅度（学习活跃度）
    grads_Sigma: (N,) |∇Σ| 协方差梯度
    
    策略：
    - densify: radii > thresh AND grad > thresh → 克隆或分裂
    - prune: alpha < 0.005 OR scale < threshold → 删除
    """
    # Densify mask（两个条件都要满足）
    to_densify = (radii > 3.0) & (grads_mu > 0.0002)
    
    # Prune mask（任一条件满足就删）
    scales = gaussians.get_scales()
    max_scale = scales.max(dim=1)[0]
    to_prune = (gaussians.alpha < 0.005) | (max_scale < 0.01)
    
    # 执行分裂/克隆（简化版）
    densify_indices = torch.where(to_densify)[0]
    new_gaussians = []
    for idx in densify_indices:
        if torch.rand(1) > 0.5:
            # 克隆：直接复制，稍偏移位置
            new_gaussian = clone(gaussians, idx)
        else:
            # 分裂：沿最大梯度方向分成两个小的高斯
            g1, g2 = split(gaussians, idx)
            new_gaussians.extend([g1, g2])
    
    # 删除不需要的
    to_keep = ~(to_densify | to_prune)
    gaussians.mu = gaussians.mu[to_keep]
    gaussians.Sigma = gaussians.Sigma[to_keep]
    # ... 其他参数同理
    
    # 添加新的
    if new_gaussians:
        gaussians.append(new_gaussians)
    
    # 重建 optimizer（因为 N 变了）
    rebuild_optimizer(optimizer, gaussians)
```

**关键**:densify 要基于**梯度累积**,不是每步都判断。通常每 1000 步缓存一次梯度，在第 1000/2000/3000...步时执行 densify。

**你会看到**:N 从初始的~10k 逐渐增长到 50k-200k（取决于场景复杂度）。PSNR 突破 25→30+。

---

#### Day 5: 完整训练（第一次跑通 30k 步）

**集成所有组件**:
```python
# train.py (Week 2 完整版)
for step in range(30000):
    # 渲染 + loss
    rendered = render_tiled(gaussians, camera)
    loss = l1_loss(rendered, gt) * 0.8 + (1 - ssim(rendered, gt)) * 0.2
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每 1000 步缓存梯度（用于 densify）
    if step % 1000 == 0 and step > 500:
        grads_mu = gaussians.mu.grad.norm(dim=1).detach()
        compute_radii(...)  # 从投影计算影响半径
        
        densify_and_prune(gaussians, radii, grads_mu, optimizer)
    
    # 学习率衰减（7500/15000步时×0.1）
    if step in [7500, 15000]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    
    # 日志 + 可视化
    if step % 100 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, N={len(gaussians.mu)}")
```

**预期结果**(chair数据集):
- 0→500 步：PSNR 从~10→20（初始快速收敛）
- 500→7500 步：PSNR 20→28（densify 工作，N 增长）
- 7500→15000 步：PSNR 28→30（LR 衰减，精细调优）
- 15000→30000 步：PSNR 30→32（收敛到极限）

---

#### Day 6-7: 调试与评估

**如果 PSNR < 25**:
```
诊断流程:
1. N 增长了吗？→ 没增长：grad_threshold 太高，densify 不触发
2. loss 下降了吗？→ 不降：检查 LR、渲染正确性
3. 渲染图有伪影吗？→ 有条纹：Sigma_2d 加正则项；有空洞：scale_factor 调大
```

**可视化驱动调试**:保存中间结果!
```python
if step % 100 == 0:
    # 1. 渲染图（肉眼判断质量）
    save_image(rendered, f"output/step_{step}.png")
    
    # 2. 高斯分布（看是否集中在结构上）
    plt.scatter(mu_2d[:,0].cpu(), mu_2d[:,1].cpu(), s=1)
    plt.savefig(f"output/centers_{step}.png")
    
    # 3. 尺度分布（densify 应该让分布变分散）
    scales = gaussians.get_scales().max(dim=1)[0]
    plt.hist(scales.cpu().numpy(), bins=50)
    plt.savefig(f"output/scales_{step}.png")
```

---

### Week 3: "让它飞"（可选，性能优化）

这一周是**锦上添花**。如果你前两周跑通了，可以：
- float16 量化（推理速度↑2×）
- 缓存排序结果（相机移动小时复用）
- torch.compile() 或 CUDA kernel

但如果前两周没搞定——**不要急**。先把基础版跑通，性能优化是后天的事。

---

## 关键调试技巧（你一定会用到的）

### 症状速查表

```
+------------------+----------+----------------------+------------------------+
| 症状             | 优先级   | 可能原因             | 快速检查               |
+------------------+----------+----------------------+------------------------+
| 渲染全黑         | 🔴高     | Σ太小 / z 为负       | print(scales.mean())   |
|                  |          |                      | print(z.min())         |
+------------------+----------+----------------------+------------------------+
| 渲染全白/模糊    | 🔴高     | α太大 / Σ太大        | print(alpha.mean())    |
+------------------+----------+----------------------+------------------------+
| loss = NaN       | 🔴高     | Σ奇异（求逆失败）    | print(torch.det(Sigma))|
+------------------+----------+----------------------+------------------------+
| 梯度为 0          | 🟡中     | 高斯"死"了           | print(mu.grad.norm())  |
+------------------+----------+----------------------+------------------------+
| PSNR 卡在 20      | 🟡中     | densify 不工作       | print(N over time)     |
+------------------+----------+----------------------+------------------------+
| 速度没提升       | 🟢低     | 用了慢速版           | 确认调用 render_tiled  |
+------------------+----------+----------------------+------------------------+
```

---

## 最终验收：你懂了吗？

### Week 1 通过标准
- [ ] 数据加载能跑通（`dataset[0]` 返回正确的 shape）
- [ ] 慢速渲染能出图（不是全黑/全白，有大致轮廓）
- [ ] 训练 1000 步后 PSNR > 15

### Week 2 通过标准
- [ ] Tile 渲染比慢速版快>30×
- [ ] N 能从 10k 增长到 50k+（densify工作）
- [ ] 完整训练 30k 步后 PSNR > 25

### Week 3 通过标准（可选）
- [ ] float16 推理 <20ms/帧
- [ ] 在测试集上评估，和官方实现差距<2dB

---

## 如果你卡住了...

### "我 Day 4 的渲染图全黑"

**检查清单**:
```python
# 1. z 有没有负数？
print(f"z range: [{z.min():.4f}, {z.max():.2f}]")  # 应该>0

# 2. scale_factor 对不对？
scales = gaussians.get_scales()
print(f"scales mean: {scales.mean().item():.4f}")  # 应该在 0.1-1.0 范围

# 3. mu_2d 在图像范围内吗？
print(f"mu_2d range: x[{mu_2d[:,0].min():.1f}-{mu_2d[:,0].max():.1f}], y[...]"
      # 应该在 [0, W]×[0, H] 附近

# 4. alpha 是不是太小？
print(f"alpha mean: {gaussians.alpha.mean().item():.4f}")  # 应该~0.5
```

**修复**:scale_factor ×10（最常见原因）

---

### "我训练了 1000 步，loss 完全不降"

**检查清单**:
```python
# 1. 梯度有值吗？
print(f"mu.grad.norm() = {gaussians.mu.grad.norm().item():.6f}")  # 应该>0

# 2. LR 是不是太小/太大？
# 尝试：LR ×10（如果不动）或 ÷10（如果 NaN）

# 3. 渲染图有变化吗？
# 保存 step=0, 100, 500, 1000 的图对比
```

---

### "我 densify 了，但 N 不增长"

**检查**:
```python
radii = compute_radii(...)
grads_mu = gaussians.mu.grad.norm(dim=1)

print(f"radii > 3.0: {(radii > 3.0).sum().item()}")
print(f"grads_mu > 0.0002: {(grads_mu > 0.0002).sum().item()}")
print(f"both: {((radii > 3.0) & (grads_mu > 0.0002)).sum().item()}")

# 如果最后一个数字是 0，说明阈值太高
# 尝试：grad_threshold = 0.0001（更激进）
```

---

## 最后一句话

**调库跑通**:30 分钟，黑盒知识  
**从零实现**:2-3 周，深度理解

你选哪个？

如果你选择后者——欢迎加入这趟旅程。准备好 debug、查文档、踩坑了。**但三周后你会感谢自己：因为你拥有了真正的理解，而不只是会 run 代码的能力。**

现在，打开你的编辑器，写下第一行:
```python
def read_cameras_bin(path):
    """我的第一个 COLMAP 解析函数"""
```

---

**附录**:推荐学习资源
- **COLMAP 文档**: https://colmap.github.io/
- **官方代码参考**: https://github.com/graphdeco-inria/gaussian-splatting（不是让你抄，是卡住时看思路）
- **PyTorch 3D**: https://pytorch3d.org/（工具库，可选）

记住：遇到问题先 Google，再问。但**理解为什么比得到答案重要**。
