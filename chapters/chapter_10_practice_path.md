# 第10章：从零到可运行——3DGS完整实现实践指南

**学习路径**：`problem → invention → verification → integration`

**本章核心目标**：在**3周内**，用PyTorch从零实现一个能跑通真实数据集的3DGS系统。不是调库，而是理解每一行代码为什么存在。

---

## 一、实践的本质：你在构建什么？

### 1.1 目标 vs 幻觉

**目标**：一个可运行的3DGS实现，包含：
- ✅ 数据加载（COLMAP解析）
- ✅ 高斯初始化（从SfM点云）
- ✅ 可微渲染（投影+混合）
- ✅ 训练循环（损失+优化+densify）
- ✅ 实时推理（<20ms）

**幻觉**：一行代码调通（可能，但你不理解）

**选择**：我们要的是理解，不是黑盒。

---

### 1.2 三阶段学习曲线

```
阶段1: 跑通（Week 1）
  - 慢速渲染（O(N·H·W)）无所谓
  - 无densify也行
  - 目标：看到图像从噪声变清晰

阶段2: 完整（Week 2）
  - Tile优化到<100ms
  - 密度控制工作
  - 目标：PSNR>25，速度可接受

阶段3: 优化（Week 3）
  - CUDA kernel融合
  - float16量化
  - 目标：<10ms实时
```

**不要跳阶段**：先跑通再优化。

---

## 二、Axioms：实践设计的公理

### 公理1：增量验证

**每写一个函数，立即验证**：
- 投影函数 → 打印 mu_2d 范围是否在图像内
- 渲染函数 → 保存图像，看是否不是全黑
- 损失函数 → 检查梯度是否非零

**反模式**：写完所有再调试 → 无法定位问题。

---

### 公理2：慢速优先

**先用O(N·H·W)慢速渲染**，确保数学正确，再优化到Tile-based。

**为什么**：
- Tile-based有索引bug难调试
- 慢速版每行数学对应公式，易验证
- 速度可以后天优化，正确性必须优先

---

### 公理3：小数据驱动

**永远用小数据集（<100张图，<10k高斯）开始**：
- 迭代快（分钟级 vs 小时级）
- 可视化容易（可以看每张图）
- 错误代价低（重训练快）

**原则**：在chair数据集上跑通，再到更大场景。

---

## 三、Contradictions：实践中的权衡

### 矛盾1：理论正确 vs 数值稳定

**问题**：
- Σ投影公式 `J·(R·Σ·Rᵀ)·Jᵀ` 理论上正确
- 但z接近0时J爆炸，Σ_2d可能非正定
- 怎么平衡？

**实践方案**：
```python
# 1. z clamp (必须)
z = mu_cam[:, 2].clamp(min=1e-6)

# 2. Σ_2d加正则 (必须)
Sigma_2d = Sigma_2d + torch.eye(2) * 1e-8

# 3. 特征值裁剪 (可选)
eigvals = torch.linalg.eigvalsh(Sigma_2d)
eigvals = eigvals.clamp(min=1e-8)
```

**口诀**：先clamp，再加epsilon，最后clip特征值。

---

### 矛盾2：PyTorch vs CUDA

**问题**：
- PyTorch慢（150ms/帧），但易调试
- CUDA快（5ms/帧），但难写

**实践路径**：
1. **Week 1-2**: 纯PyTorch（接受慢）
2. **Week 3**: 用`torch.compile`试试加速
3. **后续**: 阅读官方CUDA kernel，选择性重写瓶颈

**不要一开始就写CUDA**，除非你有GPU编程经验。

---

### 矛盾3：完整实现 vs 简化

**问题**：
- 完整3DGS有37个超参数
- 新手容易 overwhelmed

**解决方案**：**最小可行产品（MVP）清单**

```
Phase 1 MVP (Week 1):
  ✅ 数据加载（COLMAP）
  ✅ 高斯初始化（各向同性）
  ✅ 慢速渲染（O(N·H·W)）
  ✅ L1损失（不用SSIM）
  ✅ 无densify，固定N

Phase 2 MVP (Week 2):
  ✅ Tile渲染（<100ms）
  ✅ 预筛选（包围盒）
  ✅ 早停（α累计）
  ✅ L1+SSIM
  ✅ 基础densify（克隆）

Phase 3 MVP (Week 3):
  ✅ 分裂densify
  ✅ 学习率调度
  ✅ 完整prune
  ✅ float16量化
```

**每阶段验证通过再进下一阶段**。

---

## 四、Solution Path：3周详细计划

### Week 1：基础框架（目标：看到模糊图像）

#### Day 1-2：环境 + 数据加载

**任务清单**：
- [ ] 创建conda环境，安装PyTorch
- [ ] 下载nerf_synthetic/chair数据集
- [ ] 实现 `colmap_loader.py`（解析cameras.bin, images.bin, points3D.bin）
- [ ] 实现 `Dataset` 类，返回 (image, camera) 对
- [ ] 验证：`dataset[0]` 输出合理shape

**验证标准**：
```python
dataset = SfMDataset(...)
img, cam = dataset[0]
assert img.shape == (3, H, W)
assert cam['R'].shape == (3,3)
print("✅ 数据加载就绪")
```

---

#### Day 3-4：慢速渲染（O(N·H·W)）

**任务清单**：
- [ ] 实现 `GaussianModel` 初始化（从SfM点云）
- [ ] 实现 `project_gaussian()`（投影公式）
- [ ] 实现 `render_slow()`（嵌套循环：像素×高斯）
- [ ] 可视化第一帧：`plt.imshow(render_slow(gaussians, camera))`

**常见坑**：
- Σ_2d 计算错误 → 检查 `J @ Sigma_cam @ J.T`
- 坐标系错误 → 渲染图偏移/颠倒
- 尺度太小 → 全黑 → `scale_factor *= 10`

**验证标准**：
- 渲染图不是全黑/全白
- 有大致轮廓（即使模糊）
- 颜色基本正确（来自SfM RGB）

---

#### Day 5-6：训练循环（无densify）

**任务清单**：
- [ ] 实现 `compute_loss()`（L1 + SSIM）
- [ ] 实现训练循环（前向→损失→反向→更新）
- [ ] 固定N（不densify/prune）
- [ ] 每100步保存渲染图

**验证标准**：
```python
# 运行100步
for step in range(100):
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")
# loss应该下降
```

**如果loss不下降**：
- 检查梯度：`gaussians.mu.grad.norm()` > 0?
- 检查LR：太大→NaN，太小→不动
- 检查渲染：是否输出有意义图像？

---

#### Day 7：Week 1验收

**目标**：
- 训练1000步，PSNR从0提升到15+
- 渲染图有基本结构

**如果失败**：
- 回到Day 4，检查渲染质量
- 可能scale_factor不对（调大10倍）

---

### Week 2：完整流程（目标：PSNR>25，速度<100ms）

#### Day 1-2：Tile预筛选

**任务清单**：
- [ ] 实现 `compute_bbox()`（高斯包围盒）
- [ ] 实现 `assign_gaussians_to_tiles()`（倒排索引）
- [ ] 实现 `render_tiled()`（每tile独立）
- [ ] 对比慢速版和tile版输出差异（应<1e-3）

**性能目标**：
- N=50k，H=800，W=600
- 慢速：~5s/帧
- Tile：<100ms/帧（50×加速）

**调试技巧**：
- 先跑tile版，但每个tile仍遍历所有像素（验证索引正确）
- 再跑完整tile（每个像素只遍历tile内高斯）

---

#### Day 3-4：密度控制（Densify）

**任务清单**：
- [ ] 实现 `densify_and_prune()`（克隆+分裂）
- [ ] 每1000步缓存梯度（`grads_mu = mu.grad.norm(dim=1)`）
- [ ] 实现分裂：特征值分解 + 沿主方向分裂
- [ ] 实现克隆：直接复制
- [ ] 验证：N从10k增长到50k

**关键参数**：
- `grad_threshold = 0.0002`
- `scale_threshold = 0.01`（像素）
- `densify_interval = 1000`
- `densify_from = 500`

**如果densify不触发**：
- 检查 `radii` 是否>threshold（可能scale_factor太小）
- 检查梯度是否>threshold（可能LR太小）

---

#### Day 5：完整训练（30k步）

**任务清单**：
- [ ] 集成所有组件
- [ ] 实现学习率调度（7500, 15000步 ×0.1）
- [ ] 运行完整训练（30k步）
- [ ] 监控：PSNR曲线、N增长曲线

**预期结果**（chair数据集）：
- 初始PSNR：10-15
- 10k步：25-28
- 30k步：30-32
- N：10k → 100k-300k

**如果PSNR卡住**：
- 检查densify是否工作（N是否增长）
- 检查损失是否包含梯度项（L1+SSIM）
- 尝试降低 `grad_threshold`（更激进densify）

---

#### Day 6-7：评估与调参

**任务清单**：
- [ ] 在测试集评估（10张图）
- [ ] 可视化对比（GT vs 渲染）
- [ ] 保存最终模型（gaussians.pt）
- [ ] 尝试不同scale_factor（0.5, 1.0, 2.0）
- [ ] 尝试不同grad_threshold（0.0001, 0.0002, 0.0005）

**验收标准**：
- PSNR(test) > 25
- 渲染图无大空洞
- 速度 < 100ms/帧

---

### Week 3：优化与部署（可选）

#### Day 1-2：推理优化

**任务清单**：
- [ ] 实现 `InferenceState`（缓存排序）
- [ ] 关闭autograd（`with torch.no_grad()`）
- [ ] float16量化（`gaussians.half()`）
- [ ] 测试延迟：<20ms？

---

#### Day 3-4：CUDA kernel（进阶）

**如果时间充裕**：
- [ ] 阅读官方CUDA kernel
- [ ] 用`torch.compile`试试
- [ ] 选择性重写投影部分

---

#### Day 5-7：自己的数据

**任务清单**：
- [ ] 用手机拍50-100张场景
- [ ] COLMAP SfM处理
- [ ] 3DGS训练
- [ ] 实时渲染展示

---

## 五、调试哲学：如何快速定位问题

### 5.1 分层验证法

```
数据加载层
  ↓ 验证：dataset[0]输出合理
  └─失败 → 检查COLMAP路径、图像读取

初始化层
  ↓ 验证：gaussians.N > 0, scales.mean() > 0
  └─失败 → 检查尺度估计（重投影误差计算）

渲染层
  ↓ 验证：render_slow() 有图像轮廓
  └─失败 → 检查投影公式（mu_2d范围、Sigma_2d特征值）

训练层
  ↓ 验证：100步内loss下降
  └─失败 → 检查梯度（mu.grad.norm()）、LR
```

---

### 5.2 症状速查表

```
+------------+----------+----------+----------+
| 症状       | 优先级   | 可能原因 | 快速检查 |
+------------+----------+----------+----------+
| 全黑/全白  | 高       | Σ太小/大 | scales.mean() |
| 偏移/颠倒  | 高       | 坐标系错 | 渲染vs GT对比 |
| loss NaN   | 高       | Σ奇异    | torch.det(Sigma) |
| 梯度为0    | 中       | 高斯"死" | mu.grad.norm() |
| 不收敛     | 中       | LR不对   | 尝试LR×10或÷10 |
| 速度慢     | 低       | 未优化   | 检查是否tile版 |
+------------+----------+----------+----------+
```

---

### 5.3 可视化驱动

**必须保存的中间结果**：
```python
# 每100步保存
if step % 100 == 0:
    # 1. 渲染图
    save_image(rendered, f"output/step_{step:06d}.png")
    
    # 2. 投影中心分布
    plt.scatter(mu_2d[:,0].cpu(), mu_2d[:,1].cpu(), s=1)
    plt.savefig(f"output/centers_{step:06d}.png")
    
    # 3. 尺度分布
    scales = gaussians.get_scales().max(dim=1)[0]
    plt.hist(scales.cpu().numpy(), bins=50)
    plt.savefig(f"output/scales_{step:06d}.png")
```

**看图的直觉**：
- 渲染图：从噪声→模糊→清晰
- 投影中心：从随机→聚集→匹配GT结构
- 尺度分布：从集中→分散（densify工作）

---

## 六、关键函数参考实现

### 6.1 完整 `render_tiled()` 框架

```python
def render_tiled(gaussians, camera, H, W, tile_size=16):
    with torch.no_grad():
        # 1. 投影
        mu_2d, Sigma_2d, depth = project_gaussian(
            gaussians.mu, gaussians.Sigma,
            camera['R'], camera['T'], camera['K']
        )
        
        # 2. 排序
        indices = torch.argsort(depth, descending=True)
        mu_2d = mu_2d[indices]
        Sigma_2d = Sigma_2d[indices]
        alpha = gaussians.alpha[indices]
        color = gaussians.color[indices]
        
        # 3. 包围盒
        bbox_min, bbox_max = compute_bbox(mu_2d, Sigma_2d)
        
        # 4. Tile映射
        tile_mapping = assign_gaussians_to_tiles(
            bbox_min, bbox_max, tile_size, W, H
        )
        
        # 5. 渲染
        image = torch.zeros((3, H, W), device=gaussians.mu.device)
        accum_alpha = torch.zeros((1, H, W), device=gaussians.mu.device)
        
        n_tiles_x = (W + tile_size - 1) // tile_size
        
        for tile_id, g_indices in enumerate(tile_mapping):
            if not g_indices:
                continue
            
            ty = tile_id // n_tiles_x
            tx = tile_id % n_tiles_x
            x0, x1 = tx*tile_size, min((tx+1)*tile_size, W)
            y0, y1 = ty*tile_size, min((ty+1)*tile_size, H)
            
            for y in range(y0, y1):
                for x in range(x0, x1):
                    pixel = torch.tensor([x, y], device=gaussians.mu.device)
                    for g_idx in g_indices:
                        # 计算高斯值
                        diff = pixel - mu_2d[g_idx]
                        inv_Sigma = torch.linalg.inv(Sigma_2d[g_idx])
                        exponent = -0.5 * (diff @ inv_Sigma @ diff)
                        g_val = alpha[g_idx] * torch.exp(exponent)
                        
                        # Alpha blending
                        contrib = g_val * color[g_idx] * (1 - accum_alpha[:, y, x])
                        image[:, y, x] += contrib.squeeze()
                        accum_alpha[:, y, x] += g_val
                        
                        if accum_alpha[0, y, x] >= 0.99:
                            break
        
        return image
```

---

## 七、Week 1快速启动脚本

```bash
#!/bin/bash
# setup.sh

conda create -n 3dgs python=3.10 -y
conda activate 3dgs

# PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 依赖
pip install tqdm opencv-python matplotlib numpy imageio scikit-image pycolmap

# 数据
mkdir -p data/nerf_synthetic
wget -O data/nerf_synthetic.zip https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/dataset/nerf_synthetic.zip
unzip data/nerf_synthetic.zip -d data/

echo "✅ 环境就绪，开始Day 1"
```

---

## 八、最终检查清单

### 8.1 代码完整性

```
[ ] colmap_loader.py: 解析COLMAP输出
[ ] dataset.py: SfMDataset类
[ ] gaussian.py: GaussianModel类
[ ] projection.py: project_gaussian()
[ ] render_slow.py: 慢速渲染
[ ] render_tiled.py: Tile渲染
[ ] losses.py: compute_loss()
[ ] train.py: 完整训练循环
[ ] eval.py: 测试集评估
```

---

### 8.2 功能验证

```
[ ] 数据加载：dataset[0]返回正确shape
[ ] 初始化：gaussians.N > 0, scales合理
[ ] 慢速渲染：图像有轮廓
[ ] 训练100步：loss下降
[ ] Tile渲染：速度<100ms
[ ] Densify：N增长
[ ] 完整训练：PSNR>25
[ ] 推理：<20ms延迟
```

---

### 8.3 性能基准

```
+----------------+--------+--------+
| 指标           | 目标   | 实测   |
+----------------+--------+--------+
| 训练时间       | 30分钟 |        |
| 推理延迟       | <20ms  |        |
| 内存占用       | <200MB |        |
| PSNR (test)    | >25    |        |
+----------------+--------+--------+
```

---

## 九、如果卡住了：重启检查点

### 9.1 Week 1失败

**症状**：Day 4慢速渲染出不了图

**检查顺序**：
1. 渲染是不是全黑？→ scale_factor调大10倍
2. 渲染是不是全白？→ alpha调至0.3，scale_factor调小
3. 图像偏移？→ 检查坐标系转换（R/T符号）
4. 条纹伪影？→ Sigma加正则 `+ I·1e-8`

---

### 9.2 Week 2失败

**症状**：训练1000步PSNR<20

**可能原因**：
- Densify没工作 → N没增长 → grad_threshold太高？
- LR不对 → loss不下降 → 尝试LR×10或÷10
- 初始化太差 → scale_factor不对 → 回到Week 1调

---

### 9.3 Week 3失败

**症状**：Tile渲染速度不升反降

**检查**：
- 是不是用了慢速版代码？
- Tile映射是否正确（每个像素只遍历tile内高斯）？
- 用`torch.cuda.synchronize()`准确计时

---

## 十、现在，开始烧起来吧！🔥

**记住**：
1. **先跑通，再优化**：慢速版能出图比优化但跑不通重要
2. **小数据驱动**：chair数据集<100张图，迭代快
3. **可视化驱动**：每100步保存渲染图，用眼睛判断
4. **增量验证**：每写一个函数立即测试

**3周后**，你将拥有：
- ✅ 完整的3DGS实现（3000+行）
- ✅ 对每一行代码的理解
- ✅ 调试复杂问题的能力
- ✅ 可扩展的代码框架

**这比任何调库都珍贵。**

---

**附录A：完整项目结构**

```
3dgs_implementation/
├── config.yaml              # 超参数配置
├── train.py                 # 训练入口
├── render.py                # 推理入口
├── eval.py                  # 评估脚本
├── setup.sh                 # 环境配置
├── utils/
│   ├── __init__.py
│   ├── colmap_loader.py    # COLMAP解析
│   ├── camera.py           # 相机工具
│   └── image.py            # 图像保存
├── data/
│   ├── __init__.py
│   └── dataset.py          # SfMDataset
├── gaussian/
│   ├── __init__.py
│   └── gaussian.py         # GaussianModel
├── rendering/
│   ├── __init__.py
│   ├── projection.py       # 投影函数
│   ├── render_slow.py      # 慢速渲染
│   ├── render_tiled.py     # Tile渲染
│   └── losses.py           # 损失函数
└── output/
    ├── step_000000.png
    ├── step_001000.png
    └── ...
```

---

**附录B：关键超参数速查**

```yaml
# config.yaml
total_steps: 30000
densify_interval: 1000
densify_from: 500
prune_from: 15000
grad_threshold: 0.0002
scale_threshold: 0.01  # 像素
scale_factor: 0.5      # 初始化尺度系数
lambda_ssim: 0.8
lambda_scale: 0.01
lr_mu: 1.6e-4
lr_Sigma: 1e-3
lr_alpha: 5e-2
lr_color: 5e-3
```

---

**附录C：典型时间预算**（RTX 4090, chair数据集）

```
Week 1 (nerf_synthetic/chair, 100张图, ~10k高斯):
  数据加载: 5分钟
  慢速渲染: 2s/帧（调试用）
  训练1000步: 30分钟（无densify）

Week 2 (完整流程):
  训练30k步: 30-60分钟
  Tile渲染: 10-20ms/帧

Week 3 (优化):
  float16量化: 推理<10ms
  CUDA编译: 额外2-3天学习成本
```

---

**最后一句**：遇到问题先搜，再问。但记住——**理解为什么**比**得到答案**重要100倍。

现在，开始写你的第一行代码。