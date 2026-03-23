# 第7章：完整训练流程

**学习路径**：`example`（最小可行案例）

**核心目标**：将所有组件整合成可运行的训练闭环

---

## 一、训练架构总览

### 1.1 完整流程图

```
┌─────────────────────────────────────────────────────┐
│                    训练循环                          │
├─────────────────────────────────────────────────────┤
│  1. 采样视角  →  camera, gt_image                   │
│  2. 渲染      →  rendered_image = render(gaussians) │
│  3. 计算损失  →  loss = L_img + λ·正则化            │
│  4. 反向传播  →  loss.backward()                    │
│  5. 参数更新  →  optimizer.step()                   │
│  6. 密度控制  →  densify_and_prune()（每N步）      │
│  7. 学习率调度→  lr_decay（特定步数）               │
│  8. 日志监控  →  PSNR, SSIM, 高斯数量              │
└─────────────────────────────────────────────────────┘
```

### 1.2 数据流

```
Dataset → (image, camera)
         ↓
    Gaussians (μ, Σ, α, c)
         ↓
    Renderer → rendered_image
         ↓
    Losses → total_loss
         ↓
    Backprop → gradients
         ↓
    Optimizer → updated_gaussians
         ↓
    Density Control → densify / prune
```

---

## 二、核心组件详解

### 2.1 数据集（Dataset）

**要求**：
- 随机采样视角（训练时）
- 固定视角（测试时）
- 返回：图像（C×H×W）、相机参数（R, T, K）

**实现**（见第6章）：
```python
dataset = SfMDataset(sparse_path, image_path, split='train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
```

---

### 2.2 高斯模型（GaussianModel）

**参数**：
- `mu`: (N, 3) - 位置
- `Sigma`: (N, 3, 3) - 协方差（对称）
- `alpha`: (N, 1) - 不透明度
- `color`: (N, 3) - RGB颜色

**关键方法**：
```python
class GaussianModel:
    def get_scales(self):
        """从Σ提取各轴尺度（特征值开根）"""
        eigvals = torch.linalg.eigvalsh(self.Sigma)  # (N, 3)
        return torch.sqrt(eigvals)  # (N, 3)

    def get_rotation(self):
        """从Σ提取旋转矩阵（特征向量）"""
        eigvals, eigvecs = torch.linalg.eigh(self.Sigma)
        return eigvecs  # (N, 3, 3)
```

---

### 2.3 渲染器（Renderer）

**输入**：
- 高斯模型
- 相机参数（R, T, K）
- 图像分辨率（H, W）

**输出**：
- 渲染图像 (3, H, W)
- 每个高斯的投影半径（用于densify）

**实现要点**：
- 使用tile-based优化（第4章）
- 返回 `radii`：每个高斯在屏幕上的投影半径（像素）
- 关闭梯度：推理部分用 `torch.no_grad()`

```python
def render(gaussians, camera, H, W):
    with torch.no_grad():
        # 1. 投影
        mu_2d, Sigma_2d, depth = project_gaussians(
            gaussians.mu, gaussians.Sigma, camera.R, camera.T, camera.K
        )
        # 2. 计算投影半径（用于densify）
        radii = compute_radii(Sigma_2d)  # (N,)

    # 3. 排序
    indices = torch.argsort(depth, descending=True)

    # 4. Tile-based rendering（可微）
    image = render_tiled(gaussians, indices, mu_2d, Sigma_2d, H, W)

    return image, radii
```

---

### 2.4 损失函数（Loss）

**总损失**：
```python
def compute_loss(rendered, gt, gaussians, λ_ssim=0.8, λ_scale=0.01):
    # 1. 重建损失
    L1 = F.l1_loss(rendered, gt)
    L_ssim = 1 - ms_ssim(rendered, gt)
    L_img = (1 - λ_ssim) * L1 + λ_ssim * L_ssim

    # 2. 尺度正则
    scales = gaussians.get_scales()  # (N, 3)
    L_scale = torch.clamp(scales - 1.0, min=0).mean()

    # 3. 不透明度正则（可选）
    L_opacity = torch.clamp(gaussians.alpha - 0.99, min=0).mean()

    L_total = L_img + λ_scale * L_scale + λ_opacity * L_opacity
    return L_total, L1, L_ssim
```

---

## 三、密度控制实现

### 3.1 梯度缓存

**时机**：反向传播后，optimizer.step() 前

```python
# 反向传播后
loss.backward()

# 缓存梯度幅度
grads_mu = gaussians.mu.grad.detach().norm(dim=1)  # (N,)
grads_Sigma = gaussians.Sigma.grad.detach()
grads_Sigma = grads_Sigma.view(gaussians.N, -1).norm(dim=1)  # (N,)
```

---

### 3.2 Densify & Prune

```python
def densify_and_prune(gaussians, optimizer, radii,
                      grads_mu, grads_Sigma,
                      grad_threshold=0.0002,
                      scale_threshold=0.01,
                      prune_alpha=0.001,
                      max_gaussians=2e6):
    with torch.no_grad():
        # 1. Densify条件
        large_scale = radii > scale_threshold
        large_grad = (grads_mu > grad_threshold) | \
                     (grads_Sigma > grad_threshold)
        to_densify = large_scale & large_grad

        # 2. 执行densify（克隆或分裂）
        if to_densify.any():
            # 简化：只克隆（不分裂）
            new_mu = gaussians.mu[to_densify].clone()
            new_Sigma = gaussians.Sigma[to_densify].clone()
            new_alpha = gaussians.alpha[to_densify].clone()
            new_color = gaussians.color[to_densify].clone()

            gaussians.extend(new_mu, new_Sigma, new_alpha, new_color)
            optimizer.add_param_group([
                {'params': new_mu},
                {'params': new_Sigma},
                {'params': new_alpha},
                {'params': new_color}
            ])

        # 3. Prune条件
        scales = gaussians.get_scales().max(dim=1)[0]
        to_prune = (gaussians.alpha.squeeze() < prune_alpha) | \
                   (scales < 1e-6)

        if to_prune.any():
            keep_mask = ~to_prune
            gaussians.mask = keep_mask
            optimizer.prune(keep_mask)  # 需要实现optimizer.prune

        # 4. 高斯数量上限
        if len(gaussians) > max_gaussians:
            # 随机删除多余高斯
            keep = torch.randperm(len(gaussians))[:int(max_gaussians)]
            gaussians.mask = keep
            optimizer.prune(keep)
```

**注意**：`optimizer.prune()` 需要自己实现：
```python
def prune_optimizer(optimizer, keep_mask):
    """根据keep_mask删除优化器中的参数"""
    # 对每个param_group
    for group in optimizer.param_groups:
        new_params = []
        for i, param in enumerate(group['params']):
            if keep_mask[i]:
                new_params.append(param)
        group['params'] = new_params
```

---

## 四、学习率调度

### 4.1 三阶段调度

**阶段1**（0-7k步）：快速生长
- lr = 1e-2（所有参数）
- 密集densify（每500步）

**阶段2**（7k-15k步）：精细调整
- lr = 1e-3
- densify 变少（每1000步）

**阶段3**（15k+）：微调
- lr = 1e-4
- 不再densify（只prune）

### 4.2 实现

```python
# 初始学习率
initial_lrs = {
    'mu': 1.6e-4,
    'Sigma': 1e-3,
    'alpha': 5e-2,
    'color': 5e-3
}

# 调度点
lr_schedule = {
    7500: 0.1,   # 乘0.1
    15000: 0.1   # 再乘0.1
}

for step in range(total_steps):
    # ... 训练步骤 ...

    # 学习率调度
    if step in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_schedule[step]
        print(f"Step {step}: learning rate decayed to {param_group['lr']}")
```

---

## 五、训练监控指标

### 5.1 图像质量指标

| 指标 | 公式 | 范围 | 说明 |
|------|------|------|------|
| PSNR | 10·log10(MAX²/MSE) | >30 好 | 越高越好 |
| SSIM | (2μxμy+C1)(2σxy+C2)/(...) | 0~1 | 越接近1越好 |
| LPIPS | 学习感知距离 | 0~1 | 越低越好 |

**计算**：
```python
def psnr(rendered, gt):
    mse = ((rendered - gt) ** 2).mean()
    return 10 * torch.log10(1.0 / mse)

def ssim(rendered, gt):
    return ms_ssim(rendered, gt)  # 使用torchmetrics或自定义
```

---

### 5.2 高斯统计指标

- **高斯数量**：初始N → 最终N'（通常增长2-3倍）
- **平均尺度**：应该稳定在合理范围（如 [0.01, 1.0] 世界坐标单位）
- **平均α**：应该集中在 [0.5, 0.95]
- **Σ的条件数**：max(eigval)/min(eigval) → 不应太大（如 < 100）

---

### 5.3 梯度监控

- 记录 |∂L/∂μ|, |∂L/∂Σ| 的分布（均值、分位数）
- 如果梯度长期很小 → 学习率可能太小或高斯已收敛
- 如果梯度爆炸 → 降低学习率或加梯度裁剪

```python
grad_norm_mu = gaussians.mu.grad.norm().item()
grad_norm_Sigma = gaussians.Sigma.grad.norm().item()
print(f"Gradient norms: mu={grad_norm_mu:.6f}, Sigma={grad_norm_Sigma:.6f}")
```

---

## 六、完整训练循环（伪代码）

```python
# 超参数
total_steps = 30000
densify_interval = 1000
densify_from = 500
prune_from = 1500
grad_threshold = 0.0002
scale_threshold = 0.01  # 像素

# 初始化
gaussians = init_from_sfm(sfm_points)
gaussians.to('cuda')
optimizer = Adam([
    {'params': gaussians.mu, 'lr': 1.6e-4},
    {'params': gaussians.Sigma, 'lr': 1e-3},
    {'params': gaussians.alpha, 'lr': 5e-2},
    {'params': gaussians.color, 'lr': 5e-3}
])

# 训练
for step in range(total_steps):
    # 1. 采样
    image, camera = dataset[step % len(dataset)]
    image = image.cuda()
    camera = {k: v.cuda() for k, v in camera.items()}

    # 2. 渲染
    rendered, radii = render(gaussians, camera, camera['height'], camera['width'])

    # 3. 损失
    loss, L1, L_ssim = compute_loss(rendered, image, gaussians)

    # 4. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 5. 缓存梯度（densify用）
    if step < densify_from or step % densify_interval == 0:
        grads_mu = gaussians.mu.grad.detach().norm(dim=1)
        grads_Sigma = gaussians.Sigma.grad.detach()
        grads_Sigma = grads_Sigma.view(gaussians.N, -1).norm(dim=1)

    # 6. 密度控制
    if densify_from <= step < prune_from and step % densify_interval == 0:
        densify_and_prune(gaussians, optimizer, radii,
                          grads_mu, grads_Sigma,
                          grad_threshold, scale_threshold)
    if step >= prune_from and step % densify_interval == 0:
        prune(gaussians, optimizer, radii)

    # 7. 学习率调度
    if step in [7500, 15000]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    # 8. 日志
    if step % 100 == 0:
        psnr_val = psnr(rendered, image).item()
        print(f"Step {step:06d}: loss={loss:.4f}, PSNR={psnr_val:.2f}, "
              f"#gauss={len(gaussians)}, lr={optimizer.param_groups[0]['lr']:.2e}")

        # 保存中间结果
        if step % 1000 == 0:
            save_image(rendered, f"output/step_{step:06d}.png")
            save_gaussians_ply(gaussians, f"output/step_{step:06d}.ply")

    # 9. 验证（每N步在测试集上跑一次）
    if step % 2000 == 0:
        evaluate_on_test_set(gaussians, test_dataset)
```

---

## 七、收敛判断

### 7.1 视觉判断

- 渲染图像与GT无明显差异
- 几何结构完整（无空洞或无意义噪点）

### 7.2 定量判断

- **PSNR**：连续1000步变化 < 0.1 dB
- **高斯数量**：稳定（不再增长）
- **损失**：趋于平稳

### 7.3 时间

- 通常需要 20k-30k 步（取决于数据规模）
- 单GPU（RTX 3090/4090）：~30分钟到2小时

---

## 八、常见问题诊断

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 渲染全黑 | 高斯尺度太小 | 增大初始尺度（scale乘子×5-10） |
| 渲染全白 | α太大或Σ太大 | 减小α初始值，检查Σ初始化 |
| 噪点很多 | 高斯数量不足 | 降低grad_threshold，增加densify频率 |
| 条纹伪影 | Σ奇异（扁高斯） | 添加Σ正则，限制最小特征值 |
| 训练不收敛 | 学习率太高 | 降低所有参数LR 10倍 |
| 内存爆炸 | 高斯增长太多 | 降低grad_threshold，提前开始prune |
| 梯度为0 | 高斯已"死"（α≈0） | 检查α初始化，增大α初始值 |

---

## 九、思考题（独立重推检验）

1. **优化器分组**：为什么不同参数用不同学习率？（μ, Σ, α, c）
2. **densify时机**：为什么要在densify前缓存梯度？为什么densify后要重置优化器状态？
3. **学习率调度**：三次衰减（1e-2 → 1e-3 → 1e-4）分别对应训练什么阶段？
4. **梯度消失**：如果某个高斯长期梯度接近0，可能是什么原因？如何处理？

---

## 十、下一章预告

**第8章**：Feed-Forward推理 - 训练完成后，如何实现实时渲染（<10ms/帧）？详解推理优化策略（预排序缓存、tile极致优化、float16精度）。

---

**关键记忆点**：
- ✅ 训练循环：渲染 → 损失 → 反向 → 优化 → densify/prune
- ✅ 学习率调度：三阶段衰减（1e-2 → 1e-3 → 1e-4）
- ✅ 密度控制：投影尺度大 + 梯度大 → densify；α太小 → prune
- ✅ 监控指标：PSNR、高斯数量、梯度范数
- 🎯 **收敛时间**：20k-30k步，30分钟-2小时（单卡）
