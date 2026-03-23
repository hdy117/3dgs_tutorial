# 第7章：完整训练流程

**学习路径**：`example`（最小可行案例）

---

## 引言：把碎片拼成闭环

本章整合：
- 第4章：渲染管线
- 第5章：损失函数
- 第6章：数据准备
- 密度控制策略

**目标**：写出能真正跑起来的训练循环（伪代码级）

---

## 1. 训练架构总览

```python
# 初始化
gaussians = load_from_sfm(sfm_output)   # 第6章
optimizer = Adam([gaussians.params], lr=1e-3)
dataset = load_dataset(sfm_output)

# 训练
for step in range(total_steps):
    # 1. 采样一批视角
    camera = dataset.sample_camera()

    # 2. 渲染
    rendered_image = render(gaussians, camera)  # 第4章

    # 3. 加载GT图像
    gt_image = dataset.get_image(camera)

    # 4. 计算损失
    loss = compute_loss(rendered_image, gt_image, gaussians)  # 第5章

    # 5. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 6. 密度控制（每N步）
    if step % densify_interval == 0:
        densify_and_prune(gaussians, optimizer)

    # 7. 日志
    if step % log_interval == 0:
        log_metrics(step, loss, rendered_image, gt_image)
```

---

## 2. 损失函数实现

```python
def compute_loss(render, gt, gaussians):
    # 重建损失
    l1 = torch.abs(render - gt).mean()
    ssim = 1 - ms_ssim(render, gt)  # 多尺度SSIM
    L_img = (1 - λ) * l1 + λ * ssim

    # 正则化
    scale = gaussians.get_scales()  # 各轴尺度
    L_scale = torch.clamp(scale - 1.0, min=0).mean()  # 惩罚过大的尺度
    L_opacity = torch.clamp(gaussians.alpha - 0.99, min=0).mean()  # 惩罚过高的α

    L_total = L_img + λ_scale * L_scale + λ_opacity * L_opacity
    return L_total
```

**注意**：
- `get_scales()`：从协方差 Σ 提取各轴长度（特征值开根）
- 正则化项的 clamp 确保只有"过大"值被惩罚

---

## 3. 密度控制详解

### 3.1 确定哪些高斯需要操作

** Densify（分裂）条件**：
1. 该高斯在某个视角下的**投影尺度** > 阈值（如 0.05 * image_size）
   - 说明3D尺度太大，应该分成两个
2. 且该高斯的**梯度**足够大
   - `|∂L/∂μ| > threshold` 或 `|∂L/∂Σ| > threshold`
   - 说明该高斯对损失影响大，需要更精细表示

**Prune（删除）条件**：
- α < α_threshold（如 0.001）
- 且尺度 < scale_threshold（太小的点可删）

---

### 3.2 Densify操作

**方法A：克隆（Clone）**
- 在当前位置 μ 再放一个同样的高斯
- 用途：在梯度大的位置增加密度

**方法B：分裂（Split）**
- 将一个大高斯沿主方向分成两个
- 新尺度 = 原尺度 / √2
- 新位置 = μ ± 0.01 * 主方向（微小偏移）

**实现**：
```python
def densify(gaussians, grads, grad_threshold, scale_threshold, camera):
    to_clone = []
    to_split = []

    for i, g in enumerate(gaussians):
        # 1. 投影尺度
        proj_scale = project_scale(g.Sigma, camera)  # 投影后的半轴长度（像素）
        max_proj_scale = proj_scale.max()

        # 2. 梯度幅度
        grad_norm = grads[i].norm()

        if max_proj_scale > scale_threshold and grad_norm > grad_threshold:
            if g.Sigma.mean() > 0.01:  # 尺度较大，适合分裂
                to_split.append(i)
            else:  # 尺度小，只克隆
                to_clone.append(i)

    # 执行操作
    for idx in to_split:
        split_gaussian(gaussians, idx)
    for idx in to_clone:
        clone_gaussian(gaussians, idx)
```

---

### 3.3 Prune操作

```python
def prune(gaussians, alpha_threshold, scale_threshold):
    keep_mask = []
    for g in gaussians:
        scale = g.get_scale().max()
        if g.alpha > alpha_threshold and scale > scale_threshold:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    gaussians.mask = keep_mask  # 标记删除
    optimizer.prune(keep_mask)  # 同时清理优化器状态
```

**注意**：删除后，gaussians列表其实没变，只是mask标记。实际实现中可能真的删除数组元素，但优化器参数需要同步。

---

### 3.4 密度控制时机

- 每 **1000步**（或根据数据量调整）执行一次
- 太频繁：计算开销大（需要计算每个高斯的投影尺度和梯度）
- 太稀疏：密度调整不及时，训练不稳定

---

## 4. 训练监控指标

### 4.1 图像质量指标

- **PSNR**（峰值信噪比）：
  ```
  PSNR = 10 * log10(MAX² / MSE)
  ```
  - 越高越好，>30 算不错，>35 优秀

- **SSIM**（结构相似性）：
  - 0~1，越接近1越好

- **LPIPS**（感知相似性）：
  - 越低越好（学习感知图像块相似度）

### 4.2 高斯统计指标

- **高斯数量**：初始N → 最终N'（通常增长2-3倍）
- **平均尺度**：应该稳定在合理范围（如 [0.01, 1.0] 世界坐标单位）
- **平均α**：应该集中在 [0.5, 0.95]

### 4.3 梯度监控

- 记录 |∂L/∂μ|, |∂L/∂Σ| 的分布
- 如果梯度长期很小 → 学习率可能太小或高斯已收敛
- 如果梯度爆炸 → 降低学习率或加梯度裁剪

---

## 5. 训练调度（Scheduling）

### 5.1 学习率调整

**阶段1**（0-7k步）：快速生长
- lr = 1e-2（所有参数）
- 密集densify（每500步）

**阶段2**（7k-15k步）：精细调整
- lr = 1e-3
- densify 变少（每1000步）

**阶段3**（15k+）：微调
- lr = 1e-4
- 不再densify（只prune）

**为什么分阶段？**
- 初始：快速增加高斯数量，覆盖场景
- 中期：稳定优化，避免过度densify
- 后期：微调已有高斯，防止过拟合

---

### 5.2 密度控制调度

-  Densify 只在**前N步**进行（如15k步）
- 之后只 prune（删除冗余高斯）
- 最终高斯数量会稳定

---

## 6. 完整伪代码（整合版）

```python
# 超参数
total_steps = 30000
densify_interval = 1000
densify_from = 500
prune_from = 1500
learning_rates = {
    'mu': 1.6e-4,
    'Sigma': 1e-3,
    'alpha': 5e-2,
    'color': 5e-3,
}

# 初始化
gaussians = init_from_sfm(sfm_points)
optimizer = Adam(gaussians.parameters(), lr=learning_rates)

for step in range(total_steps):
    # 1. 采样
    camera = dataset.sample()
    gt = dataset.get_image(camera)

    # 2. 渲染
    render, radii = render_gaussians(gaussians, camera)  # radii用于可见性判断

    # 3. 损失
    loss, Ll1, Lssim = compute_loss(render, gt, gaussians)

    # 4. 反向
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 5. 梯度缓存（用于densify）
    if step < densify_from or step % densify_interval == 0:
        cache_gradients(gaussians)

    # 6. 密度控制
    if densify_from <= step < prune_from and step % densify_interval == 0:
        densify(gaussians, optimizer, radii, camera)
    if step >= prune_from and step % densify_interval == 0:
        prune(gaussians, optimizer, radii)

    # 7. 学习率调度
    if step in [7500, 15000]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    # 8. 日志
    if step % 100 == 0:
        print(f"Step {step}: loss={loss:.4f}, PSNR={psnr(render, gt):.2f}")
```

---

## 7. 收敛判断

**视觉判断**：
- 渲染图像与GT无明显差异
- 几何结构完整（无空洞或无意义噪点）

**定量判断**：
- PSNR 连续1000步变化 < 0.1 dB
- 高斯数量稳定（不再增长）

**时间**：
- 通常需要 20k-30k 步（取决于数据规模）
- 单GPU（RTX 3090/4090）：~30分钟到2小时

---

## 思考题（独立重推检验）

1. **优化器分组**：为什么不同参数用不同学习率？（μ, Σ, α, c）
2. **densify时机**：为什么要在densify前缓存梯度？为什么densify后要重置优化器状态？
3. **学习率调度**：三次衰减（1e-2 → 1e-3 → 1e-4）分别对应训练什么阶段？
4. **梯度消失**：如果某个高斯长期梯度接近0，可能是什么原因？如何处理？

---

**下一章**：Feed-Forward推理 - 训练完成后，如何实现实时渲染？
