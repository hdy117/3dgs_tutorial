# 第5章：优化目标与损失函数

**学习路径**：`verification`

**核心目标**：理解如何设计损失函数，让高斯学习到正确的几何和外观

---

## 一、引言：优化什么？

**问题**：我们有可微分渲染管线，但优化什么？损失函数如何设计？

**核心思路**：
- **重建损失**：让渲染图像 ≈ 真实图像
- **正则化**：防止高斯退化（无限扁、无限不透明）
- **密度控制**：动态调整高斯数量（densification/pruning）

---

## 二、重建损失（Image Reconstruction Loss）

### 2.1 为什么需要重建损失？

**目标**：最小化渲染图像与Ground Truth图像的差异

**挑战**：
- 像素级差异（L1/L2）可能导致模糊
- 人类视觉对结构更敏感 → 需要结构感知损失

---

### 2.2 L1 Loss

**定义**：
```
L_L1 = |C_pred - C_gt|_1 = Σ_{pixels} |C_i - C_i^gt|
```

**性质**：
- ✅ 对离群点鲁棒
- ✅ 保证颜色不漂移
- ❌ 可能模糊（L1倾向于中等解，不锐利）

**梯度**：
```
∂L_L1/∂C_pred = sign(C_pred - C_gt)
```
- 梯度恒为±1，不会消失
- 但梯度大小固定，不利于精细调整

---

### 2.3 SSIM Loss（结构相似性）

**动机**：
- L1是像素级损失，不考虑局部结构
- 人类视觉系统对结构/纹理更敏感
- 需要感知质量指标

**SSIM公式**（单通道，局部窗口）：
```
SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2) / (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)
```
其中：
- μ: 窗口均值
- σ²: 窗口方差
- σ_xy: 窗口协方差
- C1, C2: 稳定常数（避免除0）

**MSSIM（多尺度SSIM）**：
- 在不同分辨率（如1x, 0.5x, 0.25x）计算SSIM
- 平均得到最终分数
- 更robust，捕捉多尺度结构

**损失**：
```
L_SSIM = 1 - MSSIM(C_pred, C_gt)
```

**为什么加SSIM？**
- 提升视觉质量（纹理更清晰）
- 减少模糊伪影
- 梯度更符合感知

---

### 2.4 总重建损失

3DGS论文公式：
```
L_img = (1 - λ) * L_L1 + λ * L_SSIM
```

**超参数选择**：
- 通常 λ = 0.8（SSIM权重更高）
- 理由：SSIM对质量影响更大，L1保证基础颜色准确

**梯度分析**：
- L1提供稳定梯度（±1）
- SSIM提供感知引导（梯度方向更符合人眼）
- 两者互补

---

## 三、正则化项

### 3.1 尺度正则（Scale Regularization）

**问题**：高斯尺度 Σ 可以任意大或任意小

**不良情况**：
- 尺度无限大 → 高斯覆盖整个图像（严重模糊）
- 尺度无限小 → 退化为点（噪声）

**正则化策略**：
```
L_scale = Σ_i max(0, σ_i - σ_max)
```
其中 σ_i 是高斯的各轴尺度（√特征值）

**为什么只惩罚过大尺度？**
- 小尺度 → 点状，但不会模糊，且易于densify扩大
- 大尺度 → 会"污染"大范围像素，必须限制

**实现**：
```python
scales = gaussians.get_scales()  # (N, 3)
L_scale = torch.clamp(scales - 1.0, min=0).mean()
```

**超参数**：
- σ_max = 1.0（世界坐标单位，需根据场景尺度调整）

---

### 3.2 不透明度正则（Opacity Regularization）

**问题**：α 可以接近 1（完全不透明）或接近 0（完全透明）

**不良情况**：
- α=1 且尺度大 → 不透明大斑块（遮挡问题，无法叠加）
- α=0 → 无贡献，浪费参数

**正则化策略**：
- 鼓励 α 接近 0.5-1 之间的合理值
- 或使用稀疏性诱导（如L1 on α）

**实现**（论文未明确，但实践中常用）：
```python
# 惩罚过高的α
L_opacity = torch.clamp(gaussians.alpha - 0.99, min=0).mean()
# 或惩罚过低的α
L_opacity = torch.clamp(0.01 - gaussians.alpha, min=0).mean()
```

---

## 四、密度控制（Densification & Pruning）

**这是3DGS的核心创新之一**：动态调整高斯数量

### 4.1 为什么需要？

- 初始高斯来自SfM点云 → 通常稀疏且分布不均匀
- 训练中：某些区域高斯过少（重建空洞），某些区域过多（冗余）
- 需要自动调整：不够的地方加，太多的地方删

---

### 4.2 Densify（分裂）策略

**触发条件**（需同时满足）：

1. **投影尺度大**：
   - 该高斯在某个视图下的**投影尺度** > 阈值（如3像素）
   - 说明该高斯在3D空间尺度太大，需要分裂成多个小高斯

2. **梯度大**：
   - |∂L/∂μ| > threshold 或 |∂L/∂Σ| > threshold
   - 说明该位置重建误差大，需要更多表示能力

**操作**：

**方法A：克隆（Clone）**
- 在当前位置 μ 再放一个同样的高斯
- 用途：在梯度大的位置增加密度
- 适用：尺度不大但梯度大的情况

**方法B：分裂（Split）**
- 将一个大高斯沿主方向分成两个
- 新尺度 = 原尺度 / √2
- 新位置 = μ ± 0.01 * 主方向（微小偏移）
- 适用：尺度大且梯度大的情况

**实现**：
```python
def densify(gaussians, grads, radii, grad_threshold=0.0002, scale_threshold=0.01):
    # radii: 投影后的半轴长度（像素）
    large_scale = radii.max(dim=1)[0] > scale_threshold
    large_grad = (grads.mu > grad_threshold) | (grads.Sigma > grad_threshold)
    to_densify = large_scale & large_grad

    for idx in torch.where(to_densify)[0]:
        if gaussians.scale[idx].mean() > 0.01:
            split_gaussian(gaussians, idx)  # 分裂
        else:
            clone_gaussian(gaussians, idx)  # 克隆
```

---

### 4.3 Prune（删除）策略

**触发条件**：
- 高斯不透明度 α 太小（如 < 0.001）
- 且尺度太小（几乎不可见）

**操作**：
- 直接删除该高斯

**为什么能删？**
- α=0 对渲染无贡献
- 尺度小且α小 → 该高斯"死了"，留着浪费内存

**实现**：
```python
def prune(gaussians, alpha_threshold=0.001, scale_threshold=1e-4):
    scales = gaussians.get_scales().max(dim=1)[0]
    keep = (gaussians.alpha.squeeze() > alpha_threshold) & \
           (scales > scale_threshold)
    gaussians.mask = keep
```

---

### 4.4 密度控制频率

- 每 **N次迭代**（如每1000步）执行一次densify/prune
- 太频繁：计算开销大（需要计算每个高斯的投影尺度和梯度）
- 太稀疏：密度调整不及时，训练不稳定

**调度策略**：
- 前期（0-7k步）：每500步densify
- 中期（7k-15k步）：每1000步densify
- 后期（15k+）：只prune，不densify

---

## 五、总体损失函数

**完整训练损失**：
```
L_total = L_img + λ_scale * L_scale + λ_opacity * L_opacity
```

**超参数典型值**：
| 参数 | 典型值 | 作用 |
|------|--------|------|
| λ（SSIM权重） | 0.8 | 平衡L1和SSIM |
| λ_scale | 0.01 | 尺度正则强度 |
| λ_opacity | 0.01 | 不透明度正则 |

---

## 六、训练循环（整合版）

```python
# 超参数
total_steps = 30000
densify_interval = 1000
densify_from = 500
prune_from = 1500
grad_threshold = 0.0002
scale_threshold = 0.01

# 初始化
gaussians = init_from_sfm(sfm_points)
optimizer = Adam([
    {'params': gaussians.mu, 'lr': 1.6e-4},
    {'params': gaussians.Sigma, 'lr': 1e-3},
    {'params': gaussians.alpha, 'lr': 5e-2},
    {'params': gaussians.color, 'lr': 5e-3}
])

for step in range(total_steps):
    # 1. 采样
    camera = dataset.sample()
    gt_image = dataset.get_image(camera)

    # 2. 渲染
    rendered, radii = render_gaussians(gaussians, camera)
    # radii: 每个高斯在屏幕上的投影半径（用于densify判断）

    # 3. 损失
    loss, L1, SSIM = compute_loss(rendered, gt_image, gaussians)

    # 4. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 5. 梯度缓存（用于densify）
    if step < densify_from or step % densify_interval == 0:
        grads_mu = gaussians.mu.grad.detach().norm(dim=1)
        grads_Sigma = gaussians.Sigma.grad.detach().view(N, -1).norm(dim=1)

    # 6. 密度控制
    if densify_from <= step < prune_from and step % densify_interval == 0:
        densify_and_prune(gaussians, optimizer, radii, grads_mu, grads_Sigma,
                          grad_threshold, scale_threshold)
    if step >= prune_from and step % densify_interval == 0:
        prune(gaussians, optimizer, radii)

    # 7. 学习率调度
    if step in [7500, 15000]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    # 8. 日志
    if step % 100 == 0:
        psnr = 10 * torch.log10(1.0 / ((rendered - gt_image)**2).mean())
        print(f"Step {step}: loss={loss:.4f}, PSNR={psnr:.2f}, "
              f"#gauss={len(gaussians)}")
```

---

## 七、超参数调优指南

### 7.1 学习率

| 参数 | 初始LR | 调整时机 | 原因 |
|------|--------|----------|------|
| μ（位置） | 1.6e-4 | 7.5k, 15k步 ×0.1 | 位置需精细调整 |
| Σ（协方差） | 1e-3 | 7.5k, 15k步 ×0.1 | 尺度变化需谨慎 |
| α（不透明度） | 5e-2 | 7.5k, 15k步 ×0.1 | α收敛快，初始LR大 |
| c（颜色） | 5e-3 | 7.5k, 15k步 ×0.1 | 颜色易调 |

**为什么分组？**
- 不同参数尺度不同（μ在world units，α在[0,1]）
- 更新步长应相对一致（Adam自适应但初始LR重要）

---

### 7.2 密度控制阈值

| 阈值 | 典型值 | 作用 |
|------|--------|------|
| grad_threshold | 0.0002 | 梯度多大才densify |
| scale_threshold | 0.01（像素） | 投影多大才densify |
| prune_alpha | 0.001 | α多小才删除 |
| prune_scale | 1e-4 | 尺度多小才删除 |

**调优建议**：
- 如果densify太激进（高斯爆炸）→ 提高 grad_threshold
- 如果densify不够（空洞多）→ 降低 grad_threshold
- 如果高斯太扁（条纹）→ 降低 scale_threshold

---

## 八、思考题（独立重推检验）

1. **损失设计**：为什么不用纯L2损失？SSIM相比L2/L1的优势是什么？
2. **正则化必要性**：如果不加尺度正则，会发生什么？画个草图说明。
3. **密度控制逻辑**：为什么梯度大的高斯才需要densify？如果梯度大但尺度小，该分裂吗？
4. **优化器选择**：为什么用Adam而不是SGD？Adam对高斯参数的尺度差异（μ vs Σ vs α）有何优势？

---

## 九、下一章预告

**第6章**：数据采集与初始化 - 如何从SfM点云得到高斯集合的初始状态？详解COLMAP输出解析、尺度估计、坐标系转换。

---

**关键记忆点**：
- ✅ 重建损失：L_img = (1-λ)L1 + λ·SSIM，λ=0.8
- ✅ 尺度正则：惩罚过大的σ，防止模糊
- ✅ Densify条件：投影尺度大 + 梯度大
- ✅ Prune条件：α太小 或 尺度太小
- 🎯 **核心创新**：自适应密度控制让稀疏性动态调整
