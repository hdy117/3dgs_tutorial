# 第7章：训练闭环——从初始化到收敛的完整迭代

**学习路径**：`problem → invention → verification`

**本章核心问题**：初始化完成后，怎么设计训练循环让高斯"学会"正确的几何和外观？密度控制什么时候介入？学习率怎么调度？怎么判断训练成功了还是失败了？

---

## 一、问题的本质：训练循环是什么？

### 1.1 训练 vs 推理

**推理**（我们已经有了）：
```
输入：相机参数
输出：渲染图像
操作：前向传播（投影+混合）
```

**训练**（还需要什么）：
```
输入：相机参数 + 真实图像
输出：更新后的高斯参数
操作：前向 → 计算损失 → 反向传播 → 优化器更新 → 密度控制
```

**训练循环是"闭环"**：输出（渲染）要和真实图像比较，误差驱动参数更新。

---

### 1.2 三个核心需求

从第一性原理，训练循环必须满足：

**需求1：梯度有效**
- 反向传播的梯度必须指向"改进方向"
- 如果梯度消失或爆炸，训练失败

**需求2：密度自适应**
- 高斯数量不能固定，要动态调整
- 但调整必须基于有效信号（梯度+几何）

**需求3：收敛可判断**
- 需要监控指标（PSNR、损失、高斯数量）
- 需要明确收敛标准（什么情况下停止）

**这三个需求是训练成功的必要条件。**

---

## 二、Axioms：训练循环的设计公理

### 公理1：梯度需要稳定传递

**事实**：3DGS渲染管线包含多个可微操作（投影、混合），但：
- 排序不可微（argsort）
- 密度控制是离散操作（增删高斯）

**要求**：
- 排序在训练时应该"稳定"，不能每帧都乱序
- 密度控制不应该破坏梯度流（通过缓存机制）

**结论**：训练时每N步才重排一次，期间顺序固定；densify/prune 使用缓存的梯度，不干扰当前优化。

---

### 公理2：密度控制需要"学习信号"

**问题**：什么时候增加高斯？什么时候删除？

**洞察**：
- **梯度大**：该高斯对误差贡献大 → 需要更多表示 → 增加
- **投影尺度大**：该高斯在3D空间太大 → 应该分裂
- **透明度低**：该高斯无贡献 → 删除

**结论**：densify/prune 决策应该基于**可微信号**（梯度范数、投影半径），而不是启发式规则。

---

### 公理3：学习率需要阶段适应性

**观察**：
- 初期：参数远离最优，需要大LR快速调整
- 中期：接近最优，需要小LR精细调整
- 后期：微调，需要更小LR防止过拟合

**结论**：学习率调度应该是**递减**的，通常在特定步数衰减。

---

## 三、Contradictions：训练循环中的张力

### 矛盾1：优化器状态 vs 参数数量变化

**问题**：
- Adam 优化器维护状态（动量、方差），数量与参数成正比
- densify/prune 会改变高斯数量N
- 优化器状态怎么更新？

**方案对比**：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A | 每步重建优化器 | 简单，状态对齐 | 浪费（Adam状态重建开销） |
| B | 仅增删对应参数的状态 | 高效 | 实现复杂 |
| C | 用SGD（无状态） | 无需管理状态 | 收敛可能慢 |

**3DGS选择**：方案A（每densify/prune后重建优化器），因为：
- 实现简单
- densify/prune 每1000步才一次，开销可接受
- Adam状态重建代价 < 训练开销

---

### 矛盾2：梯度缓存 vs 实时性

**问题**：
- densify 需要梯度信息（判断哪些高斯需要分裂/克隆）
- 但梯度是反向传播后立即存在的
- 为什么需要"缓存"？不能实时用吗？

**答案**：
- 实时梯度是**当前步**的，但 densify 决策应该基于**最近N步**的平均梯度
- 如果只用当前步，噪声大（batch sampling 随机性）
- 缓存最近N步的梯度范数，取平均，更稳健

**实现**：
```python
if step % densify_interval == 0:
    grads_mu = gaussians.mu.grad.detach().norm(dim=1)  # (N,)
    grads_Sigma = gaussians.Sigma.grad.detach().view(N, -1).norm(dim=1)
    # 存储到缓存，后续densify用
```

---

### 矛盾3：全局排序 vs 局部渲染

**问题**：
- Alpha blending 需要从后往前混合（全局深度顺序）
- 但 Tile-based 渲染是每tile独立处理
- 如何保证每个tile内的高斯顺序是全局顺序的子集？

**解决方案**：
1. 每帧渲染前，**全局排序一次**（所有高斯按深度）
2. Tile映射时，从排序后的列表中分配高斯
3. 每个tile内的高斯自然保持全局顺序

**关键**：排序只在阶段1做一次，阶段2-4使用这个顺序。

**性能**：排序 O(N log N) ≈ 5-10ms（N=1M），可接受。

---

## 四、Solution Path：完整训练循环设计

### 4.1 状态机

```
[初始化]
   ↓
[循环开始] step=0
   ↓
   ├─→ 采样视角 (gt_image, camera)
   ├─→ 前向渲染 (rendered, radii)
   ├─→ 计算损失 (L_total, L1, L_ssim)
   ├─→ 反向传播 (loss.backward())
   ├─→ 优化器更新 (optimizer.step())
   ├─→ 缓存梯度 (if step % interval == 0)
   ├─→ 密度控制 (if step % interval == 0)
   ├─→ 学习率调度 (if step in decay_steps)
   └─→ 日志/验证 (if step % log_interval == 0)
   ↓
   step += 1
   ↓
   [收敛?] → 是 → 结束
          → 否 → 继续循环
```

---

### 4.2 详细算法（伪代码）

```python
# 超参数配置
config = {
    "total_steps": 30000,
    "log_interval": 100,
    "densify_interval": 1000,
    "densify_from": 500,
    "prune_from": 15000,
    "lr_decay_steps": [7500, 15000],
    "lr_decay_factor": 0.1,
    "grad_threshold": 0.0002,
    "scale_threshold": 0.01,  # 像素
    "prune_alpha": 0.001,
    "prune_scale": 1e-6,
    "max_gaussians": 2e6,
    "lambda_ssim": 0.8,
    "lambda_scale": 0.01,
}

# 初始化
gaussians = init_from_sfm(...)  # 第6章
optimizer = torch.optim.Adam([
    {"params": gaussians.mu, "lr": 1.6e-4},
    {"params": gaussians.Sigma, "lr": 1e-3},
    {"params": gaussians.alpha, "lr": 5e-2},
    {"params": gaussians.color, "lr": 5e-3},
])

# 缓存（用于densify）
grads_mu_cache = None
grads_Sigma_cache = None
radii_cache = None

for step in range(config["total_steps"]):
    # 1. 采样
    gt_image, camera = dataset.sample()
    
    # 2. 前向渲染
    rendered, radii = render(gaussians, camera)  # radii: (N,)
    
    # 3. 损失
    L1 = F.l1_loss(rendered, gt_image)
    L_ssim = 1 - ms_ssim(rendered, gt_image)
    L_img = (1 - config["lambda_ssim"]) * L1 + config["lambda_ssim"] * L_ssim
    
    # 正则化
    scales = gaussians.get_scales()  # (N,3)
    L_scale = torch.clamp(scales - 1.0, min=0).mean()
    
    L_total = L_img + config["lambda_scale"] * L_scale
    
    # 4. 反向传播
    optimizer.zero_grad()
    L_total.backward()
    
    # 5. 缓存梯度（用于densify）
    if step % config["densify_interval"] == 0:
        grads_mu_cache = gaussians.mu.grad.detach().norm(dim=1)  # (N,)
        grads_Sigma_cache = gaussians.Sigma.grad.detach().view(len(gaussians), -1).norm(dim=1)
        radii_cache = radii.detach()  # (N,)
    
    # 6. 优化器更新
    optimizer.step()
    
    # 7. 密度控制
    if step % config["densify_interval"] == 0 and step >= config["densify_from"]:
        # 7.1 Densify
        if step < config["prune_from"]:
            # 条件：投影尺度大 + 梯度大
            scale_condition = radii_cache > config["scale_threshold"]
            grad_condition = (grads_mu_cache > config["grad_threshold"]) | \
                             (grads_Sigma_cache > config["grad_threshold"])
            
            densify_mask = scale_condition & grad_condition
            
            # 执行 densify（分裂或克隆）
            new_gaussians = []
            indices = torch.where(densify_mask)[0]
            for i in indices:
                # 判断尺度
                scale_i = scales[i].max()
                if scale_i > 0.01:  # 尺度大，分裂
                    g1, g2 = split_gaussian(gaussians, i)
                    new_gaussians.extend([g1, g2])
                else:  # 尺度小，克隆
                    new_gaussians.append(clone_gaussian(gaussians, i))
            
            # 添加新高斯
            if new_gaussians:
                gaussians.extend(new_gaussians)
                # 重建优化器（参数数量变了）
                optimizer = rebuild_optimizer(gaussians, config)
        
        # 7.2 Prune（后期才开启）
        if step >= config["prune_from"]:
            # 条件：α太小 或 尺度太小
            alpha_condition = gaussians.squeeze_alpha() < config["prune_alpha"]
            scale_condition = scales.max(dim=1)[0] < config["prune_scale"]
            prune_mask = alpha_condition | scale_condition
            
            # 删除
            if prune_mask.any():
                gaussians = gaussians[~prune_mask]
                # 重建优化器
                optimizer = rebuild_optimizer(gaussians, config)
        
        # 7.3 数量上限保护
        if len(gaussians) > config["max_gaussians"]:
            # 随机删除到上限
            n_remove = len(gaussians) - config["max_gaussians"]
            indices_to_remove = torch.randperm(len(gaussians))[:n_remove]
            mask = torch.ones(len(gaussians), dtype=torch.bool)
            mask[indices_to_remove] = False
            gaussians = gaussians[mask]
            optimizer = rebuild_optimizer(gaussians, config)
    
    # 8. 学习率调度
    if step in config["lr_decay_steps"]:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= config["lr_decay_factor"]
    
    # 9. 日志
    if step % config["log_interval"] == 0:
        psnr = 10 * torch.log10(1.0 / L1.item())
        print(f"Step {step}: PSNR={psnr:.2f}, Loss={L_total:.4f}, "
              f"#gauss={len(gaussians)}, LR={optimizer.param_groups[0]['lr']:.2e}")
    
    # 10. 验证（可选，每1000步跑一次验证集）
    if step % 1000 == 0:
        validate(gaussians, val_dataset)
```

---

### 4.3 辅助函数

```python
def split_gaussian(gaussians, idx):
    """分裂高斯（沿主方向）"""
    Sigma = gaussians.Sigma[idx]  # (3,3)
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    
    # 主方向（最大特征值）
    main_dir = eigvecs[:, -1]  # (3,)
    
    # 新尺度 = 原尺度 / √2
    new_scales = torch.sqrt(eigvals) / np.sqrt(2)
    new_Sigma = eigvecs @ torch.diag(new_scales**2) @ eigvecs.T
    
    # 新位置 = μ ± 0.01·主方向
    offset = 0.01 * main_dir
    mu1 = gaussians.mu[idx] + offset
    mu2 = gaussians.mu[idx] - offset
    
    # 其他参数复制
    alpha = gaussians.alpha[idx]
    color = gaussians.color[idx]
    
    g1 = Gaussian(mu1, new_Sigma, alpha, color)
    g2 = Gaussian(mu2, new_Sigma, alpha, color)
    return g1, g2

def clone_gaussian(gaussians, idx):
    """克隆高斯（直接复制）"""
    return Gaussian(
        mu=gaussians.mu[idx].clone(),
        Sigma=gaussians.Sigma[idx].clone(),
        alpha=gaussians.alpha[idx].clone(),
        color=gaussians.color[idx].clone(),
    )

def rebuild_optimizer(gaussians, config):
    """重建Adam优化器（参数数量变化后）"""
    return torch.optim.Adam([
        {"params": gaussians.mu, "lr": config["lr_mu"]},
        {"params": gaussians.Sigma, "lr": config["lr_Sigma"]},
        {"params": gaussians.alpha, "lr": config["lr_alpha"]},
        {"params": gaussians.color, "lr": config["lr_color"]},
    ])
```

---

## 五、收敛判断与诊断

### 5.1 收敛标准（定量）

```
指标1: PSNR
  - 连续1000步变化 < 0.1 dB
  - 且当前 PSNR > 30（合成数据）或 > 25（实景）

指标2: 高斯数量N
  - 连续N步不变（或波动 < 1%）
  - 且 N 在合理范围（10⁵-10⁶）

指标3: 损失L_total
  - 连续1000步变化 < 0.001
```

**满足以上任意两条** → 认为收敛。

---

### 5.2 训练失败诊断表

```
+------------+----------+----------+----------+----------+
| 症状       | 可能原因 | 检查点   | 解决方案 | 优先级 |
+------------+----------+----------+----------+--------+
| PSNR不升   | LR太小   | grads范数 | LR×10    | 高     |
|            | 初始化差 | 初始渲染  | scale_factor×5-10 | 高 |
|            | 梯度消失 | grads范数 | 检查Σ是否奇异 | 中 |
| 高斯爆炸   | densify太激进 | N增长曲线 | grad_threshold↑, scale_threshold↑ | 高 |
| (N>10M)    |          |          | max_gaussians限制 | 中 |
| 渲染全黑   | Σ太小    | scales.mean() | scale_factor×10 | 高 |
| 渲染全白   | Σ太大    | scales.mean() | scale_factor÷2 | 高 |
| 条纹伪影   | Σ奇异    | det(Sigma) | Σ += I·1e-8 | 中 |
| 颜色漂移   | 颜色LR太大 | color.grad | 降低c的LR | 低 |
| 空洞不愈合 | densify不触发 | grads_mu_cache | grad_threshold↓ | 中 |
+------------+----------+----------+----------+----------+
```

---

### 5.3 监控指标可视化

**必须记录的曲线**（TensorBoard或WandB）：
1. PSNR（训练集、验证集）
2. 损失（L_total, L1, L_ssim, L_scale）
3. 高斯数量N
4. 参数统计（μ、Σ、α、c的均值/方差）
5. 梯度范数（μ、Σ、α、c）

**诊断时看**：
- PSNR曲线：是否平滑上升？有无震荡？
- N曲线：是否先升后稳？有无爆炸？
- 梯度范数：是否逐渐变小（收敛）？有无归零（死亡）？

---

## 六、思考题（第一性原理式）

**1. 为什么densify用缓存的梯度，不用当前步的梯度？**
- 当前步梯度受batch采样随机性影响大
- 缓存N步平均更稳健
- 但如果缓存间隔太长（比如10000步），会错过重要信号吗？
- 推导：缓存间隔与batch size的关系

**2. 学习率调度的"经济学"**
- 三阶段衰减：1e-2 → 1e-3 → 1e-4
- 为什么不是线性衰减？指数衰减的数学依据？
- 如果场景特别复杂（比如大森林），需要更多阶段吗？

**3. 优化器选择：Adam vs SGD**
- 当前用Adam，自适应LR
- 如果换成SGD（固定LR，带动量），会怎样？
- 从梯度量级差异（μ、Σ、α、c）分析，SGD需要多少手动调LR？

**4. 早停策略**
- 当前30k步固定，但不同场景收敛速度不同
- 如何设计**自适应早停**？
  - 如果PSNR连续1000步变化<0.05，停
  - 如果N连续1000步不变，停
  - 但可能陷入局部最优，怎么办？
- 你能想到更智能的停止条件吗？（比如验证集PSNR下降）

**5. 梯度消失的"复活"机制**
- 如果某个高斯梯度长期为0（"死亡"），怎么让它恢复？
- 方案：随机重置梯度大的高斯？或基于密度估计添加新高斯？
- 从信息论角度：梯度为0意味着该参数对损失不敏感，如何重新激活？

---

## 七、本章核心记忆点

✅ **训练循环状态机**：采样 → 渲染 → 损失 → 反向 → 更新 → densify/prune → LR调度 → 日志

✅ **密度控制时机**：每1000步，缓存梯度+投影半径，基于双条件（尺度+梯度）决策

✅ **学习率调度**：三阶段衰减（7.5k、15k步 ×0.1），分参数组（μ、Σ、α、c不同LR）

✅ **收敛判断**：PSNR连续1000步变化<0.1 + N稳定

✅ **关键洞察**：densify/prune 需要缓存梯度（避免噪声），优化器需要重建（参数数量变）

---

**下一章**：训练完成后，如何实现**实时推理**（<10ms/帧）？我们将详解推理优化策略：CUDA kernel融合、内存布局、batch推理、以及如何部署到生产环境。
