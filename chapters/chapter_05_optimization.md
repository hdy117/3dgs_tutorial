# 第5章：让高斯学会"正确"的事——优化目标的完整设计

**学习路径**：`problem → invention → verification`

**本章核心问题**：有了可微分渲染管线，我们还需要设计**损失函数**来告诉高斯"什么是对的"。怎么设计损失，才能让高斯：
- 学出准确的几何形状（位置、大小、方向）
- 学出准确的表面外观（颜色、透明度）
- 不会学出 degenerate 的情况（无限大/小、不透明斑块）
- 自动调整密度（该密的地方密，该疏的地方疏）

---

## 一、问题的本质：我们到底在优化什么？

### 1.1 目标 vs 手段

**目标**：让渲染图像 `C_pred` 接近真实图像 `C_gt`

**手段**：调整高斯参数 `{μ, Σ, α, c}`

**问题**："接近"怎么定义？
- 像素级准确？（L2 loss）
- 结构相似？（SSIM）
- 锐利边缘？（感知损失）
- 还要防止学出奇怪的高斯？（正则化）

**这不是单一问题**，而是多个目标的权衡。

---

### 1.2 三个核心需求

从第一性原理出发，我们需要满足：

**需求1：重建准确**
- 颜色误差小（像素级）
- 结构清晰（边缘锐利，无模糊）

**需求2：参数健康**
- 高斯尺度合理（不会无限大或无限小）
- 透明度合理（不会全不透明或全透明）
- 协方差正定（数学有效）

**需求3：密度自适应**
- 简单区域：少量高斯
- 细节区域：大量高斯
- 自动发现需要多少高斯

**这三个需求是硬约束，必须同时满足。**

---

## 二、Axioms：从公理推导损失设计

### 公理1：梯度应该指向"改进方向"

**事实**：反向传播时，梯度 ∂L/∂θ 告诉我们"如果θ增加一点点，L会怎么变"

**要求**：
- 如果渲染颜色太暗 → 梯度应该让颜色变亮
- 如果高斯太大（模糊）→ 梯度应该让尺度变小
- 如果高斯太小（噪声）→ 梯度应该让尺度变大

**这是损失函数设计的"箭头方向"公理**：梯度必须有明确的物理意义。

---

### 公理2：局部性 vs 全局性

**局部损失**（如L1）：每个像素独立，梯度稳定，但可能模糊

**全局损失**（如SSIM）：考虑窗口/图像统计，保持结构，但计算量大

**需求**：两者结合，既有局部稳定性，又有全局结构

---

### 公理3：参数需要有"自然范围"

**观察**：某些参数容易跑偏：
- 尺度σ：可以趋向 ∞（完全模糊）或 0（退化为点）
- 透明度α：可以趋向 1（不透明斑块）或 0（无贡献）

**要求**：损失函数应该惩罚极端值，引导参数到"合理范围"

---

### 公理4：密度控制必须可微（或至少稳定）

**问题**：高斯数量N是离散的（整数），怎么"学习"N？

**洞察**：虽然N本身不可微，但我们可以：
- 通过梯度判断"哪里需要更多高斯"（梯度大 → 误差大 → 需要增加表示）
- 通过阈值判断"哪里高斯冗余"（α太小 → 删除）

**密度控制是"离散优化"**，但决策规则可以基于可微信号（梯度）。

---

## 三、Contradictions：损失设计的张力

### 矛盾1：L1 vs L2 vs SSIM

| 损失 | 梯度 | 优点 | 缺点 |
|------|------|------|------|
| L1 | ±1 | 鲁棒，不漂移 | 可能模糊 |
| L2 | 2·Δ | 数学简洁 | 对离群点敏感 |
| SSIM | 感知梯度 | 结构清晰 | 计算稍慢 |

**矛盾**：
- L1稳定但模糊（梯度恒定，无法区分"小误差"和"大误差"的严重性）
- L2能区分误差大小，但离群点会主导梯度
- SSIM保持结构，但梯度复杂，计算慢

**问题**：怎么组合？

---

### 矛盾2：正则化 vs 灵活性

**正则化要惩罚极端值**，但：
- 惩罚太重 → 高斯学不到真实尺度（都被压到中间）
- 惩罚太轻 → 高斯还是可能 degenerate

**问题**：正则化强度怎么选？不同参数（μ, Σ, α, c）需要不同强度吗？

---

### 矛盾3：densify的粒度

**densify触发条件**：投影尺度大 + 梯度大

**矛盾**：
- 如果梯度大但尺度小（已经很精细）→ 要不要分裂？
- 如果梯度小但尺度大（已经很大）→ 要不要分裂？

**问题**：阈值怎么定？动态还是静态？

---

## 四、Solution Path：从需求到具体公式

### 4.1 重建损失：L1 + SSIM 组合

**为什么选这个组合？**

**分析**：
- L1提供稳定梯度（±1），防止离群点主导
- SSIM提供结构感知梯度，保持边缘锐利
- 权重：L1 0.2 + SSIM 0.8（SSIM为主）

**公式**：
```python
L_img = (1 - λ_ssim) * L1 + λ_ssim * (1 - SSIM)
其中 λ_ssim = 0.8
```

**SSIM的梯度特性**：
```
SSIM(x,y) = (2μ_xμ_y + C1)(2σ_xy + C2) / ((μ_x²+μ_y²+C1)(σ_x²+σ_y²+C2))

∂SSIM/∂x 涉及：
  - 均值 μ 的梯度
  - 方差 σ² 的梯度
  - 协方差 σ_xy 的梯度

这是一个局部统计量，梯度会"感知"窗口内的结构一致性。
```

**为什么不用纯L2？**
- 想想：如果一张图有一个像素完全错误（比如传感器噪点），L2会给出巨大梯度，整个图像都会向那个噪点妥协
- L1只给±1，鲁棒性强
- SSIM进一步确保结构不模糊

**需要细化**：SSIM的梯度具体怎么计算？反向传播时如何实现？

---

### 4.2 尺度正则：只惩罚"过大"

**为什么只惩罚过大，不惩罚过小？**

**第一性分析**：
- 尺度太小（σ→0）：高斯退化为点，虽然不理想，但：
  - 它不会"污染"其他像素（影响范围小）
  - 可以通过 densify 自动增加（分裂）
  - 不会破坏整体渲染质量

- 尺度太大（σ→∞）：高斯覆盖整个图像，严重模糊，且：
  - 遮挡其他高斯
  - 无法通过 densify 修复（需要分裂但分裂后还是大）
  - 必须直接惩罚

**惩罚函数**：
```python
# 对每个高斯，计算其特征值（尺度平方）
eigvals = torch.linalg.eigvalsh(Sigma)  # (N, 3)
scales = torch.sqrt(eigvals)  # (N, 3)

# 惩罚超过阈值的尺度
L_scale = torch.clamp(scales - σ_max, min=0).mean()
其中 σ_max = 1.0（世界坐标单位）
```

**为什么用 clamp 而不是平方？**
- clamp 只惩罚超过阈值的部分（线性惩罚）
- 平方惩罚太重（σ=2时惩罚是σ=1的4倍）
- 线性更温和，允许一定程度的"大尺度"（比如表面点需要沿法线拉长）

**思考**：如果场景中有大平面（比如墙），高斯尺度天然会大，这时候惩罚会不会太严？

---

### 4.3 不透明度正则：只惩罚"过高"

**为什么只惩罚过高（α接近1），不惩罚过低？**

**分析**：
- α太低（接近0）：无贡献，浪费参数
  - 但可以通过 prune 直接删除（α<0.001就删）
  - 不需要损失函数惩罚，直接删除更干净

- α太高（接近1）且尺度大：不透明大斑块
  - 会遮挡后面所有高斯
  - 破坏 alpha blending 的层次感
  - 必须惩罚，防止出现

**惩罚函数**：
```python
L_opacity = torch.clamp(alpha - 0.99, min=0).mean()
```

**为什么阈值是0.99？**
- 经验值：α≥0.99 时，透射率 T = Π(1-α) 会迅速降到0
- 这样的高斯实际上"完全不透明"，破坏混合

**需要展开**：α的优化学习率为什么是5e-2（比μ, Σ大10-100倍）？因为α的梯度量级天然较小。

---

### 4.4 总损失架构

**组合方式**：
```python
L_total = L_img + λ_scale * L_scale + λ_opacity * L_opacity
其中：
  L_img = 0.2 * L1 + 0.8 * (1 - SSIM)
  λ_scale = 0.01
  λ_opacity = 0.01
```

**权重选择的直觉**：
- 重建损失是主要目标（权重1.0）
- 正则化是辅助约束（权重0.01），不能主导优化
- 如果正则化权重太大，高斯学不到真实内容（都被"规范"束缚）

**你能想到其他组合吗？**
- 比如 L2 + Perceptual loss？
- 为什么3DGS选择 L1+SSIM？

---

## 五、密度控制：自适应稀疏性的核心

### 5.1 为什么需要动态密度？

**问题**：初始高斯数量固定（比如从SfM点云来），但：
- 简单区域（天空、墙壁）不需要那么多高斯
- 复杂区域（树叶、毛发）需要很多高斯
- 初始分布可能不均匀（SfM点稀疏处需要增加）

**需求**：训练过程中自动调整高斯数量
- 不够的地方增加（densify）
- 冗余的地方删除（prune）

**这是3DGS的核心创新**：稀疏性不是固定的，而是"学出来"的。

---

### 5.2 Axioms of Densify

**公理1：梯度大表示"不理解"**
- 如果某个高斯对渲染误差的梯度很大
- 说明这个高斯"没学好"，或者"不够用"
- 需要增加表示能力

**公理2：投影尺度大表示"3D尺度过大"**
- 一个高斯在屏幕上很大（比如>3像素）
- 说明它在3D空间尺度太大
- 应该分裂成两个小的

**公理3：梯度小+尺度小表示"冗余"**
- 梯度小：这个高斯对误差贡献小（ already good）
- 尺度小：它已经很精细
- 可以删除（如果透明度也低）

---

### 5.3 Densify条件：双条件触发

**条件1：投影尺度阈值**
```python
# 2D投影尺度（像素）
scale_2d = radii  # 来自渲染管线的输出，每个高斯在屏幕上的影响半径

if scale_2d > scale_threshold:  # 默认 0.01 像素（注意：是2D尺度）
    # 该高斯在屏幕上太大
```

**条件2：梯度阈值**
```python
# 梯度范数
grad_mu = gaussians.mu.grad.norm(dim=1)  # (N,)
grad_Sigma = gaussians.Sigma.grad.view(N, -1).norm(dim=1)

if grad_mu.mean() > grad_threshold or grad_Sigma.mean() > grad_threshold:
    # 该高斯学习信号强，需要更多表示
```

**联合条件**：
```python
if scale_2d > scale_thresh and (grad_mu > grad_thresh or grad_Sigma > grad_thresh):
    # 触发 densify
```

**为什么是"且"？**
- 尺度大但梯度小：这个高斯已经稳定但太大 → 应该分裂（不依赖梯度）
- 梯度大但尺度小：这个高斯已经很精细但误差大 → 应该克隆（复制一份增强）
- 实际实现：两个条件都触发时分裂，只触发梯度时克隆

---

### 5.4 Densify操作：克隆 vs 分裂

**克隆（Clone）**：
- 场景：梯度大但尺度小（已经很精细的高斯）
- 操作：直接复制一份
- 效果：在该区域增加高斯数量，但保持原有尺度
- 用途：增强精细区域的表示

```python
def clone_gaussian(i):
    return Gaussian(
        mu = mu[i].clone(),
        Sigma = Sigma[i].clone(),
        alpha = alpha[i].clone(),
        color = color[i].clone()
    )
```

**分裂（Split）**：
- 场景：尺度大（在屏幕上很大）
- 操作：沿主方向分裂成两个
- 原理：大尺度高斯 Σ 的特征值分解 → 最大特征向量方向是"拉长"方向
- 新尺度 = 原尺度 / √2（两个小高斯合起来体积不变）
- 新位置 = μ ± 0.01·主方向（稍微分开）

```python
def split_gaussian(i):
    # 特征值分解
    eigvals, eigvecs = torch.linalg.eigh(Sigma[i])  # Σ = VΛVᵀ
    main_dir = eigvecs[:, -1]  # 最大特征值对应的特征向量
    
    # 新尺度（沿主方向分裂，其他方向不变）
    new_scales = torch.sqrt(eigvals) / np.sqrt(2)
    new_Sigma1 = eigvecs @ diag(new_scales**2) @ eigvecs.T
    new_Sigma2 = new_Sigma1.clone()
    
    # 新位置（沿主方向偏移）
    offset = 0.01 * main_dir
    new_mu1 = mu[i] + offset
    new_mu2 = mu[i] - offset
    
    return Gaussian(new_mu1, new_Sigma1, alpha[i], color[i]), \
           Gaussian(new_mu2, new_Sigma2, alpha[i], color[i])
```

**为什么沿主方向分裂？**
- 高斯是椭球，主方向是"最长"方向
- 沿最长方向分裂，能最有效减小每个高斯的尺度
- 如果随机方向分裂，可能分裂后尺度仍然很大

---

### 5.5 Prune：删除冗余高斯

**条件**：
1. α < α_threshold（默认0.001）：透明度太低，无渲染贡献
2. 尺度 < scale_min（默认1e-6）：数值不稳定，无实际意义

**操作**：直接从张量中删除

```python
def prune(gaussians, alpha_thresh=0.001, scale_min=1e-6):
    # 计算尺度
    eigvals = torch.linalg.eigvalsh(gaussians.Sigma)
    scales = torch.sqrt(eigvals.max(dim=1)[0])
    
    # 掩码：保留条件
    mask = (gaussians.alpha > alpha_thresh) & (scales > scale_min)
    
    # 应用掩码
    gaussians.mu = gaussians.mu[mask]
    gaussians.Sigma = gaussians.Sigma[mask]
    gaussians.alpha = gaussians.alpha[mask]
    gaussians.color = gaussians.color[mask]
```

**为什么 prune 只在后期（15k步后）开始？**
- 前期：高斯数量不足，需要 densify 增加
- 后期：高斯数量趋于稳定，开始清理冗余
- 如果 prune 太早，会把"有潜力但还没学好"的高斯删掉

---

### 5.6 调度策略：三阶段

**阶段1：快速生长（0-7.5k步）**
- densify 频率：每500步
- prune：关闭
- 目标：快速增加高斯数量，覆盖场景

**阶段2：精细调整（7.5k-15k步）**
- densify 频率：每1000步
- prune：关闭
- 目标：稳定优化，避免过度densify

**阶段3：稳定收敛（15k-30k步）**
- densify：关闭
- prune：每1000步
- 目标：清理冗余，最终确定高斯数量

**为什么这样调度？**
- 训练是"先扩张后收缩"的过程
- 类似神经网络的"先过拟合再正则化"
- 前期不怕多（densify），后期需要精简（prune）

**你能想到更好的调度吗？** 比如根据验证集PSNR动态决定？

---

## 六、完整训练循环：所有部分如何串联

### 6.1 超参数全景

```python
# 损失权重
λ_ssim = 0.8
λ_scale = 0.01
λ_opacity = 0.01

# Densify 阈值
grad_threshold = 0.0002  # 梯度范数阈值
scale_threshold = 0.01   # 2D投影尺度阈值（像素）

# Prune 阈值
α_threshold = 0.001
scale_min = 1e-6

# 学习率（分参数组）
lr_mu = 1.6e-4
lr_Sigma = 1e-3
lr_alpha = 5e-2
lr_color = 5e-3

# 调度
total_steps = 30000
densify_from = 500
densify_interval = 1000
prune_from = 15000
lr_decay_steps = [7500, 15000]
lr_decay_factor = 0.1
```

**学习率差异的直觉**：
- α 需要大学习率（5e-2）因为其梯度小（α∈[0,1]，变化范围小）
- μ 需要小学习率（1.6e-4）因为位置精度重要
- Σ 中等（1e-3）

---

### 6.2 训练循环伪代码（详细版）

```python
# 初始化
gaussians = initialize_from_sfm(pointcloud)  # 第6章详解
optimizer = torch.optim.Adam([
    {'params': gaussians.mu, 'lr': lr_mu},
    {'params': gaussians.Sigma, 'lr': lr_Sigma},
    {'params': gaussians.alpha, 'lr': lr_alpha},
    {'params': gaussians.color, 'lr': lr_color}
])

# 调度器（可选）
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=lr_decay_steps, gamma=lr_decay_factor
)

for step in range(total_steps):
    # 1. 数据采样
    gt_image, camera = dataset[step % len(dataset)]
    
    # 2. 前向渲染
    rendered, radii = render(gaussians, camera)  # radii: 每个高斯的2D投影半径
    
    # 3. 计算损失
    L1 = torch.abs(rendered - gt_image).mean()
    L_ssim = 1 - ssim(rendered, gt_image)  # 多尺度SSIM
    L_img = (1 - λ_ssim) * L1 + λ_ssim * L_ssim
    
    # 正则化
    eigvals = torch.linalg.eigvalsh(gaussians.Sigma)
    scales = torch.sqrt(eigvals)
    L_scale = torch.clamp(scales - 1.0, min=0).mean()
    L_opacity = torch.clamp(gaussians.alpha - 0.99, min=0).mean()
    
    L_total = L_img + λ_scale * L_scale + λ_opacity * L_opacity
    
    # 4. 反向传播
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
    
    # 5. 缓存梯度（用于densify决策）
    if step % densify_interval == 0:
        grads_mu = gaussians.mu.grad.detach().norm(dim=1)  # (N,)
        grads_Sigma = gaussians.Sigma.grad.detach().view(len(gaussians), -1).norm(dim=1)
    
    # 6. 密度控制
    if step % densify_interval == 0:
        if densify_from <= step < prune_from:
            # Densify & Prune
            # 注意：prune只删α太小或尺度太小的，不基于梯度
            mask = (gaussians.alpha > α_threshold) & \
                   (torch.sqrt(torch.linalg.eigvalsh(gaussians.Sigma).max(dim=1)[0]) > scale_min)
            
            # Densify 条件
            densify_mask = (radii > scale_threshold) & \
                           ((grads_mu > grad_threshold) | (grads_Sigma > grad_threshold))
            
            # 执行 densify（克隆或分裂）
            new_gaussians = []
            for i in torch.where(densify_mask)[0]:
                eigvals, eigvecs = torch.linalg.eigh(gaussians.Sigma[i])
                if torch.sqrt(eigvals).max() > 0.01:  # 尺度大，分裂
                    g1, g2 = split_gaussian(i)
                    new_gaussians.extend([g1, g2])
                else:  # 尺度小，克隆
                    new_gaussians.append(clone_gaussian(i))
            
            # 合并
            if new_gaussians:
                gaussians.extend(new_gaussians)
                # 重新构建 optimizer（因为参数数量变了）
                optimizer = torch.optim.Adam([
                    {'params': gaussians.mu, 'lr': lr_mu},
                    {'params': gaussians.Sigma, 'lr': lr_Sigma},
                    {'params': gaussians.alpha, 'lr': lr_alpha},
                    {'params': gaussians.color, 'lr': lr_color}
                ])
            
            # Apply prune mask
            gaussians = gaussians[mask]
            # 重新构建 optimizer
            optimizer = torch.optim.Adam([...])
            
        elif step >= prune_from:
            # 只 prune，不 densify
            mask = (gaussians.alpha > α_threshold) & \
                   (torch.sqrt(torch.linalg.eigvalsh(gaussians.Sigma).max(dim=1)[0]) > scale_min)
            gaussians = gaussians[mask]
            optimizer = torch.optim.Adam([...])
    
    # 7. 学习率调度
    scheduler.step()
    
    # 8. 日志
    if step % 100 == 0:
        psnr = 10 * torch.log10(1.0 / L1.item())
        print(f"Step {step}: L={L_total:.4f}, PSNR={psnr:.2f}, #gauss={len(gaussians)}")
```

**关键点**：densify/prune 会改变高斯数量，需要**重建 optimizer**（因为参数张量长度变了）。

---

## 七、思考题（第一性原理式）

**1. 损失函数的设计哲学**
- L1+SSIM 组合中，SSIM权重0.8，L1权重0.2。如果SSIM权重降到0.5，图像会有什么变化？
- 尝试推导：SSIM的梯度在什么情况下会"忽略"小误差？什么时候会"敏感"？

**2. 正则化的边界**
- 尺度正则只惩罚"过大"，但如果有场景确实需要大尺度高斯（比如远处的山），怎么办？
- 不透明度正则阈值0.99是固定的，如果场景中有半透明物体（玻璃、烟雾），这个阈值会误伤吗？

**3. 密度控制的"智能"**
- 当前densify基于"投影尺度大+梯度大"。如果某个高斯梯度大但投影尺度小（已经很精细），克隆它会怎样？
- 分裂时沿主方向，但如果高斯接近球形（各向同性），主方向不稳定，怎么办？

**4. 学习率的分组逻辑**
- 估算 μ, Σ, α, c 的梯度量级差异。为什么α需要5e-2，而μ只需要1.6e-4？
- 如果所有参数用相同学习率，会出现什么问题？

**5. 训练动态的"经济性"**
- 为什么前期densify频繁（每500步），后期稀疏（每1000步或不densify）？
- 从"欠拟合→过拟合→正则化"的角度，解释这个调度策略。

---

## 八、本章核心记忆点

✅ **总损失**：`L = 0.2·L1 + 0.8·(1-SSIM) + 0.01·L_scale + 0.01·L_opacity`

✅ **重建损失选择**：L1鲁棒 + SSIM结构感知 → 兼顾稳定与质量

✅ **正则化**：尺度只惩罚过大（防止模糊），不透明度只惩罚过高（防止不透明斑块）

✅ **Densify条件**：投影尺度大（2D>阈值）+ 梯度大（学习信号强）
  - 尺度大 → 分裂（沿主方向）
  - 尺度小 → 克隆

✅ **Prune条件**：α太小（<0.001）或 尺度太小（<1e-6）

✅ **调度策略**：三阶段（快速生长→精细调整→稳定收敛）

✅ **关键洞察**：密度控制是"离散优化"，但决策基于可微信号（梯度）和几何信号（投影尺度）

---

**下一章**：我们将进入**数据准备**——如何从SfM点云初始化高斯？COLMAP输出怎么解析？初始尺度怎么估计？坐标系如何对齐？这些"脏活"决定了训练能否成功起步。