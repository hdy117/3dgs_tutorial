# Ch03 — 无约束优化 (SGD)：为什么随机噪声能带我们走出局部极小值？

> **本章目标**：理解现代机器学习的引擎——随机梯度下降 (SGD)。  
> **前置知识**：Ch01 (凸性), Ch02 (梯度与泰勒展开)。  
> **核心问题**：如果你有一个包含一千万个像素的 Loss 函数，为什么不用所有数据算梯度，而是只随机抽几百个（Batch）来算？

---

## 🎯 问题驱动：全量计算 vs 随机采样

### 场景 1：3DGS 的训练循环

```python
# 你的训练数据集有 N=50,000 张图片 (包含数亿个像素)。
# 每一轮优化 (Epoch) 要做的事情:

for step in range(Iterations):
    # ❌ 笨办法 (Batch Gradient Descent, BGD):
    loss = 0
    for all_pixels in entire_dataset:
        loss += L(pred_pixel, gt_pixel)
    gradient = calculate_gradient(loss)  # ←←← 算一次要跑完整个数据集！太慢了！
    
    # ✅ 聪明办法 (Stochastic GD, SGD):
    batch = random_sample(dataset) # 只抽 1024 个像素!
    loss_batch = L(batch.pred, batch.gt)
    gradient = calculate_gradient(loss_batch) # ←←← 快 50,000 倍!
```

**关键问题**：用一小部分数据 (Batch) 算出来的梯度，真的是指向"全局最优"的方向吗？还是说这只是个随机误差？

### 答案：**SGD (Stochastic Gradient Descent)** —— "带着噪声的下山"

---

## 📐 Part 1: SGD 的数学定义与核心性质

### Boxed Result：SGD 的更新公式

$$\boxed{\mathbf{x}_{k+1} = \mathbf{x}_k - \eta_k \cdot \nabla L_{batch}(\mathbf{x}_k)}$$

其中 $L_{batch}$ 是基于随机采样 Batch 计算出来的**近似梯度**。

### 💡 核心洞察：SGD 的本质是无偏估计 (Unbiased Estimation)

假设总数据集的 Loss 是 $L_{total}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N l_i(\mathbf{x})$。
如果我们随机抽取一个 Batch $\mathcal{B}$，其平均梯度为 $\nabla L_{batch} = \frac{1}{|\mathcal{B}|} \sum_{j \in \mathcal{B}} \nabla l_j(\mathbf{x})$。

**期望等于真值 (Jensen 不等式的逆应用)**：
$$\boxed{E[\nabla L_{batch}] = \nabla L_{total}}$$

这意味着：**虽然单次 SGD 的梯度是"歪"的（带着噪声），但它的平均方向绝对正确！** 🎯

---

## 🔥 Part 2: 为什么我们需要"随机噪声"？(Escape Saddle Points)

### 直觉：噪声帮你"跳坑"

如果你在一个完美的凸函数里，GD (确定性梯度下降) 是最优的。
但在 3DGS 这种**非凸 (Non-convex)** 世界里，到处都是小水坑（局部极小值）和鞍点（Saddle Points）。
* **鞍点**：某个方向是谷底，另一个方向是山脊。在鞍点上梯度为 $\nabla f = 0$。确定性 GD 会在这里死锁！

### Boxed Result：随机扰动的逃逸能力

假设我们在一个一维的鞍点附近：$f(x) \approx -x^2/2$ (负曲率)。
* **GD**: $x_{k+1} = x_k$. 如果 $x_k=0$，永远不动。
* **SGD**: $x_{k+1} = x_k - \eta (-x_k + \text{noise}) = x_k(1+\eta) + \text{noise\_term}$。

由于噪声的存在，SGD 在鞍点处的更新方向是随机的。根据随机游走理论 (Random Walk)，只要有非零的负曲率方向，**SGD 最终一定会从鞍点"跳"出来**。 ∎

---

## 💻 Part 3: PyTorch 验证 — SGD vs GD 的对比实验

```python
import torch

# ============================================================
# 1. 构造一个带局部极小值的非凸函数 (双峰函数)
# f(x) = x^4/4 - x^2/2 + x/3  (在 x=-0.5, x=0.8 附近有两个坑)
# ============================================================
def f(x): return x**4 / 4 - x**2 / 2 + x / 3

# 解析梯度
def grad_f(x): return x**3 - x + 1/3

# GD: 确定性步长 (只看大方向)
# SGD: 随机噪声步长 (会抖动，但能跳出坑)

print("=== SGD vs GD 对比实验 ===")
start_x = torch.tensor(0.4) # 起点在两个坑之间的斜坡上

# --- GD (模拟 BGD) ---
x_gd = start_x.clone()
lr = 0.1
for i in range(20):
    dx = grad_f(x_gd)
    x_gd -= lr * dx
    
# --- SGD (模拟随机梯度，带噪声) ---
x_sgd = start_x.clone()
for i in range(20):
    # 在真实梯度基础上加上高斯噪声
    noisy_grad = grad_f(x_sgd) + torch.randn(1, dtype=torch.float32) * 0.2 
    x_sgd -= lr * noisy_grad

print(f"起始点: {start_x.item():.4f}")
print(f"GD 最终位置: {x_gd.item():.4f} (Loss={f(x_gd):.4f})")
print(f"SGD 最终位置: {x_sgd.item():.4f} (Loss={f(x_sgd):.4f})")

# 💡 注意：由于噪声，SGD 可能会跳到另一个更深的坑里！
# 这就是为什么 SGD 虽然慢，但能找到比 GD 更好的全局解。

# ============================================================
# 2. PyTorch 内置 SGD (最简写法)
# ============================================================
from torch.optim import SGD

params = torch.tensor([0.4], dtype=torch.float32, requires_grad=True)
optimizer = SGD([params], lr=lr) # 创建一个 SGD 优化器

for i in range(10):
    optimizer.zero_grad()
    loss = f(params).float()
    loss.backward()
    optimizer.step() # ←←← 这就是执行 x -= lr * grad
    
print(f"\nPyTorch SGD 结果: {params.item():.4f}")

# ============================================================
# 3. 学习率衰减 (Learning Rate Decay) —— 收敛的秘诀
# ============================================================
print("\n=== LR Decay 模拟 ===")
x_decay = start_x.clone()
for i in range(20):
    # 随着训练进行，逐步缩小步长，避免在最低点附近震荡
    current_lr = lr / (1 + 0.1 * i) 
    dx = grad_f(x_decay) + torch.randn(1, dtype=torch.float32) * 0.2
    x_decay -= current_lr * dx
    
print(f"带 LR Decay 的 SGD 最终位置: {x_decay.item():.4f}")
```

---

## 🗺️ Part 4: 与 3DGS 的衔接点 — 为什么 Adam 是标配？

### 核心问题：SGD 的致命弱点 (The "Saddle Point" Trap in High Dimensions)

在高维参数空间（如 3DGS 有几百万个维度），**鞍点的数量远多于局部极小值**。
* **SGD**: 虽然靠噪声能跳出来，但效率很低。它需要精心调节学习率（LR Warmup + Decay）。如果 LR 一开始太大，模型会直接发散 (Loss = NaN)。

### 💡 优化器的进化树 (Optimizer Evolution)

| 优化器 | 核心创新 | 物理直觉 |
|--------|----------|----------|
| **GD** (Ch02) | 用全部数据算梯度 | "全知全能"但太慢，容易死锁在鞍点。 |
| **SGD** (本章) | 随机采样 + 噪声 | "盲人摸象"，靠噪声乱撞逃出鞍点。简单粗暴。 |
| **Momentum** | 加上历史梯度的惯性 $\beta \cdot v_{k}$ | "下坡时加速，上坡时减速"。帮助冲过平缓的鞍点区域。 |
| **Adam** (下一章) | Momentum + RMSProp (自适应 LR) | **"给每个参数单独配一个导航仪"**：在陡峭处自动降速，在平缓处自动提速。3DGS 的黄金搭档！ |

### ✅ Boxed Result：3DGS 的默认配置

```python
# 3DGS 官方训练代码中常用的配置
optimizer = Adam([params], lr=1.6e-4) 
# 注意：不用 SGD，因为 3DGS 的参数空间极其病态 (Ill-conditioned)。
# Adam 能自动处理不同 Splat 之间梯度量级的巨大差异。
```

---

## 🎓 本章小结

### 核心公式

$$\boxed{E[\nabla L_{batch}] = \nabla L_{total} \quad (\text{SGD 的无偏性保证})}$$

$$\boxed{\mathbf{x}_{k+1} = \mathbf{x}_k - \eta_k \cdot \nabla L_{batch}(\mathbf{x}_k) + \text{Noise}}$$

### 关键洞察

> **SGD 不是"近似"，而是一种探索策略** —— 噪声帮助我们在非凸的 Loss 曲面中随机游走，从而跳出局部的陷阱。
> 
> **"无偏估计"是 SGD 的灵魂**：只要保证每次采样的期望等于真实梯度，理论上我们就能收敛到最优解（即使过程充满了抖动）。
> 
> **3DGS 为什么选 Adam？** 因为 Splat 的协方差矩阵 $\Sigma$ 和位置 $\mu$ 的优化难度完全不同。SGD 需要一个统一的学习率，而 Adam 能自动为它们分配不同的步长。

---

## 📚 习题

### ✅ 基础题

**3.1** 为什么 SGD 不需要一次性遍历整个数据集就能开始训练？
<details>
<summary>💡 提示</summary>
因为梯度的期望是无偏的 ($E[\nabla_{batch}] = \nabla_{full}$)。我们不需要知道全貌，只需要一个"平均指向正确方向"的随机样本即可。这带来了巨大的速度提升和内存节省。
</details>

**3.2** 在 SGD 中，如果 Batch Size 变得非常大（接近全量数据集），SGD 的行为会退化为什么？
<details>
<summary>💡 提示</summary>
退化为 GD (Batch Gradient Descent)。噪声消失，收敛变慢，且容易死锁在鞍点。这也是为什么大 Batch Training 很难训练深度网络的原因。
</details>

### 🔥 进阶题

**3.3** 假设 Loss 函数是 $f(x) = x^2 + \cos(10x)$（有很多小波纹）。证明：随着训练步数 $k$ 增加，如果学习率 $\eta_k \to 0$ (例如 $\eta_k = 1/k$)，SGD 最终会收敛到全局极小值吗？
<details>
<summary>💡 提示</summary>
理论上会。只要满足 Robbins-Monro 条件 ($\sum \eta_k = \infty, \sum \eta_k^2 < \infty$)。前者保证能走得足够远，后者保证后期噪声不会导致剧烈震荡。3DGS 中的 "Cubic Learning Rate Schedule" 正是为了满足这些条件而设计的。
</details>

### 💡 3DGS 关联题

**3.4** (Adam 预告)：在 3DGS 中，有些 Splat 位于边缘（梯度很大），有些位于平坦区域（梯度很小）。如果你只用 SGD，你需要为这两个不同的 Splat 设置不同的学习率吗？Adam 是怎么解决这个问题的？
<details>
<summary>💡 提示</summary>
(1) 如果只用 SGD，必须手动调参给边缘 Splat 设小 LR，否则它会震荡。
(2) Adam 通过维护每个参数的"一阶矩 (均值)"和"二阶矩 (方差)"，自动计算出：梯度大的地方自动缩小步长，梯度小的地方放大步长。这叫 "Per-parameter Adaptive Learning Rates"。
</details>

---

> **Ch03 (Optimization) 完成！** 🔥  
> 
> Part 4 下一站：**Ch04 — 动量与 Adam** —— 如何给梯度加上"惯性"并实现自适应学习率？直接说 "继续"。
