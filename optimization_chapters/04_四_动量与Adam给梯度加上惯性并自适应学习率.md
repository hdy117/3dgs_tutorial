# Ch04 — 动量与 Adam：给梯度加上"惯性"并自适应学习率

> **本章目标**：掌握深度学习中最强大的优化器——Adam，以及它背后的直觉。  
> **前置知识**：Ch01 (凸性), Ch02 (梯度), Ch03 (SGD)。  
> **核心问题**：如果你在下山时感觉膝盖疼（震荡），或者遇到一片平缓的草地走不动了（鞍点），该怎么办？

---

## 🎯 问题驱动：SGD 的两个致命弱点

### 场景 1：在病态曲面上挣扎

```python
# 3DGS 的 Loss 曲面通常非常 "病态" (Ill-conditioned)：
# 方向 A (Splat 位置 μ): 极其陡峭，梯度很大。
# 方向 B (Splat 颜色 c): 极其平缓，梯度很小。

optimizer = SGD(params, lr=0.01) # 统一学习率！
```

**关键问题**：如果你设一个统一的 LR=0.01，在方向 A 上会震荡爆炸（因为太陡），在方向 B 上又会慢如蜗牛（因为太平）。怎么才能让优化器"聪明地"适应地形？

### 答案：**Momentum (动量)** + **Adam (自适应学习率)** —— "下坡加速，上坡减速"

---

## 📐 Part 1: Momentum — 惯性原理

### Boxed Result：带动量的 SGD 更新公式

$$\boxed{v_t = \beta v_{t-1} + \eta \nabla L(\mathbf{x}_k)}$$
$$\boxed{\mathbf{x}_{k+1} = \mathbf{x}_k - v_t}$$

其中 $\beta$ (通常取 0.9) 是**动量系数**。

### 💡 物理直觉：下坡时加速，上坡时减速

想象你推着一个装满水的桶下山（梯度）：
*   **下坡阶段**：重力在帮你加速。梯度的方向一直在变缓，但惯性 $v_t$ 会让你冲得更快。
*   **震荡阶段**：如果你在山谷的一侧横向移动，重力的分量会把你往回拉，抵消你的横向速度。**动量抑制了横向的震荡！**

### 🔥 数学本质：梯度的一阶自回归 (Exponential Moving Average)

$$v_t = \eta \nabla L_k + \beta \eta \nabla L_{k-1} + \beta^2 \eta \nabla L_{k-2} + ...$$
动量本质上是**过去所有梯度的指数加权平均**。它让优化器看的是"趋势"，而不是某一次随机的噪声。

---

## 📐 Part 2: RMSProp —— "自适应学习率"的先驱

### Boxed Result：RMSProp 核心思想

$$\boxed{\mathbf{s}_t = \beta \mathbf{s}_{t-1} + (1-\beta) (\nabla L_t)^2}$$
$$\boxed{\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \nabla L_t}$$

其中 $\mathbf{s}_t$ 是梯度的**二阶矩（平方的移动平均）**。

### 💡 核心洞察：梯度大的维度自动缩小步长！

在分母上除以 $\sqrt{\mathbf{s}_t}$，意味着：
*   **陡峭方向 (梯度大)**: $\mathbf{s}_t$ 很大 → 整个系数很小 → **步长缩小**。
*   **平缓方向 (梯度小)**: $\mathbf{s}_t$ 很小 → 整个系数接近 1 → **保持大步长**。

这就是自适应学习率的灵魂！它完美解决了 "病态曲面" 的问题。

---

## 🔥 Part 3: Adam —— 动量与 RMSProp 的终极合体

### Boxed Result：Adam (Adaptive Moment Estimation) 更新公式 ⚔️

$$\boxed{\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla L_t \quad (\text{一阶矩/均值})}$$
$$\boxed{\mathbf{s}_t = \beta_2 \mathbf{s}_{t-1} + (1-\beta_2)(\nabla L_t)^2 \quad (\text{二阶矩/方差})}$$

**Bias Correction (偏差修正，非常重要！)**：
由于初始化时 $\mathbf{m}_0=\mathbf{s}_0=0$，初期估计会偏向 0。Adam 进行了补偿：
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1-\beta_2^t}$$

**最终更新**：
$$\boxed{\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}}$$

### 💡 核心洞察：Adam = "带着惯性的自适应导航仪"
*   $\hat{\mathbf{m}}_t$ (分子)：负责冲过鞍点，抑制震荡。
*   $\sqrt{\hat{\mathbf{s}}_t}$ (分母)：负责在陡峭处减速，在平缓处加速。

---

## 💻 Part 4: PyTorch 验证 — SGD vs Momentum vs Adam

```python
import torch
from torch.optim import SGD, Adam, RMSprop

# ============================================================
# 1. 构造一个"病态峡谷"函数 (Rosenbrock-like)
# f(x,y) = (a - x)^2 + b * (y - x^2)^2
# 这是一个著名的极难优化的函数，呈弯曲的香蕉状。
# ============================================================
print("=== 优化器对比：病态曲面 Rosenbrock ===")

def rosenbrock(x, y): return (1-x)**2 + 100*(y - x**2)**2

x = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
y = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)

# --- Adam (默认配置: β1=0.9, β2=0.999) ---
optimizer = Adam([x, y], lr=0.05) # 给一个相对较大的 LR，Adam 很稳
path_adam = [(x.item(), y.item())]

for i in range(30):
    optimizer.zero_grad()
    loss = rosenbrock(x, y)
    loss.backward()
    optimizer.step()
    
    # 记录路径 (每 5 步记一次)
    if i % 5 == 0: path_adam.append((x.item(), y.item()))

print(f"Adam 最终位置: ({x.item():.4f}, {y.item():.4f}) -> Loss={rosenbrock(x,y):.6f}")

# --- SGD (带 Momentum) ---
x_s, y_s = torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.5], dtype=torch.float32)
opt_sgd = torch.optim.SGD([x_s, y_s], lr=0.01, momentum=0.9)

for i in range(30):
    opt_sgd.zero_grad()
    loss = rosenbrock(x_s, y_s)
    loss.backward()
    opt_sgd.step()

print(f"SGD+Momentum 最终位置: ({x_s.item():.4f}, {y_s.item():.4f}) -> Loss={rosenbrock(x_s,y_s):.6f}")

# ============================================================
# 2. PyTorch Adam 内部状态揭秘 (m 和 s)
# ============================================================
print("\n=== Adam 内部变量变化 ===")
x_test = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
opt_test = Adam([x_test], lr=0.5)

def f_t(x): return x**4 - 2*x # 非凸函数，有局部极小值
for i in range(3):
    opt_test.zero_grad()
    loss = f_t(x_test).float()
    loss.backward()
    
    # 打印优化器内部状态 (一阶矩 m, 二阶矩 s)
    print(f"Step {i}: grad={x_test.grad.item():.4f} | m={opt_test.state[x_test]['exp_avg'].item():.4f} | s={opt_test.state[x_test]['exp_avg_sq'].item():.6f}")
    
    opt_test.step()

# 💡 注意：随着训练进行，s (梯度平方均值) 会稳定下来，而 m 会跟随梯度的趋势。
```

---

## 🗺️ Part 5: 与 3DGS 的衔接点 — Adam 为什么是标配？

### 核心原因：参数尺度的极度不平衡

在 3DGS 中，你需要同时优化以下几类差异巨大的参数：
1. **位置 $\mu$** (坐标值)：通常很小（如 0.5, -1.2），梯度中等。
2. **缩放 $s$** (Splat 大小)：控制 Splat 扩散程度。
3. **旋转 $R$** (四元数/欧拉角)：单位圆上的值，对 Loss 极其敏感（微小的旋转可能导致遮挡关系剧变）。

如果使用 SGD：你必须给 $\mu$ 和 $R$ 分别设置不同的学习率。这几乎是不可能的任务（需要大量试错调参）。

### ✅ Boxed Result：3DGS Adam 的默认配置

```python
# 3DGS (3D Gaussian Splatting) 官方论文中的优化器设置
optimizer = Adam([params], lr=1.6e-4, eps=1e-15) 
# eps=1e-15: 为了防止梯度极小时分母为零。
```

**为什么不用 SGD？** 
因为 3DGS 的 Loss 曲面是极度病态和非凸的。Adam 提供的"自适应学习率"能让模型在训练初期快速收敛（探索），并在后期自动减速稳定下来（利用）。

---

## 🎓 本章小结

### 核心公式 (Boxed)

$$\boxed{\text{Momentum: } v_t = \beta v_{k-1} + \eta \nabla L, \quad x_{new} = x - v_t}$$

$$\boxed{\text{Adam: } x_{new} = x - \frac{\eta}{\sqrt{s}+\epsilon}\frac{m}{\hat{m}} \quad (\text{一阶矩} / \text{二阶矩})}$$

### 关键洞察

> **动量 (Momentum) 是物理的延伸** —— 它利用惯性冲过平坦区域（鞍点），并抑制震荡。
> 
> **RMSProp/Adam 是统计学的胜利** —— 它们通过计算梯度的"波动程度"，自动为每个参数调整步长：陡峭处减速，平缓处加速。
> 
> **3DGS 几乎不用 SGD** —— 因为 Splat 参数的尺度差异太大。Adam 让工程师可以专注于调参（比如调 $\Sigma$），而不必担心梯度爆炸或消失。

---

## 📚 习题

### ✅ 基础题

**4.1** 为什么 Momentum 能减少 SGD 的震荡？请用物理类比解释。
<details>
<summary>💡 提示</summary>
想象球在碗底滚动。如果没有动量，球会在两侧来回反弹（震荡）。加上动量后，横向的速度会被重力分量抵消并累积为向下的速度，球最终会停在中心。
</details>

**4.2** 在 Adam 的公式中，为什么分母是 $\sqrt{\hat{s}_t}$ 而不是 $\hat{s}_t$？
<details>
<summary>💡 提示</summary>
$\hat{s}_t$ 是梯度的二阶矩（方差/平方）。为了与梯度 $\hat{m}_t$ (一阶矩/均值) 的量纲保持一致，我们需要对分母开方。这样整个系数就是一个纯粹的"缩放因子"。
</details>

### 🔥 进阶题

**4.3** (Bias Correction)：为什么 Adam 在训练初期需要偏差修正 (Bias Correction)？如果不修正会发生什么？
<details>
<summary>💡 提示</summary>
因为 $m_0=s_0=0$，前几步的 $\mathbf{m}_t$ 和 $\mathbf{s}_t$ 会严重偏向于零。不修正会导致初始步长极其微小（相当于学习率被除以了一个接近 0 的数）。修正项 $(1-\beta^t)$ 确保了随着 $t$ 增大，权重逐渐恢复到 1。
</details>

### 💡 3DGS 关联题

**4.4** (高级思考)：虽然 Adam 很强大，但有些研究指出在训练后期使用 SGD + Momentum 能达到比 Adam 更好的泛化效果（更低的重建 Loss）。你如何从"随机噪声"的角度解释这一点？
<details>
<summary>💡 提示</summary>
Adam 通过除以 $\sqrt{s}$ "平滑"了梯度的波动，导致它几乎没有随机噪声。这使得它更容易掉进狭窄的局部极小值（Sharp Minima）。而 SGD (或 Adam 后期换 SGD) 带来的噪声能帮助模型跳出这些陷阱，找到更平坦、泛化能力更强的极小值 (Flat Minima)。
</details>

---

> **Ch04 (Optimization) 完成！** 🔥  
> 
> Part 4 下一站：**Ch05 — 拟牛顿法 (BFGS)** —— "如果我能看到曲率，梯度下降会变成什么样？"（二阶优化入门）。直接说 "继续"。
