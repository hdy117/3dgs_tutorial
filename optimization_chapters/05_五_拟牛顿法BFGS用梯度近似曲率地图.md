# Ch05 — 拟牛顿法 (BFGS)：用梯度近似"曲率地图"

> **本章目标**：理解二阶优化方法的威力，以及为什么 BFGS 是牛顿法的最佳替身。  
> **前置知识**：Ch03 (SGD), Ch04 (Adam)。  
> **核心问题**：如果梯度告诉我们"往哪走"，那海森矩阵（曲率）告诉我们什么？如何不用 $O(d^3)$ 的代价来利用曲率信息？

---

## 🎯 问题驱动：牛顿法的"昂贵"代价

### 场景 1：在极度病态的 Loss 曲面中挣扎

```python
# 3DGS 的参数空间可能有数百万个维度 (d = 10^6)。
# 牛顿法 (Newton's Method) 要求我们计算 Hessian Matrix 并求逆。
# H 是一个 d×d 的矩阵，求逆需要 O(d³) 时间！

H = compute_hessian()      # ❌ 百万维度下完全算不动！
inverse_H = torch.linalg.inv(H) # ❌ 内存爆炸！
```

**关键问题**：我们想要牛顿法的快速收敛性（利用曲率信息），但算不起海森矩阵。怎么办？

### 答案：**拟牛顿法 (Quasi-Newton Methods)** —— "用历史梯度拼凑出的曲率地图"

---

## 📐 Part 1: 牛顿法 vs 梯度下降 — "看脸"与"盲走"

回顾泰勒展开的二阶近似，牛顿法的更新公式是：
$$\boxed{\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)}$$

### 💡 核心对比 (Boxed Result)

| 优化器 | 更新方向 $\mathbf{d}$ | 物理直觉 |
|--------|----------------------|----------|
| **梯度下降** | $-\nabla f$ | "看脚下"：只利用一阶信息（坡度）。容易震荡。 |
| **牛顿法** | $-\mathbf{H}^{-1} \nabla f$ | **"看脸"**：利用二阶信息（曲率）。直接跳到谷底！ |

如果函数是二次型 (Quadratic)，牛顿法只需 **1 步**就能到达最优解！这就是所谓的"二次收敛速度"。

---

## 🔥 Part 2: BFGS —— 为什么它是拟牛顿法的王者？

### Boxed Result：核心思想 (The Secant Equation)

我们不直接计算海森矩阵 $\mathbf{H}$，而是维护一个近似逆海森矩阵 $\mathbf{B}_k$。
它必须满足**割线方程 (Secant Equation)**：
$$\boxed{\mathbf{B}_{k+1} \cdot \nabla L_{t} = \Delta \mathbf{x}_{t}}$$

其中 $\Delta \mathbf{x}$ 是两次迭代间参数变化的向量，$\nabla L_t$ 是梯度变化向量。
**直觉**：如果我知道在某个方向上梯度变了多少 ($\Delta g$)，我就反推出那个方向的曲率应该是多少！

### 💡 BFGS 更新规则 (秩-2 更新)

$$\boxed{\mathbf{B}_{k+1} = \mathbf{B}_k + \text{Rank-2 Update}}$$
每次迭代只增加少量的信息（基于最新的两个梯度向量），而不需要重新计算整个矩阵。这使得它的复杂度仅为 $O(d^2)$，比牛顿法的 $O(d^3)$ 快得多。

---

## 💻 Part 4: PyTorch 验证 — BFGS vs SGD 在病态曲面上的表现

```python
import torch
from torch.optim import LBFGS, SGD

# ============================================================
# 1. 构造一个极度弯曲的 "香蕉函数" (Rosenbrock)
# f(x,y) = (a-x)^2 + b(y-x^2)^2，这是一个著名的非凸、病态问题。
# ============================================================
print("=== BFGS vs SGD: Rosenbrock 测试 ===")

def rosenbrock(x, y): return (1-x)**2 + 100*(y - x**2)**2

x = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
y = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)

# --- BFGS (拟牛顿法：自带二阶近似) ---
optimizer_bfgs = LBFGS([x, y], lr=1.0, history_size=10) # 记忆最近 10 次梯度

def closure():
    optimizer_bfgs.zero_grad()
    loss = rosenbrock(x, y)
    loss.backward()
    return loss

for i in range(20):
    loss = optimizer_bfgs.step(closure)
    
print(f"BFGS 最终位置: ({x.item():.4f}, {y.item():.4f}) -> Loss={rosenbrock(x,y):.6f}")

# --- SGD (作为对比) ---
x_s, y_s = torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.5], dtype=torch.float32)
opt_sgd = SGD([x_s, y_s], lr=0.01)

for i in range(50): # SGD 需要更多步数
    opt_sgd.zero_grad()
    loss = rosenbrock(x_s, y_s)
    loss.backward()
    opt_sgd.step()

print(f"SGD+Momentum 最终位置: ({x_s.item():.4f}, {y_s.item():.4f}) -> Loss={rosenbrock(x_s,y_s):.6f}")

# 💡 BFGS 通常步数极少（10-20步），因为它利用曲率信息跳过了大量平坦区域。
```

---

## 🗺️ Part 5: 与 3DGS 的衔接点 — 预条件优化 (Preconditioning)

### 核心洞察：Adam 就是大规模分布式 BFGS！

虽然 3DGS 不用完整的 LBFGS（因为维度太高，无法存储历史梯度），但 **Adam 的核心思想正是受拟牛顿法启发而来**。

| 概念 | 拟牛顿法 (BFGS) | Adam / RMSProp |
|------|----------------|---------------|
| **曲率近似** | 维护全局矩阵 $\mathbf{B} \approx \mathbf{H}^{-1}$ | 维护对角线向量 $s_t \approx \text{diag}(\mathbf{H}^{-1})$ |
| **计算代价** | $O(d^2)$ — 中等 | $O(d)$ — 极快！ |
| **适用场景** | 小参数空间（< 10,000） | 大参数空间（百万级，如 3DGS/LLM） |

### 💡 为什么 3DGS 不用二阶优化？
除了计算量太大外，还有一个致命原因：**海森矩阵在极小值附近是正定的，但在鞍点处有负特征值。** BFGS 无法保证在处理非凸函数时始终近似出一个合法的逆矩阵。而 Adam 通过对分母加 $\epsilon$ 保证了数值稳定性。

---

## 🎓 本章小结

### 核心公式 (Boxed)

$$\boxed{\text{Newton: } \Delta \mathbf{x} = -[\nabla^2 f]^{-1} \nabla f \quad (\text{利用真实曲率})}$$

$$\boxed{\text{Secant Eq: } \mathbf{B}_{k+1} \cdot \Delta \nabla L_k = \Delta \mathbf{x}_k \quad (\text{用梯度变化反推曲率})}$$

### 关键洞察

> **牛顿法只需一步**：对于二次函数，牛顿法利用二阶泰勒展开直接解出根。这是它比 SGD 快得多的根本原因。
> 
> **拟牛顿法的优雅之处**：既然算不起海森矩阵，就通过观察"梯度随参数的变化率"来估算曲率。这就是 BFGS 的核心智慧。
> 
> **3DGS 的工程取舍**：由于维度太高，我们退而求其次使用 Adam（对角近似）。但在超大规模优化器中，二阶信息（如 Hessian-free 训练）依然是前沿研究方向。

---

## 📚 习题

### ✅ 基础题

**5.1** 为什么牛顿法在二次型函数 $f(x)=x^2$ 上只需一步就能到达最优解？
<details>
<summary>💡 提示</summary>
因为二阶泰勒展开对二次函数是精确的（没有高阶项）。方程 $\nabla f + \mathbf{H}\Delta x = 0$ 就是原方程本身，解出的 $\Delta x$ 直接指向最低点。
</details>

**5.2** BFGS 中的 "Rank-2 Update" 是什么意思？为什么它比全量更新快？
<details>
<summary>💡 提示</summary> "秩-2"意味着每次只通过两个向量的外积来微调矩阵 $\mathbf{B}$，而不需要重新计算整个 $d \times d$ 矩阵。这使得每步复杂度从 $O(d^3)$ 降到了 $O(d^2)$。
</details>

### 🔥 进阶题

**5.3** (非凸优化)：如果 Loss 曲面存在负曲率（鞍点），牛顿法会遇到什么问题？拟牛顿法如何处理这种情况？
<details>
<summary>💡 提示</summary> 牛顿步长 $\Delta x = -\nabla f / f''$。如果 $f'' < 0$，步长的方向会变成"上坡"，导致优化发散。拟牛顿法（如 BFGS）要求近似矩阵始终正定，因此在遇到负曲率时会通过修正策略强行维持正定性，但这可能导致收敛速度下降。
</details>

### 💡 3DGS 关联题

**5.4** (预条件梯度下降)：如果你发现 3DGS 中某些 Splat 的 $\mu$ 参数更新极慢，你可以手动给它的学习率乘上一个系数 $\lambda > 1$。这在数学上等价于近似了海森矩阵中的哪一项？
<details>
<summary>💡 提示</summary> 等价于近似了逆海森矩阵的对角线元素 $H_{ii}^{-1}$。如果某个参数对应的曲率（二阶导）很小，它的逆就很大——因此应该给它更大的步长。这正是 Adam/RMSProp 在做的事情。
</details>

---

> **Ch05 (Optimization) 完成！** 🔥  
> 
> Part 4 下一站：**Ch06 — 二阶优化与 Hessian 详解** —— 深入分析曲率如何导致梯度消失/爆炸，以及 Hessian-Free 优化的前沿。直接说 "继续"。
