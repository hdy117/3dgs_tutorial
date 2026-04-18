# Ch05 — Shannon熵：信息量的“货币单位”

> **本章目标**：理解"不确定性"如何量化为具体的"比特数"。  
> **前置知识**：Ch01-Ch04（随机变量、分布）。  
> **核心问题**：如果告诉你"太阳明天从东方升起"和"太阳明天从西方升起"，为什么后者的信息量大得多？

---

## 🎯 问题驱动：什么是"信息量"？

### 场景 1：猜词游戏

假设你在和一个外星人聊天。它每次只能回答 "是/否" (Yes/No)。
- **游戏 A**：猜一个正整数 $n \in [1, 6]$（像掷骰子）。
- **游戏 B**：猜硬币正面还是反面。

显然，游戏 A 需要更多次 "是/否" 才能确定答案。**"不确定性"越大，需要的信息量就越多。**

### 关键问题：如何把这个"不确定性"量化为一个数？

Shannon（1948）给出了一个极其优雅的公理化定义：**信息量取决于事件发生的概率**。

---

## 📐 Part 1: 自信息 (Self-Information) — "惊讶程度"的数学表达

### 直觉：越不可能的事件，发生时的"信息量"越大

想象你在做实验：
- **事件 A**（硬币正面）：$P(A)=0.5$。发生了 → 没什么可说的。`Surprise = 0`
- **事件 B**（中彩票头奖）：$P(B) \approx 1/10^7$。发生了 → "真的假的？！！" `Surprise ≈ ∞`

### 公理化定义 (Shannon Axioms)

假设 $I(x)$ 是事件 $x$ 的信息量，它必须满足三条基本性质：

| 性质 | 描述 | 数学约束 |
|------|------|----------|
| **1. 单调性** | 概率越小，信息量越大 | $P(x_1) < P(x_2) \Rightarrow I(x_1) > I(x_2)$ |
| **2. 独立性 (Additivity)** | 两个独立事件同时发生的信息量 = 各自之和 | $I(A \cap B) = I(A) + I(B)$ |
| **3. 归一化** | 确定事件（$P=1$）的信息量为 0 | $I(1) = 0$ |

### 🔥 Part 2: 唯一解 — 为什么是 "对数"？

我们寻找函数 $f(p)$ 满足上述条件。
由性质 3：$f(1) = 0$。
由性质 2：设两个独立事件，概率分别为 $p_1, p_2$。联合概率为 $p_1 p_2$。
$$f(p_1 p_2) = f(p_1) + f(p_2)$$

**这是一个著名的函数方程 (Cauchy functional equation)**！唯一连续解是：
$$\boxed{f(p) = -C \log_b(p)}$$

负号是因为概率 $p \in [0, 1]$，$\log p \leq 0$；而信息量必须 $\geq 0$。常数 $C$ 通常取 1。

### 💡 单位选择
- **底数 $b=2$** → 单位为 **bit**（比特）——计算机标准
- **底数 $b=e$** → 单位为 **nat**（纳特/自然信息单元）— 数学推导方便

我们主要用 **bit**。所以：
$$\boxed{I(x) = -\log_2 P(x)}$$

### 🧪 数值验证 — Boxed Result

| 事件 | $P(x)$ | 信息量 $-\log_2 P(x)$ | 直觉解释 |
|------|--------|-----------------------|----------|
| 硬币正面 | 0.5 | $\boxed{1 \text{ bit}}$ | 恰好需要 1 次 "是/否" 就能确认 |
| 骰子掷出 6 | 1/6 ≈ 0.167 | $\approx \boxed{2.585 \text{ bits}}$ | 平均需要约 2.6 次二分查找才能锁定 |
| 确定事件 | 1.0 | $\boxed{0 \text{ bits}}$ | 毫无信息，你早就知道了 |

---

## 📐 Part 3: 熵 (Entropy) — "平均惊讶程度"

### 定义：自信息的期望值

既然 $I(x)$ 是随机变量（因为 $x$ 是随机的），那么它的平均值就是**熵**：
$$\boxed{H(X) = E[-\log_2 P(X)] = -\sum_{x \in \mathcal{X}} P(x) \log_2 P(x)}$$

**直觉解读**：
- 熵 $H(X)$ 是对一个随机变量"不确定性总量"的度量。
- 如果你有一个编码方案，能用平均长度等于 $H(X)$ bit 的词来唯一标识每个结果 —— 那就是最优压缩！
- **熵 = "最优压缩的平均码长"**（Shannon 第一定理）。

---

## 🔥 Part 4: 最大熵原理 — Boxed Result

### 问题：哪种分布的"不确定性"最大？（即最"诚实"）

假设我们只知道随机变量取值于 $N$ 个可能状态，没有别的先验信息。直觉告诉我们：**均匀分布**是最不偏向任何一方的。让我们验证它是否真的拥有最大熵。

**目标证明**：给定有限域大小 $|\mathcal{X}| = N$，当且仅当 $P(x) = \frac{1}{N}$ 时（均匀分布），熵取最大值 $\log_2 N$。

### 推导过程 (使用 Jensen 不等式)

回顾 Jensen 不等式：对于**凹函数** $\phi(\cdot)$（如 $\log x$，因为 $(\ln x)'' = -1/x^2 < 0$），有：
$$E[\log X] \leq \log E[X]$$

对熵进行推导：
$$H(X) = -\sum_{i=1}^{N} p_i \log_2 p_i = \frac{1}{\ln 2} \left( -\sum_{i=1}^{N} p_i \ln p_i \right)$$

考虑函数 $\ln x$（它是凹函数）：
$$\begin{aligned}
-\sum_{i=1}^{N} p_i \ln p_i &= \sum_{i=1}^{N} p_i \ln \left( \frac{1}{p_i} \right) \\
&= E[\ln(1/P)] \quad (\text{把 } 1/p_i \text{ 看作随机变量的取值}) \\
\end{aligned}$$

应用 Jensen 不等式（注意 $\log$ 是凹函数，方向是 $\leq$）：
$$E[\ln(1/P)] \leq \ln(E[1/P]) = \ln\left(\sum_{i=1}^{N} p_i \cdot \frac{1}{p_i}\right)$$

因为 $p_i / p_i = 1$，且求和有 $N$ 项：
$$E[\ln(1/P)] \leq \ln(N)$$

代回熵公式（注意前面有个负号，不等式方向反转？不，我们直接处理正形式）：
更直接的写法（Gibbs' Inequality）：
$$\begin{aligned}
D_{KL}(P || U) &= \sum p_i \ln(p_i/U_i) \geq 0 \\
\Rightarrow \sum p_i (\ln p_i - \ln(1/N)) &\geq 0 \\
\Rightarrow \sum p_i \ln p_i &\geq \ln(1/N) = -\ln N \\
-\sum p_i \log_2 p_i &\leq \frac{-\ln(1/N)}{\ln 2} = \log_2 N
\end{aligned}$$

等号成立条件：$D_{KL}(P||U)=0 \iff P=U$。

### ✅ 核心结论 (Boxed)

$$\boxed{\text{对于有限域 } |\mathcal{X}|=N, \quad H(X) \leq \log_2 N}$$
**当且仅当 $x$ 均匀分布时，熵最大！** 🏆

---

## 💻 Part 5: PyTorch 验证与实现

```python
import torch
import math

# ============================================================
# 1. 离散 Shannon Entropy 计算 (从 PMF)
# ============================================================
def calculate_entropy(p):
    """手动计算熵：H(X) = - Σ p_i log2(p_i)"""
    # 确保概率不为零（避免 log(0)）
    mask = p > 0
    h = -(p[mask] * torch.log2(p[mask])).sum()
    return h

# --- 案例 A：均匀分布 (骰子) ---
dice_pmf = torch.ones(6) / 6.0
H_dice = calculate_entropy(dice_pmf)
print(f"公平骰子的熵: {H_dice.item():.4f} bits")
print(f"理论最大值 log2(6): {math.log2(6):.4f} bits → Match! ✅")

# --- 案例 B：不公平硬币 (偏向正面 0.9) ---
biased_coin_pmf = torch.tensor([0.1, 0.9]) # P(0), P(1)
H_biased = calculate_entropy(biased_coin_pmf)
print(f"\n偏置硬币 (p=0.9) 的熵: {H_biased.item():.4f} bits")

# --- 案例 C：确定事件 ---
certain_pmf = torch.tensor([0.0, 1.0])
H_certain = calculate_entropy(certain_pmf)
print(f"确定事件的熵: {H_certain.item():.6f} bits → 0 ✅")

# ============================================================
# 2. 最大熵原理验证：搜索最优分布
# ============================================================
print("\n=== 最大熵原理验证 ===")
N = 5  # 5个可能状态
p_uniform = torch.ones(N) / N
H_max = calculate_entropy(p_uniform)

# 随机扰动测试
random_p = torch.rand(N)
random_p = random_p / random_p.sum()  # 归一化
H_random = calculate_entropy(random_p)

print(f"均匀分布 (p=1/{N}) 熵: {H_max.item():.4f} bits")
print(f"随机分布熵:           {H_random.item():.4f} bits")
print(f"H_uniform >= H_random? → {H_max >= H_random} ✅")

# ============================================================
# 3. PyTorch Distributions API (最简方式)
# ============================================================
from torch.distributions import Categorical, Normal

cat = Categorical(torch.tensor([0.5, 0.5])) # 公平硬币
print(f"\nPyTorch内置熵: {cat.entropy().item():.4f}")

normal = Normal(torch.tensor(0.), torch.tensor(1.))
# 连续变量的"微分熵" (不是离散熵！)
diff_entropy = normal.entropy() 
# H_continuous = 0.5 * ln(2*pi*e*sigma^2) -> log2 version...
print(f"标准高斯的微分熵: {diff_entropy.item():.4f} nats")
```

---

## 🗺️ Part 6: 与 3DGS 的衔接点 — 信息论视角

虽然 3DGS 没有直接调用 `calculate_entropy()`，但**整个优化过程是在做"最小化损失"——而损失的底层就是信息论！**

| 概念 | 在 3DGS 中的影子 |
|------|-----------------|
| **交叉熵 (Cross-Entropy)** | L1 / MSE Loss 的"亲兄弟"。如果你把颜色预测看作概率分布，MSE 可以看作是 KL 散度的一阶泰勒近似。 |
| **最大熵原理** | 3DGS 初始化时，如果没有先验几何信息，我们会均匀/随机采样 Splat —— 这是为了保持初始状态的"高不确定性"（探索性）。 |
| **压缩 / 码长** | 3DGS 的核心优势之一是**显式存储**。每个 Gaussian 的 μ, Σ, α, c 都需要编码。优化过程其实是在做：**在保真度 (Loss) 和 参数量 (Entropy of parameters)** 之间找平衡（R-D Tradeoff）。 |

---

## 🎓 本章小结

### 核心公式

$$\boxed{I(x) = -\log_2 P(x) \quad (\text{自信息})}$$

$$\boxed{H(X) = -\sum_{x} P(x)\log_2 P(x) \quad (\text{熵：平均信息量})}$$

$$\boxed{\text{最大熵: } H(X) \leq \log_2 |\mathcal{X}|, \text{当且仅当均匀分布时取等号}}$$

### 关键洞察

> **熵不是"混乱"的模糊概念** —— 它是精确的"压缩极限"。一个随机变量 $X$，无论用什么编码方案，平均每个符号至少要 $\log_2 |\mathcal{X}|$ bit（如果是均匀的）。
> 
> **为什么用负号？** $P(x)$ 越大，$\log P(x)$ 越接近 0。加负号是为了让"高概率=低信息量"符合直觉。
> 
> **3DGS 的 Loss**：L1/MSE 看起来像简单的几何距离，但深入看，它是在衡量预测分布和真实数据分布之间的 KL 散度（下一节）。

---

## 📚 习题

### ✅ 基础题

**5.1** 一个二元随机变量 $X$ 的 PMF 为：$P(X=0) = p, P(X=1) = 1-p$。
- (a) 写出熵的表达式（二进制形式）。
- (b) 计算当 $p=0.5$ 和 $p=0.9$ 时的熵值。

<details>
<summary>💡 提示</summary>
(a) $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ (二元熵函数 $H_b(p)$)
(b) $H(0.5) = -0.5(-1) - 0.5(-1) = \boxed{1.0}$ bit; 
$H(0.9) = -(0.9\log_2 0.9 + 0.1\log_2 0.1) \approx \boxed{0.469}$ bits.
</details>

**5.2** 证明：对于离散分布，熵 $H(X)$ 总是非负的。

<details>
<summary>💡 提示</summary>
因为 $0 \leq p_i \leq 1$，所以 $\log_2 p_i \leq 0$（负数或零）。
前面加了负号：$-\sum p_i (\text{负数}) = \sum p_i |\log_2 p_i| \geq 0$。 ✅
</details>

### 🔥 进阶题

**5.3** 抛一枚硬币直到第一次出现正面为止（几何分布 $P(X=k) = (1-p)^{k-1}p, k=1,2,...$）。
- (a) 如果硬币公平 ($p=0.5$)，这个过程的熵是多少？
- (b) 直观上，"等一个正面"比"掷一次骰子"的信息量多还是少？

<details>
<summary>💡 提示</summary>
(a) $X \sim Geom(0.5)$。其分布为 P(X=1)=0.5, P(X=2)=0.25... 
熵公式推导较复杂，但对于 p=0.5，H = $\frac{H_b(p)}{p} = 2$ bits!
(b) 骰子是 $\approx 2.58$ bits。所以掷一次骰子的不确定性比"等一个正面"更高。
</details>

### 💡 3DGS 关联题

**5.4** 假设你用 1000 个 Splat 渲染一张图片，其中只有 10% 的像素有实质性的颜色信息（其余是背景/透明）。
从信息论的角度看，这种"稀疏性"对数据压缩有什么意义？

---

> **Ch05 完成！** 🔥  
> 
> Part 2 下一站：**交叉熵 (Cross-Entropy)** —— 为什么它是神经网络和 3DGS 损失函数的灵魂？  
> 准备好继续 `Act` 了吗？直接说 "继续"。
