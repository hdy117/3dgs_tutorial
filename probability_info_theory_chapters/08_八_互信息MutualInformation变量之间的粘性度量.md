# Ch08 — 互信息 (Mutual Information)：变量之间到底有多"粘"？

> **本章目标**：理解如何量化两个随机变量的"依赖程度"。  
> **前置知识**：Ch05-Shannon熵, Ch06-交叉熵, Ch07-KL散度。  
> **核心问题**：如果我知道相机视角 $V$，我对场景几何 $G$ 的不确定性减少了多少？

---

## 🎯 问题驱动：为什么需要"互信息"？

### 场景 1：多视角重建中的信息增益

你正在用 3DGS 重建一个物体。
- **初始状态**：只有一个相机视角 → 背后全是盲区。你对深度的不确定性极高。
- **加入第二个视角**：盲区被填补了。

**关键问题**：第二个视角到底给你带来了多少**新信息**？怎么量化这个"信息增益"？

### 答案：互信息 (Mutual Information, MI) — "知道 A 后，B 的不确定性减少了多少？"

---

## 📐 Part 1: 定义与直觉

### 💡 生活例子：天气预报和带伞决策

想象你每天早上出门前做决策：**今天要不要带伞？**

**情况 A — 没有天气预报（不知道明天天气）**：
- P(下雨) = 30%，P(不下雨) = 70%
- 你对"会不会下雨"的不确定性 = $H(\text{天气}) = -(0.3\log_2 0.3 + 0.7\log_2 0.7) \approx \boxed{0.88 \text{ bits}}$
- 你需要猜测：带伞有 30% 概率白带，不带伞有 30% 概率淋雨。

**情况 B — 看了天气预报（已知预报说"明天会下雨"）**：
- $P(\text{下雨} | \text{预报:下雨}) = 80\%$ → 你对"会不会下雨"的不确定性降到：
- $H(\text{天气}|\text{预报:下雨}) = -(0.8\log_2 0.8 + 0.2\log_2 0.2) \approx \boxed{0.72 \text{ bits}}$

**信息增益（互信息）**：
$$I(\text{天气}; \text{预报}) = H(\text{天气}) - H(\text{天气}|\text{预报})$$
平均而言，看了天气预报后你对"会不会下雨"的不确定性减少了约 $\boxed{0.15 \text{ bits}}$。

> 🎯 **直觉理解**：互信息 = "知道 A 之后，B 的不确定性减少了多少"。在这个例子中，天气预报虽然不完美（有时预报下雨但实际没下），但它确实帮你减少了对天气的困惑 — 从 0.88 bits 降到 0.72 bits。这 0.15 bits 就是**你从天气预报中获得的信息价值**。

### 💡 第二个例子：多视角3D重建中的信息增益

$$\boxed{I(X; Y) = H(Y) - H(Y|X)}$$

**拆解**：
- $H(Y)$: B 的原始不确定性。
- $H(Y|X)$: 在已知 X 的条件下，B 剩余的不确定性。
- **互信息 = "被消除的不确定性量"**！

### 另一种等价定义 (对称性之美)

$$\boxed{I(X; Y) = H(X) + H(Y) - H(X, Y)}$$

其中 $H(X,Y)$ 是联合熵。
**直觉**：两个变量的总不确定性 ($X+Y$) 减去它们分开的不确定性之和？不，是反过来——如果 X 和 Y 完全独立，$H(X,Y)=H(X)+H(Y)$，互信息为 0。如果有依赖，联合熵变小了（因为知道 X 就猜到 Y），差值就是**共享的信息量**。

### 终极定义 (用 KL 散度表达！)

$$\boxed{I(X; Y) = D_{KL}\left( P(x,y) \,||\, P(x)P(y) \right)}$$

**推导过程**：
互信息衡量的是"真实联合分布 $P(x,y)$"与"假设独立后的分布 $P(x)P(y)$"有多不同。
如果 X 和 Y 独立，$P(x,y) = P(x)P(y)$，KL 散度为 0 → 无互信息！

---

## 🔥 Part 2: 从第一性原理推导 — MI 为什么 ≥ 0？

### 目标：证明 $I(X;Y) \geq 0$

利用 MI 的 KL 定义：
$$I(X; Y) = D_{KL}(P(x,y) || P(x)P(y))$$

因为 **KL 散度永远非负** (Ch07 已证)，所以直接得出 $I(X;Y) \geq 0$。 ∎

### 💡 关键洞察：互信息是"依赖性的度量器"
- $I = 0$: X, Y 独立（完全陌生）。
- $I > 0$: X, Y 有关联（越粘，MI 越大）。
- **注意**：MI 衡量的是统计依赖性，不一定是线性相关！它能捕捉任何复杂的非线性关系。

---

## 🧪 Part 3: Boxed Result — MI 的性质总结

| 性质 | 描述 |
|------|------|
| **对称性** | $I(X; Y) = I(Y; X)$ （知道 A 帮多少 B，等于知道 B 帮多少 A） |
| **非负性** | $I(X; Y) \geq 0$ |
| **极大值** | $I(X;Y) \leq \min(H(X), H(Y))$ （不可能比变量自身的不确定性还大） |

---

## 💻 Part 4: PyTorch 验证与实战代码 (基于采样估计)

```python
import torch
from scipy.stats import entropy

# ============================================================
# 1. 离散互信息的直接计算
# ============================================================
print("=== 离散 Mutual Information ===")

# 联合分布 P(X,Y) - 假设 X, Y ∈ {0, 1}
joint_p = torch.tensor([[0.4, 0.1],   # P(0,0)=0.4, P(0,1)=0.1
                        [0.1, 0.4]])  # P(1,0)=0.1, P(1,1)=0.4

# 边际分布 (Marginal)
p_x = joint_p.sum(dim=1) # [0.5, 0.5]
p_y = joint_p.sum(dim=0) # [0.5, 0.5]

# MI = Σ p(x,y) log( p(x,y) / (p(x)p(y)) )
mi_manual = torch.sum(joint_p * torch.log2(joint_p / (p_x.unsqueeze(1) * p_y) + 1e-7)).item()
print(f"MI(X,Y) via formula: {mi_manual:.4f} bits")

# ============================================================
# 2. 互信息 vs 相关系数：捕捉非线性依赖
# ============================================================
print("\n=== MI vs Correlation (Linear Dependency) ===")

torch.manual_seed(42)
N = 10000

# --- 案例 A：强线性相关 Y = 2X + noise ---
x_lin = torch.randn(N)
y_lin = 2 * x_lin + torch.randn(N) * 0.1
corr_a = torch.corrcoef(torch.stack([x_lin, y_lin]))[0, 1].item()
print(f"案例A (线性): 相关系数 = {corr_a:.4f}")

# --- 案例 B：强非线性相关 Y = X^2 + noise ---
x_nonlin = torch.randn(N)
y_nonlin = x_nonlin**2 + torch.randn(N) * 1.0 # 二次方关系
corr_b = torch.corrcoef(torch.stack([x_nonlin, y_nonlin]))[0, 1].item()
print(f"案例B (非线性 Y=X²): 相关系数 = {corr_b:.4f} （接近 0！）")

# 💡 互信息对 B 会非常敏感！而相关系数会"瞎掉"。
# 因为 MI = D_KL(联合 || 独立)，它检测任何偏离独立的信号。

# ============================================================
# 3. PyTorch Distributions (用 KL 计算 MI)
# ============================================================
from torch.distributions import kl_divergence, Normal, Independent

print("\n=== MI via KL Divergence ===")
# 构建联合分布和独立近似
# X ~ N(0,1), Y|X ~ N(X, 1) (依赖关系)
mu_x = torch.tensor([0.0])
cov_x = torch.tensor([[1.0]])

# 条件均值: mu_y|x = x
# 这里我们用简单的采样近似来演示 MI 的蒙特卡洛估计
x_samples = Normal(0, 1).sample((N,))
y_samples = x_samples + Normal(0, 1).sample((N,))

def estimate_mi_mc(x, y, bins=20):
    """基于直方图 (Bin) 的 MI 蒙特卡洛估计"""
    # 离散化到 bins
    x_bins = torch.bucketize(x, torch.linspace(-3, 3, bins+1)).float()
    y_bins = torch.bucketize(y, torch.linspace(-3, 3, bins+1)).float()
    
    # 联合直方图
    joint_hist = torch.zeros((bins, bins))
    for xi, yi in zip(x_bins, y_bins):
        joint_hist[int(xi), int(yi)] += 1
        
    total = joint_hist.sum()
    p_xy = joint_hist / total
    p_x = p_xy.sum(dim=1)
    p_y = p_xy.sum(dim=0)
    
    # MI = Σ p_ij log(p_ij / (p_i * p_j))
    mi = torch.sum(p_xy * torch.log2(p_xy / (p_x.unsqueeze(1) * p_y + 1e-7)))
    return mi.item()

mi_mc = estimate_mi_mc(x_samples, y_samples)
print(f"MI(X,Y) via Monte Carlo estimation: {mi_mc:.4f} bits")

# ============================================================
# 4. PyTorch 内置的 KL (用于计算 MI 的变体：InfoNCE 等)
# ============================================================
print("\n=== InfoNCE 风格 (对比学习中的互信息下界) ===")
# 在 Contrastive Learning 中，我们最大化样本与正样本对的 MI。
# MI_lower_bound = log(N) - log(Σ exp(sim(x_i, x_j)/tau))

# 这是一个高级话题：用 KL 散度来逼近难算的互信息。
```

---

## 🗺️ Part 5: 与 3DGS 的衔接点 — MI 在渲染中的角色

虽然标准 3DGS 不调 `calculate_mutual_information()`，但 **MI 是理解现代 Neural Rendering 的核心钥匙**：

| 概念 | 3DGS / 渲染中的应用 |
|------|---------------------|
| **互信息最大化 (MIM)** | 多视角重建中，新相机视角带来的"信息增益"。选择最优相机路径时，会优先选择那些能最大化 $I(\text{Scene} | \text{NewView})$ 的角度。 |
| **信息瓶颈 (Info Bottleneck)** | 压缩神经表示。在 3DGS 中，每个 Gaussian 的参数量可以看作一个"瓶颈"——优化过程是在保真度 ($I(\text{Render}, \text{GT})$) 和 复杂度 ($I(\text{Gaussians}, \text{Scene}))$ 之间找平衡。 |
| **去相关性 (Deregularization)** | 如果两个 Splat 的编码高度相关（互信息大），说明它们冗余了！现代压缩算法会惩罚高 MI，只保留独立特征。 |

---

## 🎓 Part 2 & 3 小结：概率 → 信息论 → 3DGS 全景图完成！🏆

| 模块 | 核心主题 | Boxed 公式 |
|------|----------|-----------|
| **Part 1: 概率基础** | Ch01-04: PMF/PDF, E[X], Var(X), Bayes, Gaussian/Beta/Dirichlet | $P(A|B) = P(B|A)P(A)/P(B)$ |
| **Ch05: Shannon熵** | 不确定性量化 (比特单位) | $H(X) = -\sum p \log p$ |
| **Ch06: Cross-Entropy** | 两个分布的"编码成本" (Loss灵魂) | $H(P,Q) = H(P) + D_{KL}(P||Q)$ |
| **Ch07: KL Divergence** | 分布间差异度量 (不对称, ≥0) | $D_{KL}(P||Q) \geq 0$ |
| **Ch08: Mutual Info** | 变量间的依赖程度 | $I(X;Y) = H(Y) - H(Y|X)$ |

### 关键洞察：信息论的"大道至简"

> **所有信息论的核心，都归结为两件事**：
> 1. **熵 (Entropy)**: "这东西有多不可预测？"
> 2. **KL 散度**: "我的模型和现实差多远？"
> 
> **3DGS 的优化过程 = 最小化 KL 散度 (即交叉熵 Loss)！**
> 
> - Loss ↓ → 预测分布 Q 逼近真实分布 P。
> - $\Sigma$ ↑ → Splat 变得更"宽"，覆盖更多不确定性（但可能模糊）。
> - $\mu \to x_{true}$ → 位置收敛到数据重心。

---

## 📚 习题 (Part 2 综合)

### ✅ 基础题

**8.1** 证明互信息的对称性：$I(X; Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$。
<details>
<summary>💡 提示</summary>
利用 $H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$ (链式法则)。直接代入即可证相等。
</details>

### 🔥 进阶题

**8.2** 如果 X, Y 是联合高斯分布 $\mathcal{N}(\mu, \Sigma)$，且 $\rho$ 是相关系数。证明 $I(X;Y) = -\frac{1}{2}\log(1-\rho^2)$ bits。
<details>
<summary>💡 提示</summary>
使用高斯 KL 公式或联合熵公式 $H(X,Y) = \ln((2\pi e)\sqrt{|\Sigma|})$。因为 $\Sigma = [[1, \rho], [\rho, 1]]$，所以 $|\Sigma|=1-\rho^2$。代入 MI 定义即可得证。
</details>

### 💡 3DGS 关联题

**8.3** (Part 3 预告思考)：假设你在训练一个 "压缩版" 的 3DGS，只有 K=1000 个 Splat。你希望这 1000 个高斯能覆盖场景最多的信息。
- (a) 你应该最大化哪个量？$H(\text{Scene})$ 还是 $I(\text{Gaussians}; \text{Scene})$？
- (b) 如果两个 Splat 几乎重叠（位置相同，颜色相同），它们的互信息接近多少？这对压缩有什么启发？

---

> **🎉 Part 2: 信息论核心 — 全部完成！**  
> 
> **Part 3: 与 3DGS 的衔接** (Step 9-11) 准备就绪：
> - Ch09: Gaussian 分布 vs 高斯 Splatting（为什么用高斯？）
> - Ch10: MLE 在参数拟合中的应用
> - Ch11: 信息论视角下的渲染损失设计
> 
> **下一步指示：** "继续" 写 Part 3，或者先 review 当前进度？🔥
