# Ch09 - 积分：Newton、Leibniz 与 Volume Rendering Equation (VRE)

> **本章目标**：理解为什么 3DGS 的渲染公式本质上是一个“无穷小量的求和”。
> **核心洞察**：**微积分基本定理 —— 积分是微分的逆运算**。这解释了为什么我们可以对连续的体积光场进行离散的、可微的训练。

---

## 🔥 Part 0: Kepler 的酒桶与“无穷小”的求和

### 积分是怎么诞生的？

1615年，Johannes Kepler 正在计算酒桶的容积。传统的几何公式（圆柱、圆锥）对不规则形状都失效了。
他的思路非常直观：
> **"把酒桶切成无数个极薄的圆片（圆盘），算出每一片的体积，然后全部加起来。"**

这就是**黎曼和 (Riemann Sum)** 的物理起源：
$$V \approx \sum_{i=1}^N A(x_i) \cdot \Delta x$$
其中 $A(x_i)$ 是第 i 片圆片的面积，$\Delta x$ 是其厚度。当 $\Delta x \to 0$ 时，近似变成了精确的**积分**：
$$V = \int_{a}^{b} A(x) dx$$

---

### Newton & Leibniz 的贡献：连接“加减”与“乘除”

Newton 和 Leibniz 发现了一个惊人的事实（微积分基本定理）：
> **求面积（积分，加法过程）** 是 **求斜率（导数，乘法/除法过程）** 的逆运算。

数学表达：
$$\int_{a}^{b} f(x) dx = F(b) - F(a), \quad \text{其中 } F'(x) = f(x)$$

**这对我们的意义是什么？**
如果渲染方程是一个积分，那么它的导数（梯度）一定存在且可以计算！这保证了 3DGS 的可微分性。

---

## 🔥 Part 1: Volume Rendering Equation (VRE) —— 积分为何是物理真理？

### 场景：为什么 3DGS 需要积分？

当你看一个半透明的物体（比如烟雾、云层，或者 3DGS 的 Gaussian Splat），你看到的不只是表面颜色。
**你的眼睛接收了光线在穿过物体时沿途所有点颜色的叠加。**

物理上，这被称为“体积渲染方程”：
$$C(r) = \int_{t_n}^{t_f} T(t) \cdot \sigma(t) \cdot c(t) dt$$

让我们拆解这个公式的物理意义：
1.  **$c(t)$**：在光线位置 $t$ 处的颜色。
2.  **$\sigma(t)$**：不透明度（Density）。值越大，越容易遮挡后面的东西。
3.  **$T(t) = e^{-\int_0^t \sigma(s) ds}$**：**透射率 (Transmittance)**。这是光线到达 $t$ 点之前，没有被前面物体挡住而幸存下来的概率。

---

### 从连续积分到离散求和（代码实现）

在计算机里我们不能算无穷小量，只能做有限次加法。
我们将光线分成 $N$ 个段（或者每个 Gaussian 视为一个采样点），公式转化为：
$$C(r) \approx \sum_{i=1}^N T_i \cdot (1 - e^{-\sigma_i \delta_i}) \cdot c_i$$

其中：
*   $T_i = \prod_{j < i} (1 - \alpha_j)$ 表示光线穿过前 $i-1$ 个高斯后的“幸存率”。
*   $\alpha_i \approx 1 - e^{-\sigma_i \delta_i}$ 是第 $i$ 个高斯的几何不透明度（Alpha）。

**这就是 `render.py` 里那行代码的数学来源！**

---

## 🔥 Part 2: 微积分基本定理在 3DGS 中的体现

### 为什么 VRE 是可微分的？

因为 $T(t)$ 本身就是一个积分形式：
$$T(t) = \exp\left(-\int_0^t \sigma(s) ds\right)$$

当我们对 $\sigma$（密度）求导时，根据链式法则和微积分基本定理：
$$\frac{\partial T}{\partial \sigma(t)} = -T(t)$$
**这太优雅了！** 透射率对密度的导数仅仅是它自己的相反数。这意味着在反向传播时，梯度的计算极其高效且数值稳定。

---

## 🔥 Part 3: 积分在深度学习中的其他形式

除了体积渲染，你还经常遇到积分：
1.  **L2 Loss (MSE)**：本质是像素误差的积分 $\int (pred - target)^2 dx$（离散化就是求和）。
2.  **正则化**：如 Total Variation，是对梯度的范数做积分，防止图像出现噪点。

---

## 📚 习题

### ✅ 基础题（必做）

**9.1 物理直觉 —— 为什么需要 $T(t)$？**

(a) 如果场景中没有遮挡（全透明），$T(t)$ 应该等于多少？
(b) 如果前面有一个完全不透明的物体，光线能到达 $t$ 点吗？此时 $T(t)$ 是多少？

<details>
<summary>✅ 参考答案</summary>
(a) $\sigma = 0 \implies T = e^0 = 1$（光全部通过）。  
(b) 前面挡住了 → $T = 0$。此时该点颜色的贡献为 0，这正是 Alpha blending 中“被遮挡”的数学描述。
</details>

---

**9.2 离散化推导 —— 从积分到求和**

已知连续公式：$C = \int T(t) \sigma(t) c(t) dt$。
假设我们将光线切分为长度为 $\delta_i$ 的小段，且每段内密度恒定。
(a) 写出这一段对颜色的贡献 $dC$。
(b) 为什么 $T_{i+1} = T_i (1 - \alpha_i)$？（提示：从透射率定义出发）

<details>
<summary>✅ 参考答案</summary>
(a) 每一段的贡献近似为 $T_i \cdot (\sigma_i \delta_i) \cdot c_i$。  
(b) 透射率是“未被遮挡的概率”。如果第 $i$ 段遮挡了 $\alpha_i = \sigma_i \delta_i$ 的比例，那么幸存的透射率就是原来的 $(1-\alpha_i)$ 倍。
</details>

---

### 🔥 进阶题（选做）

**9.3 3DGS 中的“可微分积分”**

在 `render.py` 中：
```python
transmittance = torch.cumprod(torch.cat([torch.ones(..., device), (1 - alphas)[:-1]], dim=-1), dim=-1)
```
这行代码是在计算 $T_i$（透射率）。思考：如果 $\alpha$ 值接近 1，`cumprod` 运算会不会导致梯度消失？

<details>
<summary>✅ 参考答案</summary>
是的。因为 $T = \prod (1-\alpha)$，如果很多高斯都很不透明，$T$ 会指数级衰减趋近于 0。此时链式法则乘积项极小，导致后面的梯度传不过来（Gradient Vanishing）。这也是为什么 3DGS 需要 careful opacity initialization 的原因。
</details>

---

## 🔗 下一章

[→ Ch08: 3DGS 实战 - loss 到 Gaussian 参数的完整梯度追踪](./08_八_3DGS 实战_完整反向传播 trace_.md)

<div align="center">
**🔥 Ember's Note**: 积分 = "无穷小量的累积"。  
→ **建议**：理解 VRE 公式中 $T(t)$ 的指数形式，这是连接物理与代码的桥梁。
</div>
