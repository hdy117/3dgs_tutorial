# Ch06 — 浮点数精度与 Machine Epsilon：为什么 `0.1+0.2 ≠ 0.3`？

> **本章目标**：理解 IEEE754 双精度/单精度的存储机制、Machine Epsilon，以及它在数值计算中的灾难性影响。  
> **前置知识**：Ch01-05 (SVD, Cholesky, PCA, Condition Number)。  
> **核心问题**：为什么计算机的算术运算会出错？`0.1 + 0.2 == 0.3` 在 Python 中是 True 还是 False？

---

## 🎯 问题驱动：浮点数误差如何毁掉你的训练？

### 场景 1：PyTorch 训练中突然出现的 NaN

```python
# Loss = nan! 排查后发现: B_inv = torch.linalg.inv(B) 
# 某个 Splat 的协方差 B 的对角线元素累积了微小的舍入误差，
# 最终导致 det(B) < 0（理论上不可能！），求逆崩溃。

# 问题：浮点数误差为什么会累积？如何检测和预防？
```

**关键问题**：IEEE754 浮点数的精度极限是什么？什么操作会导致误差爆炸式增长？

---

## 📐 Part 1: IEEE 754 标准 — "计算机如何表示小数"

### Boxed Result：双精度 (Float64) 的存储格式 ⚔️

```
| Sign (1 bit) | Exponent (11 bits) | Mantissa/Significand (52 bits) |
|   -1^s        |      E-1023         |    1.f (隐含前导1)             |
```

**公式**：$\boxed{x = (-1)^S \times 2^{E-1023} \times (1.f)}$

其中 $f$ 是 52-bit 的小数部分。

### 💡 Boxed Result：单精度 vs 双精度对比

| 类型 | 总位数 | 小数位 (mantissa) | 有效数字 | Machine Epsilon ($\epsilon_{\text{mach}}$) |
|------|--------|-------------------|----------|-------------------------------------------|
| **float32** | 32 bit | 23 bits | ~7 位 | $2^{-23} \approx 1.19\times 10^{-7}$ |
| **float64** | 64 bit | 52 bits | ~16 位 | $2^{-52} \approx 2.22\times 10^{-16}$ |

### 💡 核心洞察：Machine Epsilon = "最小可分辨的相对误差"
$$\boxed{\epsilon_{\text{mach}} = \text{机器能表示的最小相对差异}}$$

这意味着：**任何 $\delta < \epsilon_{\text{mach}}$ 的相对变化都会被浮点数吞掉！** 这就是为什么 $1 + 10^{-20} == 1.0$。

---

## 🔥 Part 2: Machine Epsilon — 严格推导与性质

### Boxed Result：ε_machine 的定义

对于 base-2 floating point system，$\epsilon_{\text{mach}} = 2^{-(p-1)}$，其中 $p$ 是 mantissa 位数（包括隐含的 leading bit）。
- **float32**: $\epsilon = 2^{-24} \approx 6\times 10^{-8}$ (或定义为单位舍入误差: $2^{-23}$)。
- **float64**: $\epsilon = 2^{-53} \approx 1.1\times 10^{-16}$。

### 💡 Boxed Result：浮点数运算的误差传播 ⚔️

对于基本算术运算 $\oplus$（加/减/乘/除）：
$$\boxed{\text{fl}(a \odot b) = (a \odot b)(1 + \delta), \quad |\delta| \leq \epsilon_{\text{mach}}}$$

**关键推论**：
- **加法/减法**：误差最大（两个相近数相减 → 有效数字丢失 → "catastrophic cancellation"）。
- **乘法/除法**：误差较小，但会累积。

---

## 💻 Part 3: PyTorch 验证 — 浮点数精度实战

```python
import torch
import numpy as np

# ============================================================
# 1. Machine Epsilon 实测 ⚔️
# ============================================================
print("=== Machine Epsilon ===")

# --- Float64 (double) ---
eps_double = np.finfo(np.float64).eps
eps_float = np.finfo(np.float32).eps
print(f"float64 ε_machine: {eps_double:.3e}")  # ≈ 2.22e-16
print(f"float32 ε_machine: {eps_float:.3e}")   # ≈ 1.19e-7

# --- Python: 0.1 + 0.2 == 0.3? ⚠️ ---
python_result = (0.1 + 0.2) == 0.3
print(f"\nPython: 0.1 + 0.2 == 0.3 ? {python_result} ❌ (False!)")
print(f"实际值: {0.1+0.2:.20f}, 期望值: {0.3:.20f}")

# --- PyTorch float vs double 对比 ---
x = torch.tensor([0.1], dtype=torch.float64) + torch.tensor([0.2], dtype=torch.float64)
y = torch.tensor([0.3], dtype=torch.float64)
diff_double = (x - y).abs().item()

x_f32 = torch.tensor([0.1], dtype=torch.float32) + torch.tensor([0.2], dtype=torch.float32)
y_f32 = torch.tensor([0.3], dtype=torch.float32)
diff_float32 = (x_f32 - y_f32).abs().item()

print(f"\nPyTorch float64: |0.1+0.2-0.3| = {diff_double:.3e}")
print(f"PyTorch float32: |0.1+0.2-0.3| = {diff_float32:.3e} (更大!)")

# ============================================================
# 2. Catastrophic Cancellation — "相近数相减的灾难" ⚔️
# ============================================================
print("\n=== Catastrophic Cancellation ===")

a, b = torch.tensor([1.0], dtype=torch.float64), torch.tensor([1.0 + 1e-12], dtype=torch.float64)

exact_diff = (b - a).item()
computed_diff = float(a - b) # ←←←  catastrophic cancellation!

print(f"精确差值: {exact_diff:.3e}")
print(f"浮点计算差值: {abs(computed_diff):.3e} → 完全错误!")
print(f"相对误差: {(abs(exact_diff - computed_diff) / exact_diff * 100):.4f}%")

# --- 数值稳定替代方案 ---
import math
stable_diff = torch.tensor([math.hypot(1, 0)], dtype=torch.float64) # 不直接相减!
print(f"使用 hypot/cbrt 等稳定函数可减少误差 ✅")

# ============================================================
# 3. 误差在矩阵运算中的累积 ⚔️ (Cholesky/求逆)
# ============================================================
print("\n=== 误差累积: Cholesky + SVD ===")

np.random.seed(42)
A = torch.randn(10, 10, dtype=torch.float32)
A_spd = A @ A.t() # SPD

# --- float32 Cholesky vs float64 Cholesky ---
L_f32 = torch.linalg.cholesky(A_spd)
L_f64 = torch.linalg.cholesky(A_spd.double())

reconstructed_f32 = L_f32 @ L_f32.t() - A_spd.float()
reconstructed_f64 = L_f64 @ L_f64.t() - A_spd.double()

print(f"float32 Cholesky 重建误差: {(reconstructed_f32**2).sum():.3e}")
print(f"float64 Cholesky 重建误差: {(reconstructed_f64**2).sum():.3e}")

# 💡 float32 的 ε ≈ 10⁻⁷ → 重建误差在 ~10⁻¹³~10⁻¹⁴ range
#    float64 的 ε ≈ 10⁻¹⁶ → 重建误差在 ~10⁻²⁸~10⁻³⁰ range (几乎完美!)

# ============================================================
# 4. PyTorch 训练中的 dtype 选择策略 ⚔️
# ============================================================
print("\n=== PyTorch: float32 vs float64 实战 ===")

model = torch.nn.Linear(100, 50)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_train = torch.randn(32, 100)
y_target = torch.randn(32, 50)

# float32 训练
model.float()
out_f32 = model(x_train.float())
loss_f32 = ((out_f32 - y_target)**2).mean().item()

# float64 训练
model.double()
out_f64 = model(x_train.double())
loss_f64 = ((out_f64 - y_target.double())**2).mean().item()

print(f"float32 Loss: {loss_f32:.4f}")
print(f"float64 Loss: {loss_f64:.4f}")
print(f"\n浮点精度对训练的影响:")
print("  float32: 快 ✅ | 显存省 ✅ | 数值稳定性 ⚠️ (可能 NaN)")
print("  float64: 慢 ❌ | 显存翻倍 ❌ | 数值稳定 ✅")

# 💡 最佳实践: 训练用 float16/bfloat16 (混合精度), 
#     梯度累加用 float32, 关键矩阵运算(求逆/Cholesky)用 float64!
```

---

## 🗺️ Part 5: 与 3DGS 的衔接点 — 浮点数误差如何影响渲染？

### Boxed Result：3DGS 中的浮点数敏感操作

| 操作 | 数据类型 | 风险 | 缓解策略 |
|------|----------|------|---------|
| **协方差求逆 $B^{-1}$** | float32 | det(B) ≈ 0 → inf/NaN | ε-Stabilizer (float64 计算后 cast 回 float32) |
| **指数函数 $\exp(-r^T B^{-1} r)$** | float32 | $r$ 过大时 exp→0，数值下溢 | clamp $r^T B^{-1} r$ 在合理范围 |
| **SH (球谐函数) 系数更新** | float32 | SH 值域 [-1,1] → 累积误差导致 >1/ < -1 | 梯度裁剪 + L2 正则化 |

### 💡 Boxed Result：为什么 3DGS 推荐用 float32？

```python
# PyTorch 默认 float32 (2.4 billion colors). 
# 对于 GPU 渲染管线，float32 是最佳平衡:
# - 显存: 2x 比 float64 → 可以容纳更多 Splat!
# - 速度: NVIDIA Tensor Core/RTX 对 float32 有硬件加速.
# - 精度: float32 ε≈10⁻⁷，对于大多数渲染任务足够（人眼无法分辨 10⁻⁷ 的差异）。
```

---

## 🎓 本章小结

### 核心公式 (Boxed)

$$\boxed{\epsilon_{\text{mach}} = 2^{-(p-1)}, \quad p=53 (\text{float64}), 24 (\text{float32})}$$

$$\boxed{\text{fl}(a \odot b) = (a \odot b)(1+\delta), |\delta| \leq \epsilon_{\text{mach}}}$$

### 关键洞察

> **Machine Epsilon 是浮点数的"最小分辨率"** —— float32: ~7 位有效数字, float64: ~16 位。任何小于 ε 的相对变化都会被吞掉！
> 
> **"0.1+0.2≠0.3"的本质**：十进制小数在二进制浮点数中无法精确表示（像 1/3 在十进制中一样）。这是 IEEE754 的根本限制。
> 
> **catastrophic cancellation = 相近数相减 → 有效数字丢失** —— 这是数值计算中最常见的误差来源，比矩阵求逆崩溃更隐蔽！

---

## 📚 习题

### ✅ 基础题

**6.1** 证明：对于 float32，$1 + 10^{-8} == 1.0$（True）。
<details>
<summary>💡 提示</summary> float32 ε≈1.19×10⁻⁷。$10^{-8}$ < $10^{-7}$ → 在精度范围内被舍入掉，所以 $1+10^{-8} = 1$ (IEEE754 round-to-nearest)。
</details>

**6.2** 为什么 float32 的指数范围是 [-126, +127]？这对 PyTorch 训练有什么影响？
<details>
<summary>💡 提示</summary> exp 范围太大时会出现 overflow (inf) 或 underflow (0)。float32 最大 ≈ $3.4\times 10^{38}$，最小非零 ≈ $1.17\times 10^{-38}$。如果梯度超过这个范围 → inf/NaN！
</details>

### 🔥 进阶题

**6.3** (数值稳定性)：计算 $(a-b)(a+b)$ 和 $a^2-b^2$，为什么前者更不稳定？
<details>
<summary>💡 提示</summary> 如果 a≈b，$(a-b)$ 会 catastrophic cancellation（有效数字丢失），然后乘上 $(a+b)$ 放大误差。而 $a^2-b^2$ 虽然也有类似问题，但现代 CPU/GPU 对乘法有更精确的硬件实现。
</details>

### 💡 3DGS 关联题

**6.4** (dtype 选择)：在 3DGS 训练中，哪些操作应该用 float64 计算？哪些可以用 float16？给出具体建议。
<details>
<summary>💡 提示</summary> 
- **必须 float64**: 协方差矩阵求逆 ($B^{-1}$)、Cholesky 分解、det(B) 检测。
- **推荐 float32**: 前向/反向传播的大部分计算（SH系数更新、位置变换）。
- **推荐 float16/bfloat16 (混合精度)**: 渲染管线中的纹理采样、alpha blending（人眼无法分辨差异，但节省显存）。
</details>

---

# 📇 数值线性代数 — 一页纸总结卡片 (Cheat Sheet)

```markdown
# 🧱 Numerical Linear Algebra Cheat Sheet (3DGS Edition)
## Part 1: IEEE 754 浮点数
- **Float64**: ε=2⁻⁵²≈2.2e-16 (~16位有效数字).
- **Float32**: ε=2⁻²³≈1.2e-7 (~7位有效数字).
- **Machine Epsilon**: 最小可分辨相对误差!

## Part 2: 浮点运算误差 ⚔️
- **fl(a⊙b) = (a⊙b)(1+δ)**, |δ|≤ε_machine.
- **Catastrophic Cancellation**: 相近数相减 → 有效数字丢失!
- **"0.1+0.2≠0.3"** → 十进制小数在二进制中无法精确表示.

## Part 3: 混合精度策略 (PyTorch/3DGS)
| 操作 | dtype | 原因 |
|------|-------|------|
| Cholesky/求逆 | float64 | 数值稳定性优先 |
| 前向传播 | float32 | 平衡精度与速度 |
| 渲染管线 | float16/bf16 | 显存节省, 人眼不可辨 |
| 梯度累加 | float32 | 防止精度丢失 |

## Part 4: Hessian & Condition Number ⚔️
- **κ=σ_max/σ_min → GD收敛速率 ((κ-1)/(κ+1))²**
- **Adam ≈ Preconditioned GD**: v_t≈Hessian diagonal自动调节步长!

## Part 5: 3DGS 数值稳定性清单
✅ ε-Stabilizer (B+εI) → 降低 κ(B).
✅ float64 计算关键矩阵运算.
✅ float16/bf16 混合精度渲染.
✅ Gradient clipping → 防止梯度溢出/NaN.

## 🏆 核心洞察
> "浮点数不是'精确的算术'. ε_machine 是它的'分辨率极限'."
> "数值稳定的代码 ≠ 数学上正确的代码 — 它必须考虑浮点误差!"
```

---

> **Ch06 (Numerical LA) 完成！** 🔥  
> 
> Part 7 下一站：**Ch07 — Matrix Norms & Vector Norms** —— 理解 L1, L2, Frobenius, Spectral norms 的区别，以及它们如何决定 3DGS Loss function 的选择。直接说 "继续"。
