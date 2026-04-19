# 3D Gaussian Splatting (3DGS) 教程

_从问题出发，到 feed-forward 推理的完整学习路径_

---

## 📖 教程导览

本教程采用 **第一性原理 + Just-in-Time** 框架：从 3DGS 的核心问题出发，按需推导所需的数学工具，最终回到代码实现。不事先学完所有数学再动手——而是在解决问题中发现需要什么、然后立即推导。

**核心脉络：**
- 🧮 **数学基础** → 遇到问题时推导所需工具（线性代数、微积分、概率论等）
- 🔗 **可微几何** → 从数学到渲染的桥梁
- 🎯 **3DGS 主线** → 从背景问题到 Feed-Forward 推理的完整 pipeline

---

## 🧮 数学基础（按需查阅或系统学习）

> **使用方式：** 在 3DGS 主线中遇到看不懂的公式时跳转到对应章节；或者按以下路径系统刷一遍。

### 1. 线性代数

| 章节 | 内容 |
|------|------|
| [Ch00 · 二次型](./linear_algebra_chapters/00_零_二次型_为什么所有_弯曲的边界_都能写成___mathbf_x__T_A__mathbf_x__.md) | $x^TAx$ — 一切弯曲边界的母语 |
| [Ch01 · 先把整张地图摊开](./linear_algebra_chapters/01_一_先把整张地图摊开.md) | 线性代数全景图 |
| [Ch02 · 重新发明3DGS被迫发明的数学](./linear_algebra_chapters/02_二_如果你要重新发明_3DGS_会被迫发明哪些数学_.md) | 为什么需要这些工具？ |
| [Ch03 · 向量](./linear_algebra_chapters/03_三_向量_它不是数组_它是一根箭头.md) | 向量 = 箭头，不是数组 |
| [Ch04 · 基与坐标](./linear_algebra_chapters/04_四_基与坐标_同一根箭头_为什么数字会变.md) | 坐标系变换的直觉 |
| [Ch05 · 矩阵](./linear_algebra_chapters/05_五_矩阵_它不是表格_它是_把整个空间一起推一下_.md) | 矩阵 = 空间变换 |
| [Ch06 · 协方差矩阵](./linear_algebra_chapters/06_六_协方差矩阵_为什么一个椭球可以被压进__3x3__数表里.md) | **3DGS 核心** — 高斯椭球的数学表示 |
| [Ch07 · 行列式与秩](./linear_algebra_chapters/07_七_行列式秩_几何视角的两个核心问题.md) | 体积压缩 & 维度坍缩 |
| [Ch08 · 特征值与特征向量](./linear_algebra_chapters/08_八_特征值与特征向量_变换最自然的方向.md) | 主方向 & PCA 基础 |
| [Ch09 · SVD](./linear_algebra_chapters/09_九_SVD_任意矩阵的三步拆解.md) | 万能矩阵分解 |
| [Ch10 · 低秩近似与压缩](./linear_algebra_chapters/10_十_低秩近似与压缩_从 SVD 到实际应用.md) | PCA、图像压缩实战 |
| [Ch11 · 回到3DGS](./linear_algebra_chapters/11_十一_现在回到_3DGS_这些数学到底在代码里干了什么.md) | 线性代数 → 代码映射 |
| [Ch12 · 易混淆点汇总](./linear_algebra_chapters/12_十二_这章最容易混淆的五个点.md) | 避坑指南 |
| [Ch13 · Python/Matplotlib实验](./linear_algebra_chapters/13_十三_五个可以直接跑的_Python___Matplotlib_实验.md) | 动手验证 |
| [Ch14 · 一页纸卡片](./linear_algebra_chapters/14_十四_最后把整章压成一页卡片.md) | 速查总结 |
| [Ch15 · 衔接3DGS主线](./linear_algebra_chapters/15_十五_接下来怎么接_3DGS_主线.md) | 下一步指引 |

**配套代码:** [`linear_algebra_chapters/code/`](./linear_algebra_chapters/code/) — Python 实验脚本

---

### 2. 数值线性代数

| 章节 | 内容 |
|------|------|
| [Ch01 · 矩阵分解与条件数](./numerical_linear_algebra_chapters/01_一_矩阵分解与条件数.md) | LU、QR、SVD — 数值稳定性的起点 |
| [Ch02 · Cholesky分解](./numerical_linear_algebra_chapters/02_二_Cholesky分解协方差矩阵的平方根密码.md) | **3DGS 参数化核心** — $Σ = LL^T$ |
| [Ch03 · SVD与低秩近似](./numerical_linear_algebra_chapters/03_三_SVD与低秩近似.md) | 数值计算视角的SVD |
| [Ch04 · PCA降维](./numerical_linear_algebra_chapters/04_四_PCA降维方差最大化到SVD.md) | 从方差最大到特征分解 |
| [Ch05 · 条件数深度分析](./numerical_linear_algebra_chapters/05_五_条件数深度分析优化与线性系统的命运之轮.md) | 数值稳定的生死线 |
| [Ch06 · 浮点数精度](./numerical_linear_algebra_chapters/06_六_浮点数精度与MachineEpsilon为什么01加02不等于03.md) | $0.1+0.2 \neq 0.3$ 的深层原因 |

---

### 3. 微积分

| 章节 | 内容 |
|------|------|
| [Ch01 · 先把整张地图摊开](./calculus_chapters/01_一_先把整张地图摊开.md) | 微积分全景图 |
| [Ch02 · 重新发明3DGS被迫发明的数学](./calculus_chapters/02_二_如果你要重新发明_3DGS_会被迫发明哪些数学_.md) | 为什么需要微积分？ |
| [Ch03 · 导数](./calculus_chapters/03_三_导数_为什么梯度是最优调整方向_.md) | 变化率 & 最优方向 |
| [Ch04 · 微分](./calculus_chapters/04_四_微分_线性近似的工程意义.md) | 局部线性化 |
| [Ch05 · 偏导数与梯度](./calculus_chapters/05_五_偏导数与梯度_.md) | 多维空间的坡向 |
| [Ch07 · 链式法则 & 反向传播](./calculus_chapters/07_七_链式法则_从第一性原理推导反向传播_.md) | **3DGS 训练核心** — autograd 的本质 |
| [Ch08 · 3DGS梯度追踪实战](./calculus_chapters/08_八_3DGS实战_loss到Gaussian参数的完整梯度追踪.md) | loss → Gaussian 参数全链路 |
| [Ch09 · 积分 & Volume Rendering](./calculus_chapters/09_九_积分_Volume_Rendering_Equation.md) | **3DGS 渲染核心** — 体积渲染方程 |
| [Ch10 · 泰勒展开](./calculus_chapters/10_十_泰勒展开_函数的局部DNA.md) | 函数逼近万能工具 |
| [Ch11 · 傅里叶级数](./calculus_chapters/11_十一_傅里叶级数_万物皆正弦波.md) | 周期信号分解 |
| [Ch12 · 傅里叶变换](./calculus_chapters/12_十二_傅里叶变换_时间的棱镜.md) | 频域分析基础 |
| [Ch13 · 拉普拉斯变换](./calculus_chapters/13_十三_拉普拉斯变换_微分方程的降维打击.md) | ODE/PDE 求解利器 |

---

### 4. 概率与信息论

| 章节 | 内容 |
|------|------|
| [Ch01 · 随机变量与概率分布](./probability_info_theory_chapters/01_一_随机变量与概率分布.md) | 不确定性建模基础 |
| [Ch02 · 期望方差与高阶矩](./probability_info_theory_chapters/02_二_期望方差与高阶矩.md) | 分布的统计特征 |
| [Ch03 · 条件概率与贝叶斯定理](./probability_info_theory_chapters/03_三_条件概率与贝叶斯定理.md) | 从结果推断原因 |
| [Ch04 · 常见分布](./probability_info_theory_chapters/04_四_常见分布高斯伯努利多项式BetaDirichlet.md) | 高斯、伯努利、Beta、Dirichlet |
| [Ch05 · Shannon熵](./probability_info_theory_chapters/05_五_Shannon熵信息量的货币单位.md) | 信息的量化 |
| [Ch06 · 交叉熵](./probability_info_theory_chapters/06_六_交叉熵CrossEntropy损失函数的灵魂.md) | **3DGS 损失函数灵魂** — CrossEntropy |
| [Ch07 · KL散度](./probability_info_theory_chapters/07_七_KL散度KL_Divergence两个分布的距离度量.md) | 分布之间的距离 |
| [Ch08 · 互信息](./probability_info_theory_chapters/08_八_互信息MutualInformation变量之间的粘性度量.md) | 变量间的依赖关系 |
| [Ch09 · Gaussian与高斯Splatting](./probability_info_theory_chapters/09_九_Gaussian分布与高斯Splatting为什么用高斯.md) | **3DGS 为什么用高斯** — 闭合性质的必然性 |
| [Ch10 · MLE最大似然估计](./probability_info_theory_chapters/10_十_最大似然估计MLE参数拟合的灵魂.md) | 从数据反推参数 |
| [Ch11 · 渲染损失设计](./probability_info_theory_chapters/11_十一_信息论视角下的渲染损失设计.md) | 信息论 → 损失函数设计 |

---

### 5. 优化理论

| 章节 | 内容 |
|------|------|
| [Ch01 · 凸性与拉格朗日乘子](./optimization_chapters/01_一_凸性与拉格朗日乘子.md) | 约束优化的数学基础 |
| [Ch02 · 梯度与泰勒展开](./optimization_chapters/02_二_梯度与泰勒展开.md) | 一阶 & 二阶近似 |
| [Ch03 · SGD随机梯度下降](./optimization_chapters/03_三_无约束优化SGD随机梯度下降.md) | **3DGS 训练基础** — SGD 原理 |
| [Ch04 · Adam自适应学习率](./optimization_chapters/04_四_动量与Adam给梯度加上惯性并自适应学习率.md) | **3DGS 实际使用** — AdamW |
| [Ch05 · BFGS拟牛顿法](./optimization_chapters/05_五_拟牛顿法BFGS用梯度近似曲率地图.md) | 用一阶信息逼近二阶优化 |
| [Ch06 · Hessian与二阶优化](./optimization_chapters/06_六_二阶优化与Hessian曲率决定优化的命运.md) | 曲率如何决定收敛速度 |

---

### 🔗 数学 → 3DGS 桥梁：可微几何

| 章节 | 内容 |
|------|------|
| [Ch01 · 曲率与测地线](./differential_geometry_chapters/01_一_曲率与测地线.md) | 曲面最基本的性质 |
| [Ch02 · 黎曼几何入门](./differential_geometry_chapters/02_二_黎曼几何入门.md) | 弯曲空间的微积分 |
| [Ch03 · 高斯与平均曲率](./differential_geometry_chapters/03_三_高斯与平均曲率.md) | 内蕴 vs 外蕴曲率 |
| [Ch04 · 形状算子与投影退化](./differential_geometry_chapters/04_四_形状算子与投影退化.md) | **3DGS 视角退化** — 为什么高斯会变成扁椭圆 |
| [Ch05 · 流形上的坐标变换](./differential_geometry_chapters/05_五_流形上的坐标变换.md) | 局部坐标系的数学基础 |
| [Ch06 · 雅可比矩阵与退化详解](./differential_geometry_chapters/06_六_雅可比矩阵与退化情况详解.md) | **3DGS 核心** — 世界→屏幕映射的 Jacobian & 协方差变换 |

> **为什么把可微几何放在数学基础之后？** 因为它不是"纯数学"，而是连接抽象数学工具到具体渲染操作的桥梁。学完上面所有基础后看这一组章节最顺畅。

---

## 🎯 3DGS 主线（核心内容）

> 按顺序阅读，遇到公式看不懂时跳转到上方「数学基础」对应章节。

| 章节 | 内容 |
|------|------|
| [Ch00 · 数学基础导览](./3dgs_chapters/chapter_00_math_foundation.md) | **必读** — 为什么学这些数学？Just-in-Time 学习策略 |
| [Ch01 · 背景与问题](./3dgs_chapters/chapter_01_background.md) | Novel View Synthesis、NeRF 的渲染瓶颈 |
| [Ch02 · 表示方法的演化](./3dgs_chapters/chapter_02_representation_evolution.md) | 体素 → 点云 → NeRF → 高斯椭球 |
| [Ch03 · Gaussian Splatting](./3dgs_chapters/chapter_03_gaussian_splatting.md) | 3DGS 的核心表示：带协方差的高斯椭球 |
| [Ch04 · 可微分渲染](./3dgs_chapters/chapter_04_differentiable_rendering.md) | 从离散点到连续图像的渲染方程 |
| [Ch05 · 优化问题形式化](./3dgs_chapters/chapter_05_optimization.md) | 损失函数、参数空间、优化目标 |
| [Ch06 · 数据准备](./3dgs_chapters/chapter_06_data_preparation.md) | COLMAP → 相机位姿 → 训练集构建 |
| [Ch07 · 训练流程](./3dgs_chapters/chapter_07_training_loop.md) | 完整训练 loop：初始化 → 采样 → 渲染 → 优化 |
| [Ch08 · 推理优化](./3dgs_chapters/chapter_08_inference_optimization.md) | Pruning、量化、加速渲染 |
| [Ch09 · 实战路径](./3dgs_chapters/chapter_09_practice_path.md) | 从理论到代码的落地路线 |
| [Ch10 · 4D Gaussian Splatting](./3dgs_chapters/chapter_10_4d_gaussian.md) | 动态场景扩展 |
| [Ch11 · Feed-Forward Gaussian](./3dgs_chapters/chapter_11_feedforward_gaussian.md) | 即时重建，一步推理 |

---

## 📊 学习路径图

```
┌───────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
│  数学基础     │    │   可微几何       │    │ 3DGS 主线    │    │ VAE & Diffusion│
│               │    │                  │    │              │    │ (独立模块)    │
│ · 线性代数    │───→│ · 曲率/测地线    │───→│ · Ch00-11    │    │   · VAE      │
│ · 数值线性代数│    │ · 黎曼几何       │    │              │    │   · Diffusion│
│ · 微积分      │    │ · 雅可比/投影    │    │              │    │              │
│ · 概率论      │───→│                  │    │              │    │              │
│ · 优化理论    │    │                  │    │              │    │              │
└───────────────┘    └──────────────────┘    └──────────────┘    └──────────────┘
```

> **VAE & Diffusion** 是独立模块，与 3DGS 主线没有强依赖关系。想看就直接进入 [`vae_diffusion.md`](./vae_diffusion.md)。

---

## 🗑️ 归档文件（不再维护）

| 文件 | 说明 |
|------|------|
| [线性代数.md](./trash/linear_algebra.md) | V1 — 已被 `linear_algebra_chapters/` 取代 |
| [calculus.md](./trash/calculus.md) | V1 — 已被 `calculus_chapters/` 取代 |
| [optimization.md](./trash/optimization.md) | V1 — 已被 `optimization_chapters/` 取代 |
| [probability_info_theory.md](./trash/probability_info_theory.md) | V1 — 已被 `probability_info_theory_chapters/` 取代 |
| [numerical_linear_algebra.md](./trash/numerical_linear_algebra.md) | V1 — 已被 `numerical_linear_algebra_chapters/` 取代 |
| [math_foundation衔接点.md](./trash/math_foundation衔接点.md) | 旧版衔接说明 |
