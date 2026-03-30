# 第 3 章：核心发明——为什么最后选了高斯？

**本章核心问题**：如果你必须发明一种新的 3D primitive，让它既稀疏、又连续、还能快速投影到图像上并支持反向传播，为什么最后会落到 Gaussian，而不是点、方块、球，或者别的椭球？

这一章真正要回答的不是"高斯是什么定义"，而是：

> 为什么 3DGS 的核心发明，最后会是 $\mu + \Sigma + \alpha + \text{color}$ 这套表示，而不是别的东西。

---

## 一、先从点云的困境开始

### 1.1 点为什么天然会有洞

点云最开始看起来很诱人，因为它真的很省：

- 每个点只存位置和颜色
- 点和点互不耦合，结构简单
- 稀疏场景下内存开销很小

但它有一个根本问题：

> **点是零体积的。**

零体积意味着什么？意味着你把它投到屏幕上时，它更像"若干离散针尖"，而不是连续表面。

```text
你真正想看到的是连续物体

xxxxxxxxxxxx
xxxxxxxxxxxx
xxxxxxxxxxxx

而只投点常常更像

x .  x   . x
.   . x .  .
x .    .  x
```

于是马上会出现：
- 点和点之间有空洞
- 视角一变，空洞位置也跟着变
- 分辨率一提高，离散感会更明显
- 遮挡边界会变得脆，容易闪烁

所以新视角合成里，只有点通常不够。你缺的不是"再多一点点"，而是另一种 primitive。

### 1.2 我们真正缺的不是更多点，而是局部连续影响

更准确地说，我们缺的是：

> **一个围绕某个 3D 位置展开、对附近区域连续起作用的局部体积元。**

你可以把需求压成一句话：

```text
一个点只能说"这里有一个位置"
而一个好的 primitive 要能说"这里附近有一小团可以连续影响成像的局部结构"
```

这就是为什么 3DGS 不满足于"点"，而要找一个更像局部云团的表示。

---

## 二、如果真要发明一个 primitive，它必须同时满足什么

这一步很重要，因为它决定了你不是在"欣赏高斯"，而是在做工程筛选。

如果你真站在发明者角度，会被下面五个约束同时逼住：

| 约束 | 为什么必须满足 |
|------|----------------|
| **稀疏** | 否则会退回体素那类把整个空间填满的高开销方案 |
| **连续** | 否则屏幕上还是会有硬边、空洞和闪烁 |
| **可各向异性** | 否则没法表达"沿某个方向长、沿另一个方向薄"的局部表面 |
| **投影要快** | 否则会重新掉进每像素重采样的慢渲染 |
| **可微** | 否则图像误差没法稳定传回位置、形状和透明度 |

所以真正的问题不是：

```text
能不能找个 3D 形状来代替点
```

而是：

```text
能不能找一个在这五个约束下最自然的局部 primitive
```

这就是 3DGS 的出发点。

---

## 三、把候选都摆上桌：为什么最后不是别的东西

### 3.1 候选对比表

你最自然会想到下面这些候选：

| Primitive | 直觉上的优点 | 真正的问题 |
|-----------|--------------|------------|
| **点** | 最省、最简单 | 零体积，投影天然有洞 |
| **小方块 / billboard** | 好画，易实现 | 边界硬，视角变化时容易假 |
| **球** | 连续，参数少 | 只能各向同性，表达局部表面太笨 |
| **一般椭球** | 可各向异性 | 只把它当"几何体"还不够，变换和投影后不够自然 |
| **高斯椭球** | 连续、可各向异性、局部衰减自然、变换后仍然好处理 | 你必须接受它是"密度云"而不是硬表面 |

### 3.2 为什么固定小方块不够好

最简单的想法是：

```text
别投点了
给每个点画一个固定大小的小方块
```

这确实比点更"有面积"，但问题也很直接：
- 边界是硬的
- 贴片感很重
- 遮挡边缘容易不自然
- 旋转和透视变化下很容易露馅

它更像"往屏幕上贴纸片"，不像"在 3D 空间里有一团局部结构"。

### 3.3 为什么球也不够

球比方块好得多，因为它：
- 连续
- 没有棱角
- 任何方向看起来都一样

但这最后反而成了问题。真实场景的局部几何很少真像球，它更常见的是：
- 沿表面切向方向展开得更宽
- 沿法线方向更薄

比如一片叶子、一条桌腿边缘、一小片墙面，往往都更像"扁平或拉长的局部结构"，而不是球。

所以球的致命问题不是不平滑，而是：

> **它太各向同性了。**

### 3.4 一般椭球其实已经很接近正确答案

如果继续推，很自然就会得到椭球。

因为椭球正好允许：
- 三个主轴方向长度不同
- 整体可以旋转到任意朝向
- 局部表面可以表现成"薄片""细杆""小团块"等不同形态

所以从几何表达能力上，椭球几乎已经到位。

真正决定胜负的，是下一步：

> **这个东西放进完整渲染链之后，谁更好算、谁更好传、谁更好求导？**

这时候 Gaussian 就开始显著领先了。

---

## 四、高斯公式到底在说什么

### 4.1 先把 3D 高斯写出来

一个 3D Gaussian 的完整密度写法是：

$$
\rho(\mathbf{x}) = \frac{1}{(2\pi)^{3/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

这里：
- $\boldsymbol{\mu}$ 是中心
- $\boldsymbol{\Sigma}$ 是协方差矩阵
- $|\boldsymbol{\Sigma}|$ 是行列式
- $\boldsymbol{\Sigma}^{-1}$ 决定"这个点离中心有多远"时，采用什么椭球距离度量

如果你只记一句人话版，那就是：

> **离 $\boldsymbol{\mu}$ 越近，密度越大；离 $\boldsymbol{\mu}$ 越远，密度按一种平滑、各向异性的方式快速衰减。**

### 4.2 为什么 3DGS 里很多时候不强调前面的归一化常数

严格的概率密度形式里，前面有：

$$
\frac{1}{(2\pi)^{3/2} |\boldsymbol{\Sigma}|^{1/2}}
$$

但在 3DGS 的渲染里，很多时候更关心的是：
- footprint 的形状
- 相对权重
- alpha 混合时的局部贡献

这时常会把一些常数吸收到 $\alpha$ 或别的权重里，于是更常见的是只保留指数核：

$$
g(\mathbf{x}) = \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

这不表示前面的常数不重要，而是表示：

> **从工程角度，形状和相对衰减往往比"它是否是一个严格归一化概率密度"更关键。**

### 4.3 $\boldsymbol{\Sigma}^{-1}$ 到底在干什么

看式子里最关键的这项：

$$
(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})
$$

它不是普通欧氏距离平方，而是一个**椭球距离**。

如果某个方向本来很宽，那么朝这个方向偏一点，代价就不会太大。
如果某个方向本来很窄，那么同样的偏移就会被认为"很远"。

所以这项真正描述的是：

> **在这个高斯自己的主轴和尺度定义下，一个点离中心有多远。**

### 4.4 $|\boldsymbol{\Sigma}|$ 又在干什么

$|\boldsymbol{\Sigma}|$ 控制的是整体体积尺度。

更准确地说，如果把 $\boldsymbol{\Sigma}$ 分解到主轴坐标里，设主轴标准差是 $s_1, s_2, s_3$，那么：

$$
|\boldsymbol{\Sigma}| = s_1^2 \cdot s_2^2 \cdot s_3^2
$$

于是：

$$
|\boldsymbol{\Sigma}|^{1/2} = s_1 \cdot s_2 \cdot s_3
$$

它本质上和椭球体积尺度相关。

所以你可以把高斯密度理解成两部分：
- **二次型**决定"形状上的距离"
- **行列式**决定"整体摊得有多开"

---

## 五、为什么 $\boldsymbol{\Sigma}$ 就是在说"一个椭球"

### 5.1 先看等密度面

把高斯值固定在同一水平上，你得到的不是任意怪形状，而是：

$$
(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = k^2
$$

这就是一个**椭球面**。

所以更准确地说：

> **高斯不是"一个实心椭球"，而是一团密度云；这团云的等密度壳层是椭球。**

这句话非常关键。

### 5.2 把 $\boldsymbol{\Sigma}$ 分解开看最直观

只要 $\boldsymbol{\Sigma}$ 是对称正定矩阵，它就可以写成：

$$
\boldsymbol{\Sigma} = \mathbf{R} \cdot \text{diag}(s_1^2, s_2^2, s_3^2) \cdot \mathbf{R}^\top
$$

这里：
- $\mathbf{R}$ 给出主轴方向（旋转矩阵）
- $s_1, s_2, s_3$ 给出三个主轴的标准差尺度

于是这条式子的几何意义特别直接：

```text
先在局部坐标里放一个轴对齐椭球
再用 R 把它转到场景里的正确朝向
```

### 5.3 为什么这和 3DGS 的局部表面表达天然契合

因为局部表面本来就常常长这样：
- 两个切向方向比较宽
- 法向方向比较薄

一个合适的 $\boldsymbol{\Sigma}$ 可以自然表达这种结构。例如：

$$
\boldsymbol{\Sigma} = \mathbf{R} \cdot \text{diag}(0.03^2, 0.02^2, 0.002^2) \cdot \mathbf{R}^\top
$$

这就像一个被压薄的椭球云，很像局部表面片。

### 5.4 为什么实现里通常不直接裸优化 $\boldsymbol{\Sigma}$

虽然几何上 $\boldsymbol{\Sigma}$ 是核心，但实际训练里更常见的是优化：
- $\text{scale} = (s_1, s_2, s_3)$
- $\text{rotation} = \mathbf{R}(\mathbf{q})$ 或别的旋转参数化

再重建：

$$
\boldsymbol{\Sigma} = \mathbf{R}(\mathbf{q}) \cdot \text{diag}(s_1^2, s_2^2, s_3^2) \cdot \mathbf{R}(\mathbf{q})^\top
$$

这样做是因为：
- 更容易保证正定性
- 更容易约束尺度为正
- 更适合数值优化

所以要同时记住两层：

```text
几何理解层：Sigma 是椭球
实现层：常常存 scale + rotation，再重建 Sigma
```

---

## 六、为什么高斯特别适合工程链，而不只是"形状好看"

到这里，Gaussian 真正的优势才开始显现。

### 6.1 第一重优势：在线性/仿射变换下特别听话

如果有：

$$
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad \mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}
$$

那么：

$$
\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \; \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)
$$

这句话极其重要，因为它意味着：

> **你对一个 Gaussian 做旋转、平移、拉伸之后，它不会变成一种完全不好处理的新对象；它还是 Gaussian，只是中心和协方差变了。**

这跟后续章节会直接接上：

$$
\begin{aligned}
&\boldsymbol{\mu}_\text{cam} = \mathbf{R} \cdot \boldsymbol{\mu}_\text{world} + \mathbf{t} \\
&\boldsymbol{\Sigma}_\text{cam} = \mathbf{R} \cdot \boldsymbol{\Sigma}_\text{world} \cdot \mathbf{R}^\top
\end{aligned}
$$

### 6.2 第二重优势：全局投影虽然非线性，但局部还能被 Jacobian 拉回线性框架

透视投影本身是：

$$
\begin{aligned}
&u = f_x \cdot \frac{X}{Z} + c_x \\
&v = f_y \cdot \frac{Y}{Z} + c_y
\end{aligned}
$$

这当然不是全局线性的。

但一个高斯本来就是局部对象，所以在中心附近，小扰动 $d\mathbf{x}$ 可以写成：

$$
d\mathbf{p} \approx \mathbf{J} \cdot d\mathbf{x}
$$

于是协方差就能继续传播：

$$
\boldsymbol{\Sigma}_{2D} \approx \mathbf{J} \cdot \boldsymbol{\Sigma}_\text{cam} \cdot \mathbf{J}^\top
$$

这里你应该能明显看见那条主线：

> **工程上并不是假装世界全局线性，而是尽量把"小范围内的传播"压成线性结构。**

Gaussian 和这个思路天生相配。

### 6.3 第三重优势：局部衰减自然，特别适合做 screen-space footprint

Gaussian 的值离中心会快速衰减，所以它天然满足：
- 影响区域是局部的
- 远离中心的像素贡献会快速变小
- 很容易做包围盒和 tile culling

这非常重要，因为实时渲染最怕的是：

```text
每个 primitive 都去影响整张图
```

Gaussian 不是。它既连续，又局部。

### 6.4 第四重优势：整条主链大部分都可微

从 $\boldsymbol{\mu}$、$\boldsymbol{\Sigma}$ 到 $\boldsymbol{\mu}_\text{cam}$、$\boldsymbol{\Sigma}_\text{cam}$，再到 $\boldsymbol{\Sigma}_{2D}$、2D Gaussian 值和 alpha blending，这一路上大部分都是：
- 矩阵乘法
- 加法
- 指数函数
- 二次型

也就是说，它不仅好表示、好投影，还好求导。

这就是 3DGS 成为"可训练系统"而不是"静态几何技巧"的关键原因。

---

## 七、在 3DGS 里，一个高斯到底存什么

如果先用最简化版本，一个高斯可以写成：

$$
G_i = \{\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i, c_i\}
$$

其中：
- $\boldsymbol{\mu}_i$：中心位置（3D）
- $\boldsymbol{\Sigma}_i$：局部形状（协方差矩阵）
- $\alpha_i$：不透明度 / 密度强度
- $c_i$：颜色

如果想写得更贴近实际实现，往往会是：

$$
G_i = \{\boldsymbol{\mu}_i, s_i, \mathbf{q}_i, \rho_i, \text{SH}\}
$$

这里：
- $s_i + \mathbf{q}_i$ 共同决定 $\boldsymbol{\Sigma}_i$
- $\rho_i$ 经过适当参数化后得到 $\alpha_i$
- $\text{SH}$ 用球谐系数描述视角相关颜色

所以第 3 章真正建立起来的语言是：

```text
3DGS 的 primitive 不是"一个点"
而是"一团有中心、有形状、有透明度、有外观的局部 Gaussian 云"
```

---

## 八、三个工程画面：为什么它比别的 primitive 更像场景局部结构

### 8.1 树叶边缘

树叶边缘不是一个硬立方块，也不是一颗球。它更像：
- 沿叶片面展开
- 沿法线方向很薄
- 边缘渐变而不是突然截断

Gaussian 椭球特别适合这种"薄而连续"的局部结构。

### 8.2 桌腿或栏杆

这类结构常常是：
- 一个方向细长
- 另外两个方向很窄

球表达它会很浪费，因为必须在所有方向都同样大。各向异性 Gaussian 则可以把"长细杆"直接表达出来。

### 8.3 一小片墙面或地面

大平面局部常常像：
- 两个切向方向大
- 法向方向小

这本来就是"扁平椭球云"的天然形状。

所以 3DGS 的 primitive 并不是在勉强适配真实世界，而是：

> **它本来就和局部表面片的几何统计结构很像。**

---

## 九、一个特别有用的比喻：高斯不是石头，是软印章

如果你总把高斯想成"一个实心椭球体"，很容易越想越别扭。

更好的比喻是：

> **一个高斯像一枚软印章，或者一小团雾。**

它的特点是：
- 中心压得最重
- 往外逐渐变淡
- 形状可以是圆的，也可以是扁的、斜的、拉长的
- 多个印章叠在一起，会自然混出更复杂的结构

这个比喻特别重要，因为 3DGS 从来不是在恢复一堆"硬表面小零件"，而是在恢复：

```text
许多局部连续密度 footprint 的叠加
```

这也正是后面 alpha blending 能工作的直觉基础。

---

## 十、一个最小可运行实验：看各向同性和各向异性 Gaussian 的 footprint 差别

下面这个例子不碰 3DGS 训练，只做一件最重要的事：

> **把"$\boldsymbol{\Sigma}$ 决定 footprint 形状"这件事变成肉眼直觉。**

```python
import numpy as np
import matplotlib.pyplot as plt


def gaussian_2d(grid_x, grid_y, mu, Sigma):
    pos = np.stack([grid_x - mu[0], grid_y - mu[1]], axis=-1)
    inv = np.linalg.inv(Sigma)
    q = np.einsum('...i,ij,...j->...', pos, inv, pos)
    return np.exp(-0.5 * q)


x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)

mu = np.array([0.0, 0.0])
Sigma_iso = np.array([
    [0.8**2, 0.0],
    [0.0, 0.8**2],
])

theta = np.deg2rad(35)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)],
])
S = np.diag([1.6**2, 0.35**2])
Sigma_aniso = R @ S @ R.T

G_iso = gaussian_2d(X, Y, mu, Sigma_iso)
G_aniso = gaussian_2d(X, Y, mu, Sigma_aniso)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, G, title in [
    (axes[0], G_iso, 'isotropic Gaussian'),
    (axes[1], G_aniso, 'anisotropic Gaussian'),
]:
    ax.contourf(X, Y, G, levels=25, cmap='viridis')
    ax.contour(X, Y, G, levels=[0.1, 0.3, 0.6], colors='white', linewidths=1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()
```

你应该看到：
- **左图**的等值线是圆，说明所有方向尺度相同
- **右图**的等值线是旋转过的椭圆，说明 $\boldsymbol{\Sigma}$ 同时编码了主轴方向和各向异性尺度
- **这就是 3DGS 中 2D footprint 的最核心视觉直觉**

如果你愿意再走一步，可以把右图里的 `theta` 和两个轴长改一改，亲手观察 $\boldsymbol{\Sigma}$ 怎样改变 footprint 的朝向和胖瘦。

---

## 十一、把整章压成一条最短心智模型

如果你只想记一条链，就记这个：

```text
点云太稀，直接投点会有洞

    ↓

我们需要一个围绕 3D 位置展开的局部连续影响区域

    ↓

这个 primitive 必须同时满足：稀疏、连续、可各向异性、投影快、可微

    ↓

Gaussian 最合适，因为它既像一个椭球云
又能把形状、变换、局部投影和像素贡献都压进线性代数框架

    ↓

于是一个 primitive 就自然写成
G_i = {mu_i, Sigma_i, alpha_i, color_i}
```

这就是第 3 章真正要你建立起来的语言。

---

## 十二、思考题（检验理解）

### Q1: "为什么点云直接投影会有空洞？"

<details>
<summary>点击查看答案</summary>

点是**零维物体**，在 3D 空间没有体积，投影到 2D 也没有面积。当相机发射射线时，99.9% 的射线会从点的间隙中穿过，无法采样到任何点颜色，形成空洞。

关键点：
- 维度不匹配：0D → 0D
- 没有"覆盖范围"的概念
- 即使增加点数，仍然会有采样间隙（除非填满空间，那就退化成体素了）
</details>

### Q2: "为什么球不够好？椭球才接近正确答案？"

<details>
<summary>点击查看答案</summary>

**球的限制**：各向同性（所有方向尺度相同）。

真实场景的局部几何很少像球，更常见的是：
- 沿表面切向方向展开得更宽
- 沿法线方向更薄（比如一片叶子、墙面）

**椭球的优势**：可以表达各向异性结构，三个主轴可以有不同的长度。这使得它能自然地拟合"扁平""细长"等常见的局部几何形态。

关键洞察：**表达能力**比"看起来平滑"更重要。
</details>

### Q3: "为什么 Gaussian 的 $\boldsymbol{\Sigma}^{-1}$ 不是普通欧氏距离？"

<details>
<summary>点击查看答案</summary>

$\boldsymbol{\Sigma}^{-1}$ 定义了**椭球距离度量**。

考虑二次型：
$$
(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})
$$

如果某个方向 $\boldsymbol{\Sigma}$ 很宽（标准差大），那么在这个方向上移动相同的距离，二次型的增长会**比较小**（代价小）。
反之，如果某个方向很窄，同样的偏移会被认为"很远"。

这正是各向异性的体现：不同方向的"远近"由不同的尺度决定。
</details>

### Q4: "为什么高斯在线性变换下还是高斯？这个性质多重要？"

<details>
<summary>点击查看答案</summary>

如果 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 且 $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$，那么：

$$
\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \; \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)
$$

**重要性**：
1. **变换一致性**：旋转、平移、拉伸后不会变成"难以处理的新对象"
2. **工程友好**：$\boldsymbol{\mu}_\text{cam} = \mathbf{R}\boldsymbol{\mu}_\text{world} + \mathbf{t}$ 和 $\boldsymbol{\Sigma}_\text{cam} = \mathbf{R}\boldsymbol{\Sigma}_\text{world}\mathbf{R}^\top$ 可以直接套用
3. **可微性**：所有变换都是矩阵运算，可以反向传播

如果没有这个性质，每次变换都要重新学一个新的分布形状，渲染链会崩溃。
</details>

### Q5: "为什么高斯局部衰减的特性对实时渲染很重要？"

<details>
<summary>点击查看答案</summary>

Gaussian 的指数衰减特性意味着：
- **影响范围有限**：远离中心的像素贡献快速趋近于 0
- **可裁剪**：可以计算包围盒，只处理可能受影响的像素（tile culling）
- **避免全局影响**：每个 primitive 不会"污染"整张图

这对实时渲染至关重要，因为：
```text
O(N_primitive × N_pixel) → O(∑ footprint_size) << O(N_primitive × N_pixel)
```
实际计算量远小于 naive 实现。
</details>

### Q6: "为什么 3DGS 的 primitive 最后是 $\{\boldsymbol{\mu}, \boldsymbol{\Sigma}, \alpha, c\}$ 这四个元素？"

<details>
<summary>点击查看答案</summary>

四个元素的**必要性分析**：

| 元素 | 作用 | 少了会怎样？ |
|------|------|-------------|
| $\boldsymbol{\mu}$ | 3D 位置中心 | 不知道高斯在哪，无法投影 |
| $\boldsymbol{\Sigma}$ | 形状（各向异性 + 朝向） | 只能表达球，无法拟合局部表面 |
| $\alpha$ | 不透明度 / 密度强度 | 无法控制贡献权重，混合会出错 |
| $c$ | 颜色 | 只能渲染灰度图 |

**四个缺一不可**，它们分别对应：
- **位置**（where）
- **形状**（what shape）
- **强度**（how strong）
- **外观**（what color）

这就是 3DGS primitive 的完整信息集。
</details>

---

## 十三、下一章预告

下一章 **[chapter_04_differentiable_rendering.md](chapter_04_differentiable_rendering.md)** 会接着回答另一半问题：

> **既然 primitive 已经选成了 Gaussian，那它到底怎样从 3D 里的 $\boldsymbol{\mu}$ 和 $\boldsymbol{\Sigma}$，一步步变成屏幕上的 2D footprint、像素颜色和可反向传播的渲染链？**

也就是从：

```text
"为什么最后选高斯"
```

走到：

```text
"高斯怎样真正变成一张图，并把梯度传回去"
```

这两章接起来，才是 3DGS 最核心的数学骨架。

---

### 📝 **学习检查站**

到这里，你应该能回答（遮住答案自己讲）：
1. ✅ **为什么点云天然会有洞？**（零维物体无面积）
2. ✅ **为什么我们真正需要的是"局部连续影响"，而不是单个点？**（避免采样间隙）
3. ✅ **为什么球不够好，椭球却很接近正确答案？**（各向异性表达能力）
4. ✅ **为什么高斯不是硬表面，而是一团密度云？**（等密度面是椭球壳层）
5. ✅ **为什么 $\boldsymbol{\Sigma}$ 可以直接看成一个椭球结构？**（$\boldsymbol{\Sigma} = \mathbf{R}\cdot\text{diag}(s_i^2)\cdot\mathbf{R}^\top$）
6. ✅ **为什么 Gaussian 不只是形状好看，而是特别适合完整工程链？**（线性变换保持 + 局部衰减 + 可微）
7. ✅ **为什么 3DGS 的 primitive 最后自然长成 $\{\boldsymbol{\mu}, \boldsymbol{\Sigma}, \alpha, c\}$ 这套形式？**（位置 + 形状 + 强度 + 外观 = 完整信息集）

如果这些问题你能自己从头讲回来，这一章就真的进脑子了。🔥

---

*本章完成时间：约 25-30 分钟阅读 | 核心推导：4 重优势分析 | 记忆负担：中（需要建立心智模型）*