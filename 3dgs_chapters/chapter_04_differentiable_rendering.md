# 第 4 章:从 3D 高斯到图像--可微渲染链到底怎样成立

**本章核心问题**:第 3 章已经解释了为什么 primitive 最后选成 Gaussian。现在真正的问题变成:

> 这些 3D 高斯到底怎样一步步变成屏幕上的像素?而且,为什么这个过程大部分还能反向传播,让图像误差流回 `mu`、`Sigma`、`alpha` 和颜色参数?

如果上一章解决的是:"为什么是高斯",这一章解决的就是:"高斯怎样真的变成一张图,并把梯度传回去"。

先把整条主线写在前面:

$$
\begin{aligned}
&\text{3D 高斯} \\
&\to \text{世界到相机变换} \\
&\to \text{局部投影成 2D 椭圆 footprint} \\
&\to \text{按深度组织} \\
&\to \text{只在局部 tile 内参与计算} \\
&\to \text{front-to-back alpha blending} \\
&\to \text{图像}
\end{aligned}
$$

图像误差再沿这条链大部分反向流回去。

---

## 一、先别掉进实现细节,先看整条渲染链

每个高斯最终变成像素,大致会经历四步:

1. **3D 高斯从世界空间送到相机空间**,再局部投影成屏幕上的 2D 椭圆 footprint
2. **给这些 footprint 一个前后顺序**
3. **不让每个像素看全部高斯**,只看真正覆盖自己的那一小批
4. **把这批高斯按顺序做 alpha blending**,得到最终颜色

这条链里,你要一直同时盯住两个目标:

- 它必须快,否则就回到每像素重采样的慢渲染
- 它必须大部分可微,否则就没法训练

3DGS 的漂亮之处,不在于某一步特别花哨,而在于:

> 它把表示、投影、筛选、混合这四步都压成了足够规则的结构。

---

## 二、输入和输出到底是什么

### 2.1 输入不是"一个场景",而是一堆带参数的 Gaussian

如果先用最简化形式,一个高斯可以写成:

$$G_i = \{\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i, \mathbf{c}_i\}$$

其中:

- $\boldsymbol{\mu}_i$:3D 中心
- $\boldsymbol{\Sigma}_i$:3D 协方差,对应一个 3D 椭球结构
- $\alpha_i$:不透明度 / 密度强度
- $\mathbf{c}_i$:颜色

如果写得更贴近真实实现,也常常会是:

$$G_i = \{\boldsymbol{\mu}_i, s_i, \mathbf{r}_i, \rho_i, \text{sh}_i\}$$

但无论参数化怎么换,本章真正关心的还是那条物理主线:

> 中心在哪?形状怎样?颜色怎样?遮挡和透射怎样发生?

### 2.2 输出是一张图像

对于给定相机 $\text{cam}_k$,渲染器最终输出的是:

$$C(p) \in \mathbb{R}^3$$

也就是每个像素 $p = (u, v)$ 的 RGB 颜色。问题在于:> 怎么从许多个 3D Gaussian,得到每个像素的最终颜色?

---

## 三、第一步:先把 3D Gaussian 送到相机坐标系

这一部分是整条渲染链的数学起点。

### 3.1 世界到相机:中心怎么变

如果一个高斯中心在世界坐标里是 $\boldsymbol{\mu}_\text{world}$,相机外参是 $(R, \mathbf{t})$,那么相机坐标下的中心就是:

$$\boldsymbol{\mu}_\text{cam} = R \cdot \boldsymbol{\mu}_\text{world} + \mathbf{t}$$

这里 $R$ 是 $3 \times 3$ 旋转矩阵,$\mathbf{t}$ 是平移向量。这件事不神秘,就是普通的刚体变换。

### 3.2 世界到相机:形状怎么变

第 3 章已经说过,高斯最重要的几何结构是 $\boldsymbol{\Sigma}$。它在相机坐标下会变成:

$$\boldsymbol{\Sigma}_\text{cam} = R \, \boldsymbol{\Sigma}_\text{world} \, R^\top$$

这条式子特别值得记住,因为它意味着:

> 高斯经过坐标变换后,还是高斯;只是中心和协方差按线性代数规则更新。

这就是 Gaussian 比很多别的局部表示更"听话"的地方。

### 3.3 到这里你已经得到了什么

到这一步,每个高斯都已经从"世界里的局部椭球云"变成了"相机眼里的局部椭球云"。

也就是说,你已经知道:

- 它在相机前方还是后方
- 它离相机有多远
- 它在相机坐标系里是什么朝向和尺度

接下来才轮到真正的成像问题。

---

## 四、第二步:高斯中心怎样投到屏幕上

### 4.1 中心投影是精确的

设相机坐标中的高斯中心是 $\boldsymbol{\mu}_\text{cam} = [X, Y, Z]^\top$,那么它的像素坐标中心由标准透视投影给出:

$$u = f_x \cdot \frac{X}{Z} + c_x, \quad v = f_y \cdot \frac{Y}{Z} + c_y$$

其中 $f_x, f_y$ 是焦距尺度,$c_x, c_y$ 是主点偏移。所以 2D footprint 的中心位置很好算:$\boldsymbol{\mu}_{2\text{d}} = [u, v]^\top$。

### 4.2 但中心好算,不等于整个形状也好算

麻烦在于,透视投影本身不是线性的:

$$(X, Y, Z) \mapsto (X/Z, Y/Z)$$

问题就出在那个 $1/Z$ 上。这意味着:

- 你不能拿一个固定的 $2 \times 3$ 矩阵,把所有 3D 协方差都一次性"精确投"到 2D
- 一个 3D 椭球经过透视投影后,整体形状会受深度影响而扭曲

所以真正的问题不是中心在哪,而是:这个 3D 椭球在屏幕上局部会变成什么 footprint?

---

## 五、第三步:为什么 `Sigma_2d ≈ J * Sigma_cam * J^T` 会出现

这一节是整章最关键的数学桥梁。

### 5.1 透视投影全局非线性,但高斯本来就很局部

虽然投影函数全局非线性,但每个 Gaussian 本来就只占一个局部小区域。于是我们不需要精确追踪"整个椭球被怎样非线性扭曲",只需要关心:> 在高斯中心附近,一个很小的 3D 扰动 $\mathrm{d}\boldsymbol{x}$ 会怎样映射成屏幕上的 2D 扰动 $\mathrm{d}\boldsymbol{p}$?这时就可以做一阶线性化:

$$\mathrm{d}\boldsymbol{p} \approx J \cdot \mathrm{d}\boldsymbol{x}$$

这里 $J$ 是投影函数在当前中心处的 Jacobian。

### 5.2 把 Jacobian 明确写出来

对投影函数:

$$u = f_x \cdot \frac{X}{Z} + c_x, \quad v = f_y \cdot \frac{Y}{Z} + c_y$$

在点 $(X, Y, Z)$ 处的 Jacobian 是:

$$J = \begin{bmatrix} \frac{f_x}{Z} & 0 & -\frac{f_x X}{Z^2} \\[1em] 0 & \frac{f_y}{Z} & -\frac{f_y Y}{Z^2} \end{bmatrix}$$

这条式子的几何意义很直观:$f_x/Z, f_y/Z$ 告诉你屏幕尺度会随着深度变化,而 $-\frac{f_x X}{Z^2}, -\frac{f_y Y}{Z^2}$ 告诉你深度方向扰动也会影响屏幕位置。

### 5.3 一旦局部线性化成立,协方差自然就能传播

如果局部有 $\mathrm{d}\boldsymbol{p} \approx J \cdot \mathrm{d}\boldsymbol{x}$,那么根据线性代数里协方差传播的标准规则,2D 协方差就是:

$$\boldsymbol{\Sigma}_{2\text{d}} \approx J \, \boldsymbol{\Sigma}_\text{cam} \, J^\top$$

这条式子非常有名,但更重要的是你要理解它在说什么:> 屏幕上的 2D 椭圆,不是随手画出来的,而是 3D 椭球在当前视点附近经过局部线性化后传播出来的 footprint。

### 5.4 这就是工程上"把非线性压回线性"的典型例子

如果你还记得 [../linear_algebra_engineering_perspective.md](../linear_algebra_engineering_perspective.md),这里其实正是在用那条最核心的工程思想:全局是非线性的,但局部足够小,就先用 Jacobian 把传播拉回线性框架。3DGS 并没有假装透视投影是线性的,它做的是:> 对一个足够小的局部高斯来说,一阶近似已经够用,而一阶近似恰好能把协方差传播写成线性代数最擅长的形式。

---

## 六、第四步:从 2D 椭圆 footprint 到像素权重

到这里,每个高斯已经在屏幕上对应:

- 一个 2D 中心 `mu_2d`
- 一个 2D 协方差 `Sigma_2d`

也就是一个 2D Gaussian。

### 6.1 单个像素离高斯中心有多远

对某个像素 $p$,定义它相对中心的偏移:$\mathrm{d} = p - \boldsymbol{\mu}_{2\text{d}}$。然后用二次型衡量"在这个椭圆几何下,它离中心有多远":$$q = \mathrm{d}^\top \, \boldsymbol{\Sigma}_{2\text{d}}^{-1} \, \mathrm{d}$$这里的 $q$ 不是普通欧氏距离平方,而是:> 按照这个椭圆自己的主轴尺度计算出来的距离。于是 Gaussian 值就是:$$g(p) = \exp\left(-\frac{1}{2} q\right)$$如果像素刚好在中心,$q = 0$,所以 $g(p) = 1$;如果像素离中心越来越远,$g(p)$ 就会快速衰减。

### 6.2 再把透明度乘进去

单个高斯在像素 $p$ 处的有效不透明度可以写成:$$w_i(p) = \alpha_i \cdot g_i(p)$$这件事直观上很好理解:$\alpha_i$ 决定这团 Gaussian 整体"有多实",$g_i(p)$ 决定这个像素在它的 footprint 里离中心有多近。两者相乘后,才是它在这个像素上的真实贡献强度。

---

## 七、第五步:为什么一定要关心深度顺序

一个像素通常不会只落在一个 Gaussian 的 footprint 里,而是会被多个 Gaussian 同时覆盖。

这时必须回答一个问题:

> 谁在前,谁在后?

### 7.1 为什么不能简单相加

如果直接把所有颜色贡献求和,会忽略遮挡。

但真实成像里,近处结构应该挡住一部分远处结构。所以多个 Gaussian 不是完全可交换的贡献源。

### 7.2 一个像素真正经历的是"剩余透射率"变化

front-to-back blending 的直觉是这样的:一开始,像素对背景完全透明,剩余透射率 $T_1(p) = 1$;经过第一个高斯后,透射率会减少;后面的高斯只能使用剩下那部分"可见性预算"。于是对按深度从近到远排序后的高斯,常见写法是:

$$
\begin{aligned}
&T_1(p) = 1 \\
&C(p) = \sum_i T_i(p) \cdot w_i(p) \cdot c_i \\
&T_{i+1}(p) = T_i(p) \cdot (1 - w_i(p))
\end{aligned}
$$

其中 $w_i(p) = \alpha_i \cdot g_i(p)$,$c_i$ 是颜色,$T_i(p)$ 是到第 $i$ 个高斯之前还剩多少透射率。这就是为什么 blending 不是简单加法,而是递推过程。

### 7.3 为什么 3DGS 喜欢 front-to-back

因为它允许一个特别实用的优化:

> 如果某个像素的剩余透射率已经非常小了,后面的高斯几乎不可能再产生明显影响,就可以提前停。

这对实时渲染很重要。

---

## 八、为什么不能让每个像素遍历所有高斯

### 8.1 最朴素复杂度会直接炸

如果图像大小是 $H \times W$,高斯数是 $N$,最朴素的做法是每个像素都遍历全部 $N$ 个高斯。复杂度大致接近:$$O(H \cdot W \cdot N)$$这在高斯数量达到几十万、上百万时几乎不可接受。

### 8.2 Gaussian 最大的工程红利之一:局部性

Gaussian 的贡献离中心会快速衰减,所以它天然只影响一小块屏幕区域。

这带来一个非常宝贵的事实:

- 不是所有高斯都和所有像素有关
- 每个高斯通常只覆盖少数 tile
- 每个 tile 只需要处理与自己相交的那批 Gaussian

### 8.3 Tile-based culling 的核心直觉

屏幕可以被切成小块,例如 $16 \times 16$ 的 tiles。然后做三件事:

1. 根据 $\boldsymbol{\Sigma}_{2\text{d}}$ 给每个 Gaussian 算一个 `k-sigma` 包围盒
2. 看它覆盖哪些 tile
3. 建立 `tile -> relevant gaussians` 的映射

这样每个像素就不再面对全部 $N$ 个高斯,而只看自己所在 tile 里那一小批相关项。

### 8.4 这一步为什么不是"小优化",而是结构性改变

因为它不是"把同样的事算快一点",而是:

> 把问题从"全局所有像素 x 所有高斯"改写成"局部 tile x 局部高斯"。

这正是 Gaussian 既连续又局部所带来的巨大工程红利。

---

## 九、哪些部分严格可微,哪些部分只是工程上足够可用

这一节非常重要,因为"可微渲染"常常容易被误解成整个系统的每个细节都处处光滑。实际不是这样。

### 9.1 主体链路里,哪些部分天然可微

下面这些基本都很顺:世界到相机的线性变换、透视中心投影里的除法(只要 $Z$ 不太接近 0)、Jacobian 线性化、协方差传播 $J \boldsymbol{\Sigma}_\text{cam} J^\top$、2D Gaussian 的指数核评估、front-to-back blending 的连续部分。所以图像损失的主干梯度,的确可以大部分传回:$$L \to C(p) \to w_i(p) \to \boldsymbol{\mu}_{2\text{d}}, \boldsymbol{\Sigma}_{2\text{d}} \to \boldsymbol{\mu}_\text{cam}, \boldsymbol{\Sigma}_\text{cam} \to \boldsymbol{\mu}, \boldsymbol{\Sigma}, \alpha, \mathbf{c}$$

### 9.2 哪些部分不是严格光滑的

真正"不那么连续"的地方主要有:

- 深度排序
- tile 分配
- footprint 截断
- 提前终止 early stop

这些都带有离散或近似色彩。

但 3DGS 仍然工作得很好,原因通常是:

- 相邻迭代里高斯不会疯狂乱跳
- 排序关系大多数时候相对稳定
- 主体梯度还是沿连续主链在流
- 离散部分主要影响计算组织,而不是完全摧毁监督信号

所以更准确的说法是:

> 3DGS 依赖的是一条工程上足够可微的渲染链,而不是一个在纯数学意义上每一步都处处光滑的系统。

### 9.3 数值稳定为什么也属于"可微渲染成立"的一部分

理论上可微,不等于数值上稳。实际实现里通常还要处理:$Z$ 太小时 Jacobian 爆炸、$\boldsymbol{\Sigma}_{2\text{d}}$ 接近奇异导致求逆不稳、特别大的 footprint 让局部线性化变差。所以工程上常见的保护包括:$$Z = \text{clamp}(Z, \min=\varepsilon), \quad \boldsymbol{\Sigma}_{2\text{d}} = \boldsymbol{\Sigma}_{2\text{d}} + \varepsilon I$$对 footprint 做合理截断,训练中把过大的 Gaussian 拆细。这不是"补丁",而是让理论模型在数值世界真正站住脚的必要步骤。

---

## 十、一个具体工程画面:为什么它比 NeRF 快得多

这一节不是为了吹性能,而是为了看清"计算结构到底换了什么"。

### 10.1 NeRF 的典型结构

传统 NeRF 更像对每条光线,沿深度采很多点,每个点都查网络,再做体渲染积分。慢的根源不是代码写得差,而是它的计算结构天然很重。

### 10.2 3DGS 的结构变化

3DGS 则是先把 3D primitive 直接投成 2D 局部 footprint,再在屏幕空间做局部混合。这带来三个巨大变化:

1. 不再需要每像素沿深度采很多样本
2. 不再对每个样本都跑网络查询
3. 计算天然带强局部性,特别适合 tile 和 GPU 并行

所以 3DGS 的快,不是"同样体渲染写得更快",而是:> 它把问题改写成了更接近图形管线、也更接近矩阵和局部 footprint 计算的结构。

---

## 十一、一个最小可运行实验:看几枚 2D Gaussian 怎样被前向混合成一张图

下面这段代码不做 3D 投影,只做这一章最核心的一步:

> 给定几个已经在屏幕上的 2D Gaussian footprint,按前后顺序做 front-to-back alpha blending,看看一张图是怎样长出来的。

```python
import numpy as np
import matplotlib.pyplot as plt


H, W = 220, 220
xs = np.linspace(0, W - 1, W)
ys = np.linspace(0, H - 1, H)
X, Y = np.meshgrid(xs, ys)


def gaussian_map(mu, Sigma):
    pos = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    inv = np.linalg.inv(Sigma)
    q = np.einsum('...i,ij,...j->...', pos, inv, pos)
    return np.exp(-0.5 * q)


gaussians = [
    {
        'depth': 1.0,
        'mu': np.array([90.0, 110.0]),
        'Sigma': np.array([[900.0, 180.0], [180.0, 500.0]]),
        'alpha': 0.70,
        'color': np.array([1.0, 0.35, 0.20]),
    },
    {
        'depth': 1.8,
        'mu': np.array([130.0, 95.0]),
        'Sigma': np.array([[650.0, -120.0], [-120.0, 420.0]]),
        'alpha': 0.65,
        'color': np.array([0.20, 0.70, 1.00]),
    },
    {
        'depth': 2.4,
        'mu': np.array([115.0, 145.0]),
        'Sigma': np.array([[500.0, 0.0], [0.0, 320.0]]),
        'alpha': 0.55,
        'color': np.array([0.95, 0.90, 0.25]),
    },
]

# front-to-back: 深度越小越靠前
sorted_gaussians = sorted(gaussians, key=lambda g: g['depth'])

C = np.zeros((H, W, 3), dtype=np.float64)
T = np.ones((H, W, 1), dtype=np.float64)
alpha_maps = []

for g in sorted_gaussians:
    footprint = gaussian_map(g['mu'], g['Sigma'])[..., None]
    w = g['alpha'] * footprint
    alpha_maps.append(w[..., 0])

    C += T * w * g['color']
    T *= (1.0 - w)

background = np.ones((H, W, 3), dtype=np.float64)
C += T * background
C = np.clip(C, 0.0, 1.0)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for i in range(3):
    axes[i].imshow(alpha_maps[i], cmap='magma')
    axes[i].set_title(f'footprint {i+1}')
    axes[i].axis('off')

axes[3].imshow(C)
axes[3].set_title('front-to-back blend')
axes[3].axis('off')

plt.tight_layout()
plt.show()
```

你应该看到:

- 前三个子图是各自的 2D footprint,都是平滑的椭圆分布
- 最后一张图不是简单叠加,而是按剩余透射率递推混出来的
- 如果你改动 `depth` 顺序,最终颜色叠放关系也会变

这就是 3DGS 中"screen-space splat + alpha blending"的最小视觉直觉。

---

## 十二、把整章压成一个最短心智模型

如果你只想记一条链,就记这个:$$\begin{aligned}&\text{3D Gaussian 先用 } R \boldsymbol{\mu} + \mathbf{t}, \, R \boldsymbol{\Sigma} R^\top \text{ 送到相机空间} \\&\downarrow \\&\text{中心投影是精确的:} u = f_x X/Z + c_x, \, v = f_y Y/Z + c_y \\&\downarrow \\&\text{整个透视投影全局非线性,但对一个局部 Gaussian,可以在中心处做 Jacobian 线性化} \\&\downarrow \\&\text{于是 3D 协方差传播成 2D 协方差:}\boldsymbol{\Sigma}_{2\text{d}} \approx J \boldsymbol{\Sigma}_\text{cam} J^\top \\&\downarrow \\&\text{每个 Gaussian 在屏幕上变成一个 2D 椭圆 footprint} \\&\downarrow \\&\text{按深度组织、按 tile 局部筛选,再做 front-to-back alpha blending} \\&\downarrow \\&\text{得到图像,并把大部分梯度沿连续主链传回去}\end{aligned}$$这就是第 4 章真正想让你建立起来的渲染骨架。

---

## 十三、本章你真正应该能自己重建的几个问题

读完以后,你至少应该能自己讲清楚:

1. 为什么 $\boldsymbol{\mu}_\text{cam} = R \boldsymbol{\mu}_\text{world} + \mathbf{t}$、$\boldsymbol{\Sigma}_\text{cam} = R \boldsymbol{\Sigma}_\text{world} R^\top$ 会自然出现?
2. 为什么高斯中心投影是精确的,而整个形状投影不是?
3. 为什么 $J$ 会成为从 3D 到 2D footprint 的关键桥梁?
4. 为什么 $\boldsymbol{\Sigma}_{2\text{d}} \approx J \boldsymbol{\Sigma}_\text{cam} J^\top$ 不是魔法,而是局部线性化的直接结果?
5. 为什么单个像素处的高斯权重要写成 $\alpha_i \exp(-q/2)$ 这种形式?
6. 为什么 blending 一定要关心顺序,而不是简单求和?
7. 为什么 tile-based culling 不是"小优化",而是实时渲染成立的关键条件?
8. 为什么 3DGS 可以被叫做"可微渲染",但又不等于每个细节都严格光滑?

如果这些问题你能自己从头推回来,这一章就真的进脑子了。

---

## 十四、本章练习题

---

### Q1: Jacobian 线性化的尺度效应是什么？

在透视投影中，Jacobian 矩阵为：

$$J = \begin{bmatrix} \frac{f_x}{Z} & 0 & -\frac{f_x X}{Z^2} \\[1em] 0 & \frac{f_y}{Z} & -\frac{f_y Y}{Z^2} \end{bmatrix}$$

**问题：** 当高斯从近处（$Z=1$）移动到远处（$Z=4$），屏幕上的 2D 协方差 $\boldsymbol{\Sigma}_{2\text{d}} \approx J \boldsymbol{\Sigma}_\text{cam} J^\top$ 的尺度会如何变化？定量计算缩放因子。

<details>
<summary>提示</summary>
只考虑对角线上的主尺度项 $f_x/Z, f_y/Z$，忽略深度扰动项（$-fX/Z^2$）。思考：Z 变为 4 倍时，这些项会变成原来的多少？
</details>

<details>
<summary>答案</summary>

当 $Z = 1$ 时，Jacobian 主尺度为 $\frac{f_x}{1}, \frac{f_y}{1}$。
当 $Z = 4$ 时，Jacobian 主尺度变为 $\frac{f_x}{4}, \frac{f_y}{4}$。

2D 协方差传播公式：$\boldsymbol{\Sigma}_{2\text{d}} \approx J \boldsymbol{\Sigma}_\text{cam} J^\top$。

由于 $J$ 的对角线元素缩小了 4 倍，而 $\boldsymbol{\Sigma}_{2\text{d}}$ 是 $J$ 的二次型（$J$ 左乘再右乘 $J^\top$），所以：

$$\boldsymbol{\Sigma}_{2\text{d}}(Z=4) \approx \left(\frac{1}{4} J(Z=1)\right) \boldsymbol{\Sigma}_\text{cam} \left(\frac{1}{4} J(Z=1)\right)^\top = \frac{1}{16} \boldsymbol{\Sigma}_{2\text{d}}(Z=1)$$

**结论：** 当深度从 $Z=1$ 增加到 $Z=4$（扩大 4 倍），屏幕上的 2D 协方差缩小为原来的 **1/16**。这就是为什么远处的物体在图像上看起来更小——不仅是中心投影的线性缩放，协方差本身以平方关系收缩。
</details>

---

### Q2: front-to-back blending 的梯度流路径

考虑单像素处的 front-to-back blending：

$$\begin{aligned}T_1(p) &= 1 \\C_i(p) &= C_{i-1}(p) + T_i(p) \cdot w_i(p) \cdot c_i \\T_{i+1}(p) &= T_i(p) \cdot (1 - w_i(p))\end{aligned}$$

其中 $w_i(p) = \alpha_i \cdot g_i(p)$。

**问题：** 假设最终像素颜色为 $C_N$（经过 N 个高斯后的结果），求 $\frac{\partial C_N}{\partial \alpha_k}$，即第 k 个高斯的不透明度对最终颜色的梯度。

<details>
<summary>提示</summary>
使用链式法则：$\alpha_k$ → $w_k$ → $(C_k, T_{k+1})$ → ... → $C_N$。注意 $T$ 的递推关系会把梯度传递到后面的所有高斯。
</details>

<details>
<summary>答案</summary>

使用链式法则，分两部分：直接影响（通过 $C_k$）和间接影响（通过 $T_{k+1}$ 传播）。

**直接项：** $\frac{\partial C_N}{\partial \alpha_k}\big|_\text{direct} = T_k(p) \cdot g_k(p) \cdot c_k$

**间接项（通过后续透射率）：**
对于任意 $j > k$，第 $k+1$ 个高斯到第 $j-1$ 个高斯的传递链为：
$$\frac{\partial C_j}{\partial T_{k+1}} = \sum_{m=k+1}^{j-1} w_m(p) c_m + (T_k - T_j)(1-w_j)c_j$$

综合起来，完整梯度为：
$$\frac{\partial C_N}{\partial \alpha_k} = g_k(p) \left[ T_k(p) c_k + (1 - c_k) \sum_{j=k+1}^N T_j(p) w_j(p) \right]$$

**物理意义：** 第 k 个高斯的不透明度梯度由两部分组成：
1. **直接贡献** $T_k c_k$：它自己在当前剩余透射率下的颜色贡献
2. **遮挡效应** $(1-c_k) \sum T_j w_j$：它遮挡了后面所有高斯的程度

这就是为什么前面的高斯梯度通常更大——它们控制着"可见性预算"的分配。
</details>

---

### Q3: Tile-based culling 的复杂度收益

假设图像尺寸为 $H \times W = 1024 \times 1024$，总共有 $N = 10^6$ 个高斯。屏幕被切成 $T = (1024/16)^2 = 4096$ 个 tile（每个 tile 为 $16 \times 16$）。

**问题：** 
(a) 朴素算法的复杂度是多少？
(b) 假设每个高斯的 footprint 平均覆盖 4 个 tile，每个像素平均只看到 10 个相关高斯。使用 tile-based culling 后，有效计算量是多少？
(c) 速度提升倍数是多少？

<details>
<summary>提示</summary>
朴素：$O(H \cdot W \cdot N)$。Tile-based：每个 pixel 只看自己所在 tile 的相关高斯数。注意总像素数是 $H \times W$，不是 tile 数。
</details>

<details>
<summary>答案</summary>

**(a) 朴素算法复杂度：**
$$O(H \cdot W \cdot N) = O(1024 \cdot 1024 \cdot 10^6) \approx 10^{12} \text{ 次高斯 - 像素交互}$$

**(b) Tile-based culling 的有效计算量：**
每个像素平均看到 10 个相关高斯，总像素数 $H \times W = 1024^2 \approx 10^6$。
$$\text{有效计算量} = H \cdot W \cdot (\text{平均每个 pixel 看到的高斯数}) = 10^6 \cdot 10 = 10^7$$

**(c) 速度提升倍数：**
$$\frac{10^{12}}{10^7} = 10^5 = \mathbf{100,000\times}$$

**结论：** Tile-based culling 不是"小优化"，而是让百万级高斯实时渲染成为可能的**结构性改变**。它将复杂度从 $O(HWN)$ 降到了 $O(HW \cdot k)$，其中 $k$ 是每个像素平均看到的相关高斯数（通常远小于总高斯数）。

---
