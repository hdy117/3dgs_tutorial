# 第 10 章：场景一旦动起来，静态 3DGS 的哪一部分先失效——4D Gaussian 到底在扩展什么

**本章核心问题**：前面几章已经把静态 3DGS 的主线解释清楚了。现在问题变成：

> 如果场景本身会随时间变化，那么第 4 章那条 Gaussian -> projection -> sorting -> blending 的渲染链，哪些部分仍然成立，哪些部分必须被改写？4D Gaussian Splatting 到底是在“加一个时间维”，还是在做更深的表示扩展？

如果前面几章回答的是：

- 第 3 章：为什么 primitive 选 Gaussian
- 第 4 章：Gaussian 怎样被渲染成图
- 第 5 章：训练目标为什么这样设计
- 第 6 章：第一批 Gaussian 从哪来
- 第 7 章：静态场景怎样被训练到收敛
- 第 8 章：静态推理为什么还要做优化
- 第 9 章：如果自己实现，应该按什么顺序落代码

那么这一章回答的就是：

```text
如果场景开始动起来
静态 3DGS 为什么不够了
以及应该怎样把静态高斯系统扩展成时间相关的表示
```

先把主线写在前面：

```text
4DGS 的关键不是重新发明一套渲染器
而是把静态高斯参数改成时间函数

canonical Gaussian 集合给你一个静态基底
时间条件形变场告诉你它在每个时刻怎样偏移、拉伸或改变外观

到了每一个具体时刻 t
你仍然是在跑同一条 projection / sorting / blending 链
只是这条链的输入不再是固定参数，而是 Theta(t)
```

也就是说：

> 4DGS 的本质不是“把 3DGS 推翻重来”，而是“让静态高斯表示带上时间依赖，同时尽量保留原有渲染骨架”。

---

## 一、先分清：动态场景到底让静态 3DGS 的哪一条假设失效了

静态 3DGS 之所以成立，背后有一个默认假设：

```text
场景几何和外观是固定的
变化的只有相机位姿
```

所以第 4 章里，我们才可以一直写：

```text
G_i = {mu_i, Sigma_i, alpha_i, sh_i}
```

并默认这些量在整个训练和推理过程中是场景常量。

### 1.1 动态场景里，坏掉的不是投影公式，而是“参数固定”这个前提

如果人开始挥手、车开始移动、布开始飘，那么变化的不是相机而已，而是：

- `mu_i` 可能会随时间变化
- `Sigma_i` 可能会随时间变化
- `alpha_i` 有时也会变化
- 外观参数 `sh_i` 在某些场景里也会变

所以动态场景真正打破的是：

```text
Theta = const
```

而不是：

```text
第 4 章的渲染链失效了
```

这一点特别重要。

### 1.2 也就是说，动态场景的新难点不是“怎么渲染”，而是“渲染什么”

对静态场景，给定相机 `cam_k`，你渲染的是：

```text
I_pred^k = render(Theta, cam_k)
```

对动态场景，同一个相机公式仍然可以用，但现在你真正要渲染的是：

```text
I_pred^t = render(Theta(t), cam_t)
```

这里最关键的新角色是：

```text
Theta(t)
```

也就是：

> 在时刻 `t`，场景里这批 Gaussian 到底处于什么状态。

所以第 10 章要解决的第一个问题不是某条新积分公式，而是：

```text
怎样定义一个时间相关的 Gaussian 场景表示
```

---

## 二、为什么“每帧各训一套静态 3DGS”不是一个好答案

最朴素的动态方案听起来很自然：

```text
视频有多少帧
我就训练多少套静态 3DGS
```

设视频共有 `T` 帧，如果第 `t` 帧对应一套独立高斯参数：

```text
Theta_t = {mu_i^t, Sigma_i^t, alpha_i^t, sh_i^t}_i
```

那么你当然可以对每一帧分别做：

```text
I_pred^t = render(Theta_t, cam_t)
```

表面看起来，这似乎已经能覆盖动态情况。

### 2.1 第一层问题：参数量和训练成本直接按帧数膨胀

如果一帧里有 `N` 个 Gaussian，那么 `T` 帧独立建模就接近：

```text
O(T * N)
```

级别的参数和存储成本。

当视频有几百帧、几千帧时，这会非常重。

### 2.2 第二层问题：相邻帧完全独立，天然缺时序一致性

更严重的问题不是大，而是不连续。

如果每一帧都各自训练：

- 第 100 帧的一个高斯和第 101 帧的一个高斯，不一定有对应关系
- 即使两帧画面都拟合得不错，相邻帧的参数也可能跳来跳去
- 最终容易出现闪烁、漂移和时序不稳定

也就是说：

```text
逐帧独立拟合
解决的是“每一帧像不像”
没有解决“前后帧是不是同一个东西在连续运动”
```

### 2.3 这说明动态问题的关键，不是“有多少帧”，而是“帧与帧之间如何共享结构”

这一步很像第 5 章里“为什么不能只靠图像 loss”。

这里也有一个类似的判断：

> 如果每帧都完全独立，系统当然能去拟合画面，但它没有被迫学会“同一套结构在时间中怎样连续变化”。

所以更合理的思路一定是：

```text
不是每一帧都从零定义一套新高斯
而是让很多时刻共享同一个静态基底
再用某种时间相关机制来描述偏移和变形
```

这就引出了 canonical space 和 deformation field。

---

## 三、4DGS 的第一层核心想法：先有一套 canonical Gaussian，再让它随时间变

很多 4DGS 方法虽然实现细节不同，但最核心的共同思想常常可以压成下面这句：

> 先定义一套 canonical Gaussian 作为“标准形态”，再用时间条件形变场告诉它在每个时刻该怎样变化。

### 3.1 canonical Gaussian 到底是什么

最常见的写法是先定义一组静态基底：

```text
G_i^0 = (mu_i^0, Sigma_i^0, alpha_i^0, sh_i^0)
```

把所有高斯收起来，就是：

```text
Theta^0 = {G_i^0}_i
```

这里上标 `0` 不一定真的代表视频第 0 帧。

它更准确的含义是：

```text
一个参考形态 / canonical space
```

你可以把它理解成：

- 场景在某个参考时刻的样子
- 或者一个更抽象的“标准姿态”
- 后续所有时间的场景，都是从它变过去的

### 3.2 最简单的时间相关写法：只让位置随时间变化

一个最小版本的动态参数化可以写成：

```text
mu_i(t) = mu_i^0 + Delta_mu_i(t)
Sigma_i(t) = Sigma_i^0
alpha_i(t) = alpha_i^0
sh_i(t) = sh_i^0
```

这相当于在说：

- 先只让高斯中心动起来
- 形状、透明度和外观先保持不变

这种最小版本虽然不够表达所有动态，但它非常有价值，因为它先回答了一个最核心的问题：

```text
如果只靠位置形变
这套静态高斯能不能已经追上大量动态场景的主运动
```

### 3.3 更完整的写法：位置、形状、透明度、外观都可以变成时间函数

更通用一点，可以写成：

```text
Theta(t) = {mu_i(t), Sigma_i(t), alpha_i(t), sh_i(t)}_i
```

如果内部参数化采用更稳定的 `scale + rotation + opacity logit` 形式，一个更工程化的版本常常会写成：

```text
s_i(t) = s_i^0 + Delta_s_i(t)
q_i(t) = normalize(q_i^0 + Delta_q_i(t))
o_i(t) = o_i^0 + Delta_o_i(t)
alpha_i(t) = sigmoid(o_i(t))
sh_i(t) = sh_i^0 + Delta_sh_i(t)
```

于是协方差再由：

```text
Sigma_i(t) = R(q_i(t)) * diag(s_i(t)^2) * R(q_i(t))^T
```

恢复出来。

### 3.4 这里最该记住的，不是所有量都必须变，而是：哪些量值得变，要看场景和稳定性

很多工程实现不会把所有项都放开，因为：

- 放开越多，自由度越强
- 但训练也会更难、更不稳
- 闪烁和漂移更容易出现

所以常见实践往往是：

- 先让 `mu(t)` 变
- 再视情况决定要不要让 `Sigma(t)` 也变
- `alpha(t)` 和 `sh(t)` 常常会更谨慎地处理

因为动态系统里，一个很大的工程挑战恰恰是：

```text
自由度越多
你越容易把短期拟合做漂亮
但越容易牺牲时间稳定性
```

---

## 四、4DGS 并没有推翻第 4 章：每个时刻仍然跑同一条投影与 blending 链

这一步是理解 4DGS 的真正桥梁。

很多人会误以为：

```text
动态高斯 = 一套完全新的渲染理论
```

其实很多时候并不是。

### 4.1 在每个固定时刻 `t`，你仍然是在做静态渲染

一旦 `Theta(t)` 已经给定，对该时刻的场景来说，你还是可以直接沿用第 4 章：

```text
mu_cam(t) = R_t * mu(t) + t_t
Sigma_cam(t) = R_t * Sigma(t) * R_t^T
```

然后中心投影：

```text
u(t) = fx * X(t) / Z(t) + cx
v(t) = fy * Y(t) / Z(t) + cy
```

再做 Jacobian 线性化：

```text
J_t = [[fx / Z(t), 0, -fx * X(t) / Z(t)^2],
       [0, fy / Z(t), -fy * Y(t) / Z(t)^2]]
```

于是：

```text
Sigma_2d(t) ≈ J_t * Sigma_cam(t) * J_t^T
```

### 4.2 后面的 blending 也没有消失

对于像素 `p`，你仍然可以写：

```text
w_i^t(p) = alpha_i(t) * g_i^t(p)
```

其中：

```text
g_i^t(p) = exp(-1/2 * (p - mu_2d^i(t))^T * Sigma_2d^i(t)^(-1) * (p - mu_2d^i(t)))
```

最后照样做 front-to-back compositing：

```text
T_1^t(p) = 1
C_t(p) = sum_i T_i^t(p) * w_i^t(p) * c_i^t
T_{i+1}^t(p) = T_i^t(p) * (1 - w_i^t(p))
```

### 4.3 所以 4DGS 最准确的工程理解不是“新渲染器”，而是“时间条件场景参数”

可以把它压成一句话：

> 第 4 章讲的是“给定一套 Gaussian 参数，怎样出图”；第 10 章讲的是“在动态场景里，这套参数本身怎样随着时间被生成出来”。

这也是为什么第 10 章一定要接在第 4 章之后。

因为动态扩展真正复用的，恰恰是前面已经建立好的静态渲染骨架。

---

## 五、时间相关参数到底由什么来产生：变形场为什么会成为主角

既然核心问题已经压成：

```text
Theta(t) 如何得到
```

下一步自然就是：

```text
Delta_mu_i(t), Delta_s_i(t), Delta_q_i(t) 这些偏移从哪来
```

这里最自然的思路就是 deformation field。

### 5.1 最常见的抽象：一个时间条件形变场

可以先用最一般的写法表示：

```text
F_phi(x, t) = x + D_phi(x, t)
```

其中：

- `x` 是 canonical space 里的位置
- `t` 是时间
- `D_phi` 是一个由参数 `phi` 控制的偏移场

于是 Gaussian 中心就可以写成：

```text
mu_i(t) = F_phi(mu_i^0, t)
        = mu_i^0 + D_phi(mu_i^0, t)
```

这条式子特别重要，因为它直接说明：

> 动态场景不一定要为每个高斯、每一帧都单独存一份参数；也可以只学一个连续时间函数，查询时再给出该时刻的偏移。

### 5.2 为什么这个函数形式比“逐帧参数表”更自然

因为视频不是一堆互不相干的图片。

大多数真实运动都具有：

- 时间连续性
- 局部平滑性
- 大量重复模式

所以用函数去拟合：

```text
位置随时间怎样变化
```

往往比把每一帧硬存成独立查找表更合理。

### 5.3 常见实现里，这个变形场为什么常常由一个小网络来表达

因为 `D_phi(x, t)` 一般很难手工写出解析式。

例如：

- 挥手
- 布料摆动
- 脸部表情变化

这些都不是简单的一条直线运动。

所以比较常见的实现会让：

```text
D_phi(x, t)
```

由一个时间条件网络来拟合。

但这里一定要记住：

> 这个网络不是最终输出整张图，它只是负责告诉高斯“你此刻应该往哪儿去，或怎样变形”。

真正的成像仍然交给第 4 章那条 renderer。

---

## 六、训练目标不再只是“每帧像不像”，还要额外管住时间稳定性

到了这一步，你已经能写出动态场景最基础的 forward：

```text
Theta(t) -> render(Theta(t), cam_t) -> I_pred^t
```

于是最直接的图像项当然是：

```text
L_photo = sum_t L_img(render(Theta(t), cam_t), I_gt^t)
```

其中 `L_img` 仍然可以沿用第 5 章：

```text
L_img = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
```

### 6.1 但只靠逐帧图像项，动态系统很容易学出“看起来能过、时间上很抖”的结果

这和静态场景的差别就在这里。

静态系统只要回答：

```text
当前这一张图像对不对
```

动态系统还要回答：

```text
前后时刻是不是同一个结构在连续变化
```

否则就容易出现：

- 相邻帧轻微闪烁
- 背景被无意义地拉着动
- 局部区域在时间上抖动
- 某些高斯为了贴合单帧误差，走出不自然路径

### 6.2 所以时间平滑正则几乎一定会出现

一个最直观的时间平滑项可以写成：

```text
L_temp = sum_{i,t} ||mu_i(t + Delta t) - 2 * mu_i(t) + mu_i(t - Delta t)||^2
```

它本质上在惩罚什么？

```text
二阶时间差分过大
```

也就是：

> 不要让一条轨迹在时间上突然出现很尖的折点和高频抖动。

### 6.3 光有时间平滑还不够，很多系统还会加 deformation magnitude regularization

因为如果 deformation field 过于自由，它可能会学出“哪里不对就大幅度乱挪”的策略。

所以常见还会加：

```text
L_def = sum_{i,t} ||Delta_mu_i(t)||^2
```

或者类似项，表达的是：

> 如果没有必要，不要离 canonical pose 偏得太远。

### 6.4 还可能会出现局部刚性 / 邻域一致性正则

很多动态物体局部并不是完全自由形变。

例如一只手掌内部、一个人的前臂、车门上的一块金属板，局部关系往往相对稳定。

于是可以引入邻域保持项，例如：

```text
L_local = sum_{(i,j) in N} sum_t ||(mu_i(t) - mu_j(t)) - (mu_i^0 - mu_j^0)||^2
```

它的含义是：

> 相近高斯之间的相对结构，不要无缘无故被撕裂。

### 6.5 所以动态训练目标更完整地可以写成

```text
L_total = L_photo + lambda_temp * L_temp + lambda_def * L_def + lambda_local * L_local
```

这里最重要的思想不是具体常数，而是要理解分工：

- `L_photo`：保证每一帧图像像
- `L_temp`：保证时间上不要抖
- `L_def`：避免形变场乱跑
- `L_local`：避免局部结构被撕裂或漂散

也就是说：

> 第 5 章里“图像监督 + 结构规则”的思想，到第 10 章并没有消失，只是多加了一层“时间结构管理”。

---

## 七、为什么 4DGS 的最大工程难点，常常不是 render，而是自由度管理

到了这里，你可能会觉得：

```text
那就把所有量都变成时间函数
表达能力不是最强吗
```

听起来很合理，但实际工程上很危险。

### 7.1 自由度太少，会欠拟合运动

如果你只允许：

```text
mu(t) 变化
Sigma, alpha, sh 全固定
```

那对一些刚体平移或温和运动也许够用。

但一旦碰到：

- 布料褶皱
- 手指张开
- 脸部表情
- 运动模糊明显的结构变化

就容易不够。

### 7.2 自由度太多，又容易学出闪烁和漂移

如果你让：

- `mu(t)` 变
- `Sigma(t)` 变
- `alpha(t)` 变
- `sh(t)` 也大幅度变

那系统就很容易走向另一边：

- 单帧很会贴图
- 但相邻帧不稳定
- 某些背景也跟着轻微呼吸
- 外观看起来像在闪

### 7.3 所以 4DGS 最核心的工程判断，不是“能不能让更多参数动”，而是“哪些参数值得动，动多少才值”

这和第 8 章的推理优化非常像。

都不是盲目堆自由度或技巧，而是：

```text
先看瓶颈在哪
再决定放开什么
```

对动态场景来说，最典型的问题往往是：

- 主要矛盾是位置运动没跟上？
- 还是局部形状本身也必须变？
- 还是外观变化（比如镜面、高光、非朗伯）才是主问题？

不同场景的答案会不同。

---

## 八、几个典型动态场景：同一套 4DGS 思路为什么表现会差很多

### 8.1 刚体主导场景：开门、旋转物体、移动车辆

这类场景最显著的特点是：

- 大部分结构保持刚性
- 运动模式比较规则
- `mu(t)` 的变换更像整体 SE(3)

这种情况下，动态高斯问题相对“简单”。

因为你可以用很少的时间自由度，就描述出主要变化。

### 8.2 轻度非刚体场景：挥手、人体关节运动、面部表情

这类场景常见特点是：

- 大体结构仍然有可追踪对应
- 局部会弯折、旋转、拉伸
- 相邻区域运动模式相近但不完全一样

这正是很多 4DGS 方法最擅长的工作区间。

因为：

```text
canonical space + 局部平滑 deformation field
```

对这类运动很自然。

### 8.3 强拓扑变化场景：撕纸、液体飞溅、烟雾爆开

这类场景会变得更难，因为问题不再只是：

```text
同一个结构怎样移动
```

而更像：

```text
结构本身在分裂、生成、消失
```

这时固定一套 canonical Gaussian 再做连续变形，就会遇到明显边界。

所以你要特别记住：

> 4DGS 很擅长“可连续追踪的动态结构”，但不一定天然擅长强拓扑变化。

这不是某个实现小缺陷，而是表示假设本身的边界。

---

## 九、动态系统最常见的四种失败模式

第 7 章讲的是静态训练时的失败模式。

到了动态场景，问题会多出一层“时间维度的坏掉方式”。

### 9.1 症状一：整体看着能拟合，但视频在闪

常见原因：

- `sh(t)` 或 `alpha(t)` 变化太自由
- 时间正则太弱
- 相邻帧监督不够稳定

这类问题本质上是在说：

```text
系统在逐帧取巧
却没有学出稳定时间轨迹
```

### 9.2 症状二：背景也在轻微呼吸或漂动

常见原因：

- deformation field 对全场统一施加位移
- 缺少静态背景约束
- canonical 初始化本身就不稳

这说明问题常常不在 renderer，而在：

```text
形变场把本不该动的高斯也带着一起动了
```

### 9.3 症状三：运动被学得过于平滑，细节动作丢失

常见原因：

- `lambda_temp` 太大
- `lambda_def` 太大
- deformation network 容量不足

这类问题说明：

```text
系统不是不会稳定
而是稳定得太狠
把该保留的动态细节也压平了
```

### 9.4 症状四：局部结构被撕裂、拉长、断开

常见原因：

- `Sigma(t)` 和 `mu(t)` 更新不协调
- 缺少邻域一致性约束
- 运动太复杂，超出当前表示能力

这提醒你：

> 动态 3DGS 的困难不只是“会不会动”，而是“动的时候还能不能维持局部几何组织”。

---

## 十、一个最小可运行实验：把 canonical Gaussian 变成随时间运动的 2D 场景

下面这段代码不跑真正的 4DGS，它只做一件特别有价值的事：

> 用一个二维 toy 例子，把“canonical Gaussian + 时间条件位移 + 同一条 blending 渲染链”这条核心直觉画出来。

这段代码里：

- 每个高斯先有一个 canonical 位置 `mu0`
- 再由一个简单的时间函数产生位移
- 每个时刻都沿用第 4 章那套 2D footprint + blending
- 最后同时画出若干时间切片和运动轨迹

```python
import numpy as np
import matplotlib.pyplot as plt


H, W = 220, 220
xs = np.linspace(0, W - 1, W)
ys = np.linspace(0, H - 1, H)
X, Y = np.meshgrid(xs, ys)


canonical_gaussians = [
    {
        'mu0': np.array([72.0, 112.0]),
        'Sigma': np.array([[260.0, 40.0], [40.0, 150.0]]),
        'alpha': 0.65,
        'color': np.array([0.95, 0.35, 0.20]),
        'phase': 0.0,
    },
    {
        'mu0': np.array([118.0, 100.0]),
        'Sigma': np.array([[220.0, -60.0], [-60.0, 130.0]]),
        'alpha': 0.62,
        'color': np.array([0.25, 0.75, 1.00]),
        'phase': 0.8,
    },
    {
        'mu0': np.array([160.0, 126.0]),
        'Sigma': np.array([[180.0, 20.0], [20.0, 110.0]]),
        'alpha': 0.58,
        'color': np.array([0.96, 0.88, 0.22]),
        'phase': 1.6,
    },
]


def gaussian_map(mu, Sigma):
    pos = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    inv = np.linalg.inv(Sigma)
    q = np.einsum('...i,ij,...j->...', pos, inv, pos)
    return np.exp(-0.5 * q)


def deform(mu0, t, phase):
    dx = 18.0 * np.sin(2.0 * np.pi * t + phase)
    dy = 10.0 * np.sin(4.0 * np.pi * t + 0.5 * phase)
    return mu0 + np.array([dx, dy])


def render_frame(t):
    C = np.zeros((H, W, 3), dtype=np.float64)
    T = np.ones((H, W, 1), dtype=np.float64)
    centers = []

    for g in canonical_gaussians:
        mu_t = deform(g['mu0'], t, g['phase'])
        footprint = gaussian_map(mu_t, g['Sigma'])[..., None]
        w = g['alpha'] * footprint

        C += T * w * g['color']
        T *= (1.0 - w)
        centers.append(mu_t)

    background = np.ones((H, W, 3), dtype=np.float64)
    C += T * background
    return np.clip(C, 0.0, 1.0), np.array(centers)


times = [0.0, 0.25, 0.5, 0.75]
trajectory_times = np.linspace(0.0, 1.0, 120)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for idx, t in enumerate(times):
    frame, centers = render_frame(t)
    axes[0, idx].imshow(frame)
    axes[0, idx].scatter(centers[:, 0], centers[:, 1], c='black', s=14)
    axes[0, idx].set_title(f't = {t:.2f}')
    axes[0, idx].axis('off')

for g_idx, g in enumerate(canonical_gaussians):
    traj = np.stack([deform(g['mu0'], t, g['phase']) for t in trajectory_times], axis=0)
    axes[1, 0].plot(traj[:, 0], traj[:, 1], linewidth=2)
    axes[1, 0].scatter(g['mu0'][0], g['mu0'][1], s=18)

axes[1, 0].set_xlim(0, W)
axes[1, 0].set_ylim(H, 0)
axes[1, 0].set_aspect('equal')
axes[1, 0].set_title('canonical to trajectory')
axes[1, 0].set_xlabel('u')
axes[1, 0].set_ylabel('v')

for idx, t in enumerate(times[1:], start=1):
    centers_t = np.stack([deform(g['mu0'], t, g['phase']) for g in canonical_gaussians], axis=0)
    centers_0 = np.stack([g['mu0'] for g in canonical_gaussians], axis=0)
    delta = np.linalg.norm(centers_t - centers_0, axis=1)
    axes[1, idx].bar(np.arange(len(delta)), delta)
    axes[1, idx].set_title(f'|delta mu| at t={t:.2f}')
    axes[1, idx].set_xlabel('gaussian id')
    axes[1, idx].set_ylabel('magnitude')

plt.tight_layout()
plt.show()
```

你应该观察到：

- 上排仍然是同一套 blending 逻辑在出图，只是中心位置随时间变化
- 下排左图把 canonical space 和时间轨迹联系起来了
- 下排右边几张柱状图展示了不同时间下的形变量大小

这段实验最重要的作用不是给你真实视频重建，而是帮你建立一个极其重要的直觉：

```text
4DGS 不是把 renderer 换掉
而是让 renderer 的输入由静态参数变成时间条件参数
```

---

## 十一、为什么很多 4DGS 实现看起来像“形变网络 + 静态渲染器”的拼接

如果你把这章退远一点看，会发现很多 4DGS 系统都像下面这个组合：

```text
一个时间条件模块
+
一个静态 3DGS renderer
```

这不是偶然，而是非常自然的工程分层。

### 11.1 renderer 负责“给定状态，怎样出图”

这部分已经在静态场景里被证明有效：

- projection
- sorting
- tile culling
- blending

### 11.2 deformation module 负责“此刻状态是什么”

它不直接生成像素，而是生成：

- `mu(t)`
- `Sigma(t)`
- `alpha(t)`
- `sh(t)`

### 11.3 这种分层为什么值钱

因为它允许你：

- 复用静态 3DGS 的大量实现和优化
- 单独分析“渲染错了”还是“形变错了”
- 把动态问题拆成两个可诊断子系统

这和第 9 章强调的实现顺序完全一致。

也就是说，第 10 章并不只是理论升级，也是在告诉你：

> 动态扩展最稳的做法，往往不是从零造一个“全动态黑箱”，而是在静态系统上增加一个时间条件层。

---

## 十二、把整章压成一个最短心智模型

如果你只想记一条链，就记这个：

```text
静态 3DGS 的核心假设是：
场景参数 Theta 固定
只随相机变化

    ↓

动态场景打破的不是投影和 blending 公式
而是“参数固定”这个前提

    ↓

所以 4DGS 的关键不是重写渲染器
而是把 Gaussian 参数改成时间函数：
Theta(t)

    ↓

最常见思路是：
先定义 canonical Gaussian 集合 Theta^0
再用时间条件 deformation field 生成
mu(t), Sigma(t), alpha(t), sh(t)

    ↓

每个固定时刻 t
仍然沿用第 4 章那条静态渲染链：
projection -> sorting -> tile -> blending

    ↓

训练时除了逐帧图像监督
还要额外加入时间平滑、形变幅度、局部结构保持等约束

    ↓

4DGS 真正解决的是：
如何让“同一套结构”在时间里连续变化
而不是把每一帧都当成互不相关的新场景
```

这就是第 10 章真正想让你建立起来的动态高斯视角。

---

## 十三、本章你真正应该能自己重建的几个问题

读完以后，遮住正文，你至少应该能自己回答：

1. 动态场景真正打破的是静态 3DGS 的哪一条假设？
2. 为什么“每帧各训一套静态 3DGS”既昂贵又缺乏时序一致性？
3. canonical Gaussian 和 deformation field 分别在系统里承担什么角色？
4. 为什么很多 4DGS 方法在每个固定时刻仍然沿用第 4 章那条投影与 blending 链？
5. 为什么动态训练不能只靠逐帧图像项，还需要时间平滑和形变正则？
6. 为什么让更多参数随时间变化，不一定就等于更好的动态建模？
7. 哪些场景更适合 canonical + continuous deformation 的假设，哪些场景会逼近它的边界？
8. 如果视频出现闪烁、背景呼吸、运动被压平、局部结构撕裂，你应该首先怀疑动态系统的哪一层？

如果这些问题你能自己从头讲回来，这一章就真的进入你的脑子了。

---

## 十四、下一章接什么

现在你已经知道：

- 静态 3DGS 可以怎样被扩展到动态场景
- 4DGS 的核心不是重写 renderer，而是让场景参数变成时间函数
- 为什么 canonical space、deformation field 和时间正则会成为动态系统的主角

下一章 [chapter_11_feedforward_gaussian.md](chapter_11_feedforward_gaussian.md) 会自然接到另一个更大的问题：

> 即使静态 3DGS 和 4DGS 都已经成立，它们大多仍然依赖 per-scene optimization。那如果我们不想每来一个新场景都优化很久，而是想让系统一次前向就直接预测 Gaussian，可能吗？

也就是从：

```text
“场景会动时，Gaussian 系统该怎样加上时间”
```

走到：

```text
“如果连长时间优化都不想做，Gaussian 能不能被直接预测出来”
```
