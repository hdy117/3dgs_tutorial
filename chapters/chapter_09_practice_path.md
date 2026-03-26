# 第 9 章：如果你要自己实现 3DGS，第一步该先写什么，怎么验证，卡住时该看哪里

**本章核心问题**：前面几章已经分别解释了：

- 第 3 章：为什么 primitive 选 Gaussian
- 第 4 章：Gaussian 怎样被渲染成图
- 第 5 章：优化目标为什么这样设计
- 第 6 章：第一批 Gaussian 从哪来
- 第 7 章：训练闭环怎样工作
- 第 8 章：为什么推理还需要专门优化

现在问题变成：

> 如果你不想只停留在“我看懂了原理”，而是想自己从零搭一版能工作的 3DGS，应该按什么顺序把这些知识落成代码？为什么实现顺序本身，就是工程成败的一部分？

如果前面几章解决的是：

```text
为什么它能表示
为什么它能渲染
为什么它能训练
为什么它还能跑快
```

那么这一章解决的就是：

```text
如果我要自己实现
第一步该先写什么
每一步该怎么验
出问题时先怀疑哪一层
```

先把主线写在前面：

```text
真正好的实现路径
不是按官方仓库的文件夹顺序抄
也不是一上来就把训练、densify、CUDA 全堆上去

而是按依赖关系一层层搭：
先做一个能看见的最小前向
再做一个能下降的最小训练
再做结构编辑
最后再做加速和变体
```

这一章本质上是在做一件事：

> 把第 3 章到第 8 章的“原理地图”，压成一条真正可执行的实现路径。

---

## 一、为什么实现顺序本身就是工程问题

很多人第一次自己做 3DGS 时，最容易犯的错不是“某条公式不会推”，而是：

```text
把太多层同时写出来
于是所有 bug 一起爆
最后根本不知道该从哪一层开始查
```

比如你如果一开始就把下面这些全部堆上去：

- COLMAP 数据解析
- 3D -> 2D 投影
- front-to-back blending
- L1 + SSIM
- densify / split / prune
- tile mapping
- mixed precision
- CUDA kernel

那你一旦看到：

- 图像全黑
- loss 不降
- `NaN`
- 高斯数量爆炸
- 帧率还很慢

你几乎没法第一时间判断，到底是哪一层先出错。

这就是为什么实现顺序不是“个人习惯”，而是一个真正的工程决策。

### 1.1 这件事很像搭一座桥，不像拼乐高

乐高错一块，通常只影响局部。

但 3DGS 更像搭桥：

- 基础坐标系一旦错，后面投影全错
- 投影一旦错，render 再快也只是更快地产生错误图像
- render 一旦不对，loss 再漂亮也只是把错误信号往回传
- 训练闭环一旦没站住，densify 就会把错误结构放大

所以正确顺序不是：

```text
功能越全越好
```

而是：

```text
每一层都先做成一个可验证的小闭环
再往上叠下一层
```

### 1.2 这也是为什么“先跑通官方代码”不等于“我会实现”

跑通官方代码当然有价值，因为它能让你看到：

- 数据长什么样
- 最终效果大概是什么级别
- 常见训练超参数怎么设

但它给你的更多是：

```text
黑盒使用经验
```

而不是：

```text
如果我要自己写
我知道先搭哪块骨架
以及这块骨架该怎么验
```

第 9 章的任务，就是把这件事讲清楚。

---

## 二、先把整条实现路径压成一页图

如果把整套实现顺序压成最短版本，可以先记这条链：

```text
Milestone A
先做 2D Gaussian footprint + alpha blending 的最小沙盒

    ↓

Milestone B
把 3D Gaussian 接上相机投影
确认 mu_2d / Sigma_2d 真能落到图像上

    ↓

Milestone C
做最小可微训练闭环
先只用简单 loss，让图像真的开始变好

    ↓

Milestone D
接真实数据和 SfM 初始化
让系统进入多视图训练状态

    ↓

Milestone E
再加 densify / split / prune
让表示容量开始动态重分配

    ↓

Milestone F
最后才做 tile、排序缓存、mixed precision、kernel fusion
把“能工作”推进到“能高效工作”
```

这条顺序背后的逻辑非常简单：

> 永远先解决“有没有对”，再解决“学不学得动”，最后才解决“跑得快不快”。

很多实现失败，问题不是努力不够，而是一开始就把顺序反过来了。

---

## 三、把前面六章的公式，映射成真正要写的模块

如果你现在准备开一个自己的实现仓库，最有用的不是先抄官方目录，而是先把“章节概念 -> 代码模块 -> 第一张检查图”对应起来。

| 你要解决的问题 | 核心公式 / 概念 | 建议模块 | 第一个验证结果 |
|---|---|---|---|
| 一个 Gaussian 到底存什么 | `G_i = {mu_i, Sigma_i, alpha_i, sh_i}` | `scene/gaussians.py` | 参数 shape 和统计量正常 |
| 世界坐标怎样进相机 | `mu_cam = R * mu + t` | `render/project.py` | `z > 0` 的点在相机前方 |
| 3D 椭球怎样变成 2D footprint | `Sigma_cam = R * Sigma * R^T`，`Sigma_2d ≈ J * Sigma_cam * J^T` | `render/project.py` | 投影椭圆覆盖在物体轮廓附近 |
| 像素怎样被多个高斯混成颜色 | `C(p) = sum_i T_i(p) * w_i(p) * c_i` | `render/blend.py` | 单帧渲染不是全黑也不是全白 |
| 系统怎样开始学 | `L_img = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)` | `train/losses.py` | loss 明显下降 |
| 第一批高斯从哪来 | SfM 点云 + 初始化启发式 | `data/colmap_loader.py`、`scene/init_from_sfm.py` | 初始 render 有大致轮廓 |
| 表示容量怎样动态调整 | densify / split / prune | `train/density_control.py` | `N` 曲线先升后稳 |
| 为什么最终还得做推理优化 | tile / sorting / bandwidth | `render/tile.py`、`tools/profile.py` | 帧时间下降且输出几乎不变 |

你会发现，第 9 章其实没有引入新数学。

它做的是另一种更重要的事：

> 把前面的数学，重新按“实现依赖关系”排序。

---

## 四、第一步为什么不是读 COLMAP，也不是写训练，而是先做一个 2D 沙盒

这一步特别重要，也特别容易被跳过。

很多人觉得：

```text
3DGS 是 3D 问题
那当然要先写 3D
```

但真正更稳的起点常常是：

```text
先在 2D 里把 footprint 和 blending 这条最小成像链站住
```

### 4.1 为什么这一步值钱

因为它把问题规模瞬间缩小了。

你暂时不需要考虑：

- 相机外参到底是 world-to-camera 还是 camera-to-world
- COLMAP 坐标系要不要翻轴
- `J * Sigma_cam * J^T` 有没有数值问题
- 多视图 loss 是不是在打架

你只需要确认一件事：

> 如果已经给你一个 `mu_2d`、一个 `Sigma_2d`、一个 `alpha` 和一个颜色，你能不能把它们稳定地混成一张图。

### 4.2 这一层你真正该写的只有三件事

```text
q(p) = (p - mu_2d)^T * Sigma_2d^(-1) * (p - mu_2d)

g(p) = exp(-1/2 * q(p))

w(p) = alpha * g(p)
```

再加上第 4 章那条 front-to-back blending：

```text
T_1(p) = 1
C(p) = sum_i T_i(p) * w_i(p) * c_i
T_{i+1}(p) = T_i(p) * (1 - w_i(p))
```

这一步做完后，你至少应该能验证两件事：

- 调整高斯顺序，叠放关系真的会变
- 把一个椭圆拉细、旋转，图像局部贡献真的会跟着变

### 4.3 为什么这一层比你想的更关键

因为如果连这一层都没站住，后面很多“3D 问题”其实都没资格讨论。

举个例子：

- 如果你把颜色简单相加，而不是做剩余透射率递推
- 或者你把 `Sigma_2d` 反了，导致椭圆主轴错位

那你后面即使把相机投影、SfM 初始化、训练循环全部接上，系统也只是在更复杂地犯同一个错。

所以第一个里程碑不是：

```text
我已经能训练了
```

而是：

```text
我已经能稳定地把几个 2D Gaussian 混成正确图像了
```

这一步是整套实现真正的地基。

---

## 五、第二步：把 2D 沙盒升级成 3D -> 2D 投影链

当地基站住后，才轮到真正接上第 4 章那条投影主链。

这一层的关键不是“一上来就把整张图 render 出来”，而是先确认：

> 3D 世界里的高斯，真的被投到了正确的 2D 位置和正确的 2D 椭圆上。

### 5.1 这一层最核心的三条公式

先是中心变换：

```text
mu_cam = R * mu_world + t
```

再是协方差旋转：

```text
Sigma_cam = R * Sigma_world * R^T
```

然后是透视投影与局部线性化：

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy

J = [[fx / Z, 0, -fx * X / Z^2],
     [0, fy / Z, -fy * Y / Z^2]]

Sigma_2d ≈ J * Sigma_cam * J^T
```

### 5.2 这一层最值得先验的，不是最终 render，而是投影调试图

很多人一接上 3D，就急着看“最终画面像不像”。

但更高收益的第一张图其实是：

```text
把所有高斯中心投到图像平面上
再把若干 Sigma_2d 对应的椭圆画出来
```

因为这一张图可以一次暴露很多根问题：

- 整体偏出画面：坐标系或外参方向错
- 上下翻转 / 左右镜像：轴约定错
- 椭圆离谱地大：`Z` 太小、scale 太大或 `J` 算错
- 椭圆接近退化：`Sigma_2d` 正定性没保住

### 5.3 这一层最常见的四个坑

#### 5.3.1 world-to-camera 和 camera-to-world 混了

这会导致你公式写得很像对的，但图永远不落对位置。

#### 5.3.2 `Z` 接近 0 或者在相机后方

于是：

```text
fx / Z
```

会直接爆掉。

所以工程上一定会做：

```text
Z = clamp(Z, min=eps)
```

并且把明显不在可见范围内的高斯先筛掉。

#### 5.3.3 `Sigma_2d` 数值上接近奇异

理论上这条链是可微的，但数值上不一定稳，所以通常要加：

```text
Sigma_2d = Sigma_2d + eps * I
```

#### 5.3.4 你把“投影椭圆对不对”这个问题，误判成“训练为什么不收敛”

这是实现初期非常常见的错位。

其实在这一步，训练都还没上场。你只是在确认：

```text
几何主链有没有成立
```

如果这一层没对，继续往训练方向查，只会越查越偏。

---

## 六、第三步：先做一个最小可微训练闭环，不要急着上完整配置

当前向渲染链已经能稳定工作后，才该把系统推进到“会学”。

但这里也有一个很重要的顺序判断：

> 一开始不要同时上真实多视图、复杂 loss、densify、缓存和优化技巧。

先让系统证明一件最朴素的事：

```text
如果我固定一小批高斯
图像误差确实会把它们往更好的方向推
```

### 6.1 最小训练闭环到底长什么样

你真正需要的骨架只有：

```python
rendered = render(gaussians, camera)
loss = l1_loss(rendered, gt_image)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这一版甚至不急着上 `SSIM`。

因为此时最关键的不是“达到最佳画质”，而是回答：

- 梯度有没有真的流回 `mu`、`Sigma`、`alpha`、颜色参数
- 图像有没有朝着 GT 方向改进
- 你的 render 路径和 backward 路径是不是一致的

### 6.2 为什么这一步更适合先用简单 loss

第 5 章已经说过，更完整的图像项常常会写成：

```text
L_img = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
```

但在实现早期，先只用 `L1` 的好处很大：

- 更容易判断数值有没有往正确方向动
- 更容易解释 loss 变化
- 出问题时排查面更小

不是说 `SSIM` 不重要，而是：

> 先让系统会走，再让它走得更漂亮。

### 6.3 这一层通过时，你应该看到什么

如果这一步成立，通常会看到：

- loss 稳定下降
- 渲染图从乱到有轮廓
- `mu.grad`、`Sigma.grad`、`alpha.grad` 不是长期为 0
- 不同参数组对图像的影响方向符合直觉

例如：

- 调 `mu` 会移动局部结构位置
- 调 `Sigma` 会改变模糊宽度与方向
- 调 `alpha` 会改变遮挡和透射
- 调颜色会直接改变外观

也就是说，这一层真正要确认的是：

```text
公式已经不只是能前向算图
而是能被 loss 驱动着改图
```

---

## 七、第四步：再把真实数据、SfM 初始化和多视图训练接上来

到了这里，才值得把第 6 章和第 7 章那条真正的训练闭环接起来。

这一步的任务不是重新发明数学，而是把前面的“toy 可学系统”推进成“真实静态场景训练系统”。

### 7.1 这一层新增的不是渲染本身，而是冷启动能力

你在第 6 章已经知道，随机撒高斯通常不靠谱。

所以真实系统常见的起点会是：

```text
mu_i = x_i^sfm
color_i = rgb_i / 255
scale_i 来自重投影误差、近邻距离或混合启发式
alpha_i 取中等透明度起点
```

如果内部参数化更贴近实现，还会是：

```text
scale_i = log([s_i, s_i, s_i])
rotation_i = identity
opacity_logit_i = log(alpha0 / (1 - alpha0))
sh_i[0] = rgb_i / 255
sh_i[1:] = 0
```

### 7.2 这一层最该先验的不是最终 PSNR，而是“进入可训练区间了没有”

你应该先看下面这些东西：

- 初始 render 不是全黑
- 初始 render 不是全白
- 投影中心大致覆盖物体区域
- scale 直方图没有极端大尾巴
- `alpha` 分布没有整体贴近 0 或 1

如果这些没站住，后面 PSNR 再差，也不一定是训练策略问题。

更可能是：

```text
初始化根本没把系统送进可训练区间
```

### 7.3 真正接上多视图训练后，你最该盯哪几条曲线

第 7 章已经给过答案，这里把它们重新放回实现视角：

- `L1 / L_img / PSNR`：看系统有没有整体变好
- `N`：看结构是否在膨胀、收敛还是失控
- `alpha` 分布：看系统是不是在堆一堆死高斯
- `scale` 分布：看结构是在细化还是在跑飞

这一层真正的标志不是“论文指标已经追平”，而是：

> 你终于拿到了一条能稳定训练真实场景的静态 3DGS 主链。

---

## 八、第五步：为什么 densify / split / prune 一定要晚于最小训练闭环

这一点非常值得单独强调。

很多人实现时最着急的就是 densify，因为它看起来最像 3DGS 的“灵魂”。

但从工程顺序上看，它恰好不该最早上。

### 8.1 为什么不能一开始就上 densify

因为 densify 不是独立模块。

它依赖前面已经成立的很多东西：

- render 是对的
- loss 是能降的
- 梯度方向大致可信
- footprint / radii 统计是可信的
- optimizer 状态管理是稳定的

如果这些前提还没站住，你一旦加上 densify，就会看到：

- `N` 疯狂增长
- 新增高斯位置乱飞
- 显存暴涨
- loss 反而更不稳

然后你会误以为“densify 策略错了”。

但很多时候，真正错的是：

```text
前面的主链还没稳
densify 只是把错误结构更快放大了
```

### 8.2 更合理的接入顺序是什么

你应该先让下面这件事成立：

```text
不做 densify 的固定高斯系统
已经能稳定下降并得到粗糙但正确的结果
```

然后再接入结构编辑：

- 梯度大 + footprint 不大 -> clone
- 梯度大 + footprint 很大 -> split
- 长期几乎没贡献 -> prune

这时 densify 才会真的在做它该做的事：

> 把已经成立的训练闭环，进一步推进成“容量会自适应”的训练闭环。

### 8.3 这一层真正该看的不是单次 loss，而是结构曲线

当 densify / prune 接上后，你最有用的观测图通常是：

- `N vs step`
- `radii` 分布
- `alpha` 分位数
- 新增与删除高斯的数量

更理想的图景通常是：

```text
前期 N 上升
中期继续细化
后期增速变慢并逐渐稳定
```

如果你看到的是：

```text
N 从头到尾无脑爆炸
```

那说明系统不是在“变强”，而是在“失控扩容”。

---

## 九、第六步：为什么性能优化必须建立在“慢速参考版”之上

到了这一步，你才真正来到第 8 章的世界。

也就是：

```text
系统已经会表示、会渲染、会训练了
现在开始问：怎样让它更快
```

### 9.1 为什么一定要保留一个慢速参考版

因为优化最容易引入一种特别危险的幻觉：

```text
速度上去了
所以我觉得自己写对了
```

但推理优化里很多动作都会改变计算组织方式：

- tile mapping
- 局部排序
- early stop
- mixed precision
- kernel fusion
- 缓存复用

这些动作未必会显式报错。

它们更常见的失败方式是：

- 图像 subtly 变了
- 局部闪烁增加了
- 边缘顺序错一点点
- 某些视角下才露出问题

所以工程上一个特别重要的原则就是：

> 每做一次加速，都拿慢速参考版做逐项对照。

### 9.2 这一层最值得引入的第一个优化通常是什么

通常不是 mixed precision，也不是 fusion，而是：

```text
tile-based culling / tile mapping
```

因为它最直接地利用了 Gaussian 的局部性，而且不需要先改掉整条数学链。

这一层一旦成立，你再继续往下接：

- 排序缓存
- tile cache
- mixed precision
- kernel fusion

就会更稳，也更容易测出每一步到底省了什么。

### 9.3 这一层真正的验收标准是什么

不是单看帧率。

而是同时满足两件事：

```text
输出几乎不变
帧时间显著下降
```

也就是说，你应该同时记录：

- `frame time`
- `max abs diff`
- `PSNR(render_fast, render_ref)`
- 如果有连续视角，还要看闪烁和稳定性

第 8 章已经告诉你优化在打哪些瓶颈。

第 9 章则是在说：

> 这些优化应该什么时候接进来，以及每接一次，拿什么当验收依据。

---

## 十、一个特别实用的实现习惯：每一层都配一张“检查图”

如果只靠 print 和 loss，你会很慢。

真正高收益的调试方式，往往是一层配一张图。

### 10.1 表示层：看参数统计，不先看最终图

- `scale` 直方图
- `alpha` 直方图
- 高斯中心的 3D scatter

### 10.2 投影层：看投影中心和椭圆覆盖

- `mu_2d` 散点图
- 若干 `Sigma_2d` 椭圆 overlay
- 哪些点在图像外

### 10.3 渲染层：看 pred / gt / abs diff

- 预测图
- 真值图
- 绝对误差图

### 10.4 结构层：看 `N` 曲线和贡献分布

- `N vs step`
- radii 统计
- prune 比例

### 10.5 性能层：看 workload，而不是只看平均 FPS

- 每 tile 高斯数目热图
- frame time breakdown
- cache hit ratio

这些图的共同作用是：

```text
把“我觉得哪里不对”
变成“我知道问题先出在哪一层”
```

这就是一个系统工程实现者和一个只会调参的人，最核心的区别之一。

---

## 十一、一个最小可运行实验：把 projected ellipses 和 tile workload 画出来

下面这段代码不跑完整 3DGS，它只做一件对实现非常有价值的事：

> 给定几枚已经投影到屏幕上的 2D Gaussian，画出它们的椭圆 footprint、近似 tile 包围盒，以及每个 tile 的负载热图。

这段实验特别适合在你刚接入 tile mapping 时使用，因为它能帮你判断：

- footprint 大小是不是离谱
- tile 分配是不是和直觉一致
- 哪些 tile 会成为热点

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


W, H = 320, 224
tile = 32
n_tiles_x = W // tile
n_tiles_y = H // tile

gaussians = [
    {
        'mu': np.array([72.0, 78.0]),
        'Sigma': np.array([[320.0, 90.0], [90.0, 180.0]]),
    },
    {
        'mu': np.array([168.0, 112.0]),
        'Sigma': np.array([[520.0, -140.0], [-140.0, 260.0]]),
    },
    {
        'mu': np.array([248.0, 138.0]),
        'Sigma': np.array([[210.0, 30.0], [30.0, 120.0]]),
    },
]

tile_load = np.zeros((n_tiles_y, n_tiles_x), dtype=np.int32)
hit_boxes = []


def ellipse_axes(Sigma, nsig=2.0):
    vals, vecs = np.linalg.eigh(Sigma)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width = 2 * nsig * np.sqrt(vals[0])
    height = 2 * nsig * np.sqrt(vals[1])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle


for g in gaussians:
    mu = g['mu']
    Sigma = g['Sigma']

    # 用 3-sigma 的轴对齐包围盒近似 tile 覆盖范围
    std_x = np.sqrt(Sigma[0, 0])
    std_y = np.sqrt(Sigma[1, 1])
    bbox_min = mu - 3.0 * np.array([std_x, std_y])
    bbox_max = mu + 3.0 * np.array([std_x, std_y])

    tx0 = int(np.clip(np.floor(bbox_min[0] / tile), 0, n_tiles_x - 1))
    ty0 = int(np.clip(np.floor(bbox_min[1] / tile), 0, n_tiles_y - 1))
    tx1 = int(np.clip(np.floor(bbox_max[0] / tile), 0, n_tiles_x - 1))
    ty1 = int(np.clip(np.floor(bbox_max[1] / tile), 0, n_tiles_y - 1))

    hit_boxes.append((mu, Sigma, tx0, ty0, tx1, ty1))
    tile_load[ty0:ty1 + 1, tx0:tx1 + 1] += 1


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

axes[0].set_title('projected ellipses and tile boxes')
for x in range(0, W + 1, tile):
    axes[0].axvline(x, color='lightgray', linewidth=0.8)
for y in range(0, H + 1, tile):
    axes[0].axhline(y, color='lightgray', linewidth=0.8)

for mu, Sigma, tx0, ty0, tx1, ty1 in hit_boxes:
    width, height, angle = ellipse_axes(Sigma, nsig=2.0)
    axes[0].add_patch(
        Ellipse(mu, width, height, angle=angle, fill=False,
                edgecolor='tab:blue', linewidth=2)
    )
    axes[0].add_patch(
        Rectangle((tx0 * tile, ty0 * tile),
                  (tx1 - tx0 + 1) * tile,
                  (ty1 - ty0 + 1) * tile,
                  fill=False, edgecolor='tab:red', linewidth=1.5)
    )
    axes[0].scatter(mu[0], mu[1], c='black', s=18)

axes[0].set_xlim(0, W)
axes[0].set_ylim(H, 0)
axes[0].set_aspect('equal')
axes[0].set_xlabel('u')
axes[0].set_ylabel('v')

im = axes[1].imshow(tile_load, cmap='magma')
axes[1].set_title('gaussians per tile')
axes[1].set_xlabel('tile x')
axes[1].set_ylabel('tile y')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

nonzero = tile_load[tile_load > 0]
axes[2].hist(nonzero, bins=np.arange(1, nonzero.max() + 2) - 0.5, rwidth=0.8)
axes[2].set_title('tile load histogram')
axes[2].set_xlabel('# gaussians touching a tile')
axes[2].set_ylabel('count')

plt.tight_layout()
plt.show()
```

你应该观察到：

- 蓝色椭圆表示真正的 Gaussian footprint 几何
- 红色矩形表示你为了加速而给它分配的 tile 覆盖区域
- 右边热图能直接看出哪些 tile 会成为局部热点
- 直方图能告诉你：tile 负载是比较均匀，还是已经有明显重尾

这段实验的价值不在于“得到最终图像”，而在于帮你建立一个非常重要的实现习惯：

```text
把一个复杂模块拆成中间检查图来验
而不是把所有正确性都押在最终 render 上
```

---

## 十二、如果你要自己开仓库，最小骨架应该长什么样

如果把前面这些里程碑压成一个最小项目结构，它可以非常朴素：

```text
data/
    colmap_loader.py
    dataset.py

scene/
    gaussians.py
    init_from_sfm.py

render/
    project.py
    blend.py
    tile.py

train/
    losses.py
    density_control.py
    trainer.py

tools/
    diagnostics.py
    profile.py
```

这里最重要的不是目录名本身，而是职责切分：

- `gaussians.py`：只负责表示，不负责训练策略
- `project.py`：只负责把 3D 送到 2D
- `blend.py`：只负责 screen-space 混合
- `losses.py`：只负责监督定义
- `density_control.py`：只负责结构编辑
- `profile.py`：只负责性能度量

这能帮你避免一个特别常见的问题：

```text
为了“方便”把所有逻辑揉在一起
最后任何 bug 都会同时牵扯几层
```

第 9 章真正推荐的，不只是实现顺序，也是这种层次分离的习惯。

---

## 十三、把整章压成一个最短心智模型

如果你只想记一条链，就记这个：

```text
第 3 到第 8 章给你的是零件说明书：
Gaussian 是什么
渲染链是什么
loss 是什么
初始化是什么
训练闭环是什么
推理瓶颈是什么

    ↓

第 9 章做的事
不是再发明一个新公式
而是告诉你这些零件该按什么顺序装

    ↓

正确顺序不是：
一上来把训练、densify、加速全写完

    ↓

而是：
先做 2D footprint + blending
再做 3D projection
再做最小训练闭环
再接真实初始化和多视图训练
再做 densify / prune
最后再做 tile / cache / mixed precision / fusion

    ↓

每一层都要有自己的检查图和验收标准
这样 bug 才会被压回具体模块
而不是在整套系统里一起爆
```

这就是第 9 章真正想给你的实现视角。

---

## 十四、本章你真正应该能自己重建的几个问题

读完以后，遮住正文，你至少应该能自己回答：

1. 为什么 3DGS 的实现顺序不能按“功能越多越好”来排？
2. 为什么第一步更适合先做 2D Gaussian footprint + blending 沙盒？
3. 为什么接上 3D 后，第一张最重要的图通常不是最终 render，而是投影中心和椭圆 overlay？
4. 为什么最小训练闭环更适合先用简单 loss，而不是一上来就全套配置？
5. 为什么 densify / split / prune 必须晚于一个已经成立的训练主链？
6. 为什么性能优化一定要有一个慢速参考版做对照？
7. 为什么“每一层一张检查图”比只盯着最终 PSNR 更能帮你调试？
8. 如果系统图像全黑、loss 不降、`N` 爆炸、速度也很慢，你应该怎样按层拆问题，而不是一次性怀疑全部模块？

如果这些问题你能自己从头讲回来，这一章就真的进入你的脑子了。

---

## 十五、下一章接什么

现在你已经知道：

- 静态 3DGS 的表示、渲染、训练和推理优化各自是什么
- 如果自己实现，应该按什么顺序把它们接成系统
- 为什么实现路径本身就是一种工程设计

下一章 [chapter_10_4d_gaussian.md](chapter_10_4d_gaussian.md) 会自然接到另一个更难的问题：

> 如果静态场景这条链已经成立，但场景本身会随时间变化，那么前面那条 Gaussian -> projection -> blending 的渲染骨架，哪些部分还不变，哪些部分必须变成时间函数？

也就是从：

```text
“静态 3DGS 该怎样被真正实现出来”
```

走到：

```text
“如果场景开始动起来，这条实现主线要怎样被扩展成 4D”
```
