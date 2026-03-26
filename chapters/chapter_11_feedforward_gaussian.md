# 第 11 章：为什么还要优化 30000 步——Feed-Forward Gaussian 到底想替代哪一段过程

**本章核心问题**：第 10 章已经把静态 3DGS 扩展到了动态场景。但无论是静态还是动态，一个很现实的问题仍然没消失：

> 为什么每来一个新场景，我们通常还是要做一轮 per-scene optimization？如果最终目标是得到一组 Gaussian，那能不能像图像分类或深度估计那样，直接一次前向把它预测出来？Feed-Forward Gaussian 真正想替代的到底是哪一段流程？

如果前面几章回答的是：

- 第 3 章：为什么 primitive 选 Gaussian
- 第 4 章：Gaussian 怎样被渲染成图
- 第 5 章：训练目标为什么这样设计
- 第 6 章：第一批 Gaussian 从哪来
- 第 7 章：静态场景怎样通过优化收敛
- 第 8 章：推理阶段为什么还要专门优化
- 第 9 章：如果自己实现，应该按什么顺序把系统搭起来
- 第 10 章：动态场景怎样把静态 Gaussian 扩展到时间维

那么这一章回答的就是：

```text
如果连长时间的 per-scene optimization 都不想做
Gaussian 能不能被直接预测出来
以及这件事为什么比“把网络最后一层改成输出高斯参数”难得多
```

先把主线写在前面：

```text
传统 3DGS 的核心范式是：
给定一个场景的数据
通过很多步优化
找到一组适合这个场景的 Gaussian

Feed-forward 的核心范式是：
提前在很多场景上学一个通用映射
让它看到新场景时
直接给出一组足够好的 Gaussian
```

但这里最重要的一句其实是：

> Feed-forward 不只是“让它更快”，而是在把“逐场景求解”改写成“跨场景摊销学习”。

也就是说，这一章真正讲的是 amortization：

```text
把原来每个场景都要重新付一次的优化成本
尽量提前摊到一个共享模型里
```

---

## 一、先分清：3DGS 里哪一段才是慢的，Feed-Forward 想砍掉的又是哪一段

很多人第一次听到 3DGS，会先记住：

```text
Gaussian Splatting 很快
```

这句话只说对了一半。

### 1.1 快的是渲染，不是得到那批 Gaussian 的过程

一旦场景已经训练好，推理时当然可以很快：

- 给定相机位姿
- 跑 projection / sorting / tile / blending
- 很快得到一张图

但在此之前，经典 3DGS 真正做的是：

```text
从一批粗糙初始化出发
跑很长的 per-scene optimization
最后才得到那组高质量 Gaussian
```

更形式化地写，传统静态 3DGS 对单个场景 `s` 要解的是：

```text
Theta_s^* = argmin_Theta E_k [L_img(render(Theta, cam_k), I_k^s)]
```

其中：

- `Theta` 是这个场景自己的 Gaussian 集合
- `I_k^s` 是场景 `s` 的训练图像
- `cam_k` 是对应相机

第 7 章讲的那条训练闭环，本质上就是在数值上近似求这个 `argmin`。

### 1.2 所以 Feed-Forward 真正想替代的，不是 renderer，而是“每个场景都重新求一遍最优解”

这点非常关键。

Feed-forward 不是在说：

```text
我不要第 4 章的渲染链了
```

它是在说：

```text
我不想每来一个新场景
都重新从初始化跑上万步优化
```

所以如果写成函数视角，传统方式更像：

```text
scene data -> optimize for many steps -> Theta_s^*
```

而 feed-forward 想做的是：

```text
scene data -> F_phi(scene data) -> Theta_hat_s
```

其中 `F_phi` 是一个在很多场景上学出来的共享模型。

### 1.3 这就是“优化式重建”和“摊销式重建”的根区别

传统 3DGS 是：

```text
每个新场景
都单独求解一次
```

feed-forward 则是：

```text
先在大量场景上学会“怎样重建”
以后面对新场景时
直接调用这个经验
```

你可以把这两种范式的差异想成：

- 传统方式像每次都现场雕刻一尊像
- feed-forward 更像先训练出一位熟练工匠，以后看到类似材料就能很快下手

这也是为什么第 11 章必须放在最后。

因为只有当你已经理解：

- Gaussian 表示是什么
- renderer 怎么工作
- 训练为什么慢
- 动态扩展怎么做

你才会真正明白：

```text
feed-forward 想偷掉的成本到底是哪一段
```

---

## 二、如果把问题正式写出来，Feed-Forward Gaussian 的目标到底是什么

对一个新场景 `s`，给定多视图输入：

```text
X_s = {(I_k^s, cam_k)}_k
```

传统优化式方法求的是：

```text
Theta_s^* = argmin_Theta E_k [L_img(render(Theta, cam_k), I_k^s)]
```

而 feed-forward 要学的是一个跨场景共享映射：

```text
F_phi : X_s -> Theta_hat_s
```

也就是：

```text
Theta_hat_s = F_phi(X_s)
```

这里 `phi` 不是某个场景自己的参数，而是在很多训练场景上学出来的模型参数。

### 2.1 这意味着目标发生了根本变化

传统 3DGS 的训练目标是：

```text
为当前场景找到一套最好参数
```

feed-forward 的训练目标则更像：

```text
学会一个函数
让它对很多未来新场景都能快速给出合理 Gaussian
```

所以它要解决的其实不是单场景最优化，而是：

```text
跨场景泛化
```

### 2.2 这就是为什么 Feed-Forward Gaussian 本质上和第 7 章不再是同一个问题

第 7 章关心的是：

- 一个场景怎样收敛
- loss 怎样往这个场景自己的最优点走
- densify / prune 怎样重分配该场景的表示容量

第 11 章关心的则是：

- 不同场景之间到底有没有共同规律
- 这些规律能不能被一个共享模型学到
- 学到以后，能不能在新场景上替代大部分 per-scene 优化

所以第 11 章不是简单加速版第 7 章。

它是一个新的问题层：

> 第 7 章是单场景求解；第 11 章是跨场景摊销求解。

---

## 三、为什么这件事比“直接回归一堆高斯参数”难得多

最直接的想法当然是：

```text
输入多张图像
输出 mu, Sigma, alpha, sh
```

听起来很自然。

但真正做起来，困难会立刻冒出来。

### 3.1 第一个困难：输出不是固定长度的向量，而是一组可变大小的 Gaussian 集合

一个静态场景最终需要多少 Gaussian，并不是固定的：

- 简单桌面可能几万就够
- 森林、城市、复杂室内可能要几十万
- 动态场景还可能随时间引入更多表达需求

所以真正的目标对象更像：

```text
Theta = {G_i}_i
```

是一个集合，而不是定长向量。

这会立刻引出两个问题：

- 高斯数量 `N` 该怎么定
- 不同场景的 `N` 不同时，网络怎么输出

### 3.2 第二个困难：高斯集合天然是无序的

这一点尤其关键。

对于同一个场景，下面这两种写法在渲染意义上是等价的：

```text
Theta = {G_1, G_2, G_3}
```

和：

```text
Theta' = {G_3, G_1, G_2}
```

也就是说，高斯集合本质上是 permutation-invariant 的。

但很多普通回归网络天然更擅长输出：

```text
第 1 个位置是什么
第 2 个位置是什么
第 3 个位置是什么
```

这会导致一个根问题：

> 你不能简单把预测的第 `i` 个高斯，拿去和“标签里的第 `i` 个高斯”做逐 index L2 监督，因为索引本身没有稳定语义。

### 3.3 第三个困难：从图像到 3D 本来就是病态逆问题

即使先不谈集合无序，单看几何恢复本身，也已经很难。

因为：

- 多视图覆盖不够时，深度本来就不充分
- 纹理弱区域本来就存在多解
- 外观变化、镜面、高光会进一步放大歧义

传统 3DGS 能做的一件事，是：

```text
通过很多步优化
不断让预测图和真实图对齐
慢慢把这些歧义压下去
```

而 feed-forward 如果想一跳直接给答案，它必须在一个前向里就把这些歧义消化掉。

这显然更难。

### 3.4 所以第 11 章真正的难点不在于“网络大不大”，而在于三件事同时存在

```text
输出是可变大小集合
集合是无序的
而问题本身还是一个病态逆问题
```

这也是为什么 feed-forward Gaussian 至今仍然是一个很活跃的研究方向，而不是一句“换成 Transformer 就行了”。

---

## 四、三种典型路线：优化式、混合式、完全 feed-forward

如果把当前主流思路按范式粗分，大致可以分成三类。

### 4.1 第一类：传统优化式（optimization-based）

这是最熟悉的 3DGS：

```text
X_s -> 初始化 -> 很多步优化 -> Theta_s^*
```

优点：

- 针对当前场景精细求解
- 质量通常最高
- 不要求跨场景训练数据

代价：

- 每个新场景都要重新优化
- 速度慢
- 难以满足即时重建场景

### 4.2 第二类：混合式 / warm-start（hybrid or warm-start）

这类方法不试图完全消灭优化，而是：

```text
先用共享模型快速给一个好起点
再用少量 per-scene optimization 补最后几步
```

可以写成：

```text
Theta_s^(0) = F_phi(X_s)
Theta_s^* ≈ refine(Theta_s^(0), X_s, K steps)
```

其中 `K` 比传统方法的小很多。

这类方法的核心思想是：

> 不必一口气替代全部优化，只要能替代最昂贵的冷启动阶段，就已经很有价值。

### 4.3 第三类：完全 feed-forward（fully amortized）

最激进的目标则是：

```text
Theta_hat_s = F_phi(X_s)
```

然后直接渲染、直接使用，尽量不再做后续 per-scene optimization。

优点当然很诱人：

- 推理快
- 使用体验好
- 部署简单

但它面临的挑战也最大，因为它要单靠共享模型吃掉几乎全部场景特异性。

### 4.4 这三类方法不是谁绝对先进，而是在不同位置做权衡

可以把它们理解成一条连续谱：

```text
优化式  -------------------  混合式  -------------------  完全前向式
质量高 / 速度慢              折中                         速度快 / 难度高
```

很多真正可落地的系统，反而更可能落在中间。

因为现实世界常常不是追求“理论上最纯粹的 end-to-end”，而是追求：

```text
在质量、速度、工程复杂度之间
得到最值钱的平衡点
```

---

## 五、为什么 warm-start 常常是最自然、也最实用的第一步

如果你把传统 3DGS 的训练曲线想一遍，会发现一个很重要的事实：

前面很多步，其实主要在做：

- 从很粗的 SfM 脚手架起步
- 把结构推到一个可用区间
- 慢慢把参数拉到正确盆地附近

而后面很多步更像：

- 精修
- 局部细化
- 把最后一点质量磨出来

### 5.1 这意味着 feed-forward 最先值得替代的，往往不是“最后精修”，而是“前面的冷启动和粗成形”

也就是：

```text
别从 random / SfM rough scaffold 开始
而是让共享模型直接给一个已经像样很多的初始 Gaussian 集合
```

于是混合范式就会显得非常自然：

```text
X_s -> F_phi(X_s) -> Theta_s^(0) -> few-step refinement -> Theta_s^*
```

### 5.2 为什么这种路线特别像“学会初始化器”

第 6 章说的是：

```text
SfM 初始化要把系统送进可训练区间
```

而 warm-start 方法更像是在做：

```text
学一个比 SfM 更聪明的初始化器
```

这个初始化器可能：

- 比稀疏点云更密
- 对尺度更敏感
- 对边缘、高频结构更有先验
- 已经隐含了很多跨场景学到的统计规律

### 5.3 这也是为什么 hybrid 方法常常比 fully feed-forward 更容易先做出成果

因为它没有要求共享模型一口气做到全部。

它只需要做到：

```text
把传统 30000 步优化里最笨重的前半段尽量替掉
```

剩下那部分仍然交给 per-scene optimization 收尾。

这会把问题变得更可控，也更符合工程直觉。

---

## 六、监督到底该怎么做：为什么“直接监督高斯参数”并不总是最自然

既然目标是让模型输出一组 Gaussian，那训练时最先想到的当然是：

```text
拿预测高斯去对齐目标高斯
```

听起来很合理，但正如前面说的，集合无序会让这件事变得不简单。

### 6.1 方案一：render-space supervision

最自然、也最稳的一种监督方式往往是：

```text
先把预测高斯渲染出来
再在图像空间监督它像不像
```

也就是：

```text
Theta_hat_s = F_phi(X_s)
L_render = E_k [L_img(render(Theta_hat_s, cam_k), I_k^s)]
```

这样做的好处是：

- 不要求预测高斯和某个“标签高斯”逐 index 对齐
- 直接对最终任务负责
- 和第 4、5 章的静态系统天然兼容

### 6.2 方案二：set matching / distillation

如果你确实有一个教师模型给出的高质量 Gaussian 集合作为参考，也可以尝试某种集合级对齐：

```text
L_set = match(Theta_hat_s, Theta_teacher_s)
```

这里 `match` 可能需要：

- 最近邻匹配
- Hungarian matching
- 局部 cluster matching
- 对分布统计做蒸馏

但这条路往往更复杂，因为它必须处理无序性。

### 6.3 方案三：优化轨迹蒸馏 / refinement distillation

如果你做的是 hybrid 或 learned refinement 路线，还可能会做：

```text
让网络不直接学最终 Theta^*
而是学“怎么从当前状态走向更好的状态”
```

这时监督对象会更像：

- 优化轨迹上的中间状态
- 某一步的 update direction
- 残差图该如何转成参数修正

### 6.4 所以 feed-forward Gaussian 的监督不只是一种“标签回归”，而是至少有三种层次

可以把它压成：

```text
最终图像像不像
预测出的高斯像不像老师的高斯分布
如果做混合式，更新方向像不像高效优化器
```

这也是为什么第 11 章实际上和第 4、5、7 章都强相关。

因为你需要同时理解：

- 渲染链
- loss
- per-scene optimization 的行为

才能真正明白：

```text
shared model 到底该学什么
```

---

## 七、为什么“高斯集合无序”会迫使你重新思考输出形式

这一节单独讲清楚，因为它太重要了。

### 7.1 如果你把输出写成固定长度数组，网络会自然把索引当成有意义的位置

例如：

```text
Theta_hat = [G_1, G_2, ..., G_N]
```

但对 Gaussian 场景来说，`G_1` 和 `G_2` 的编号本来就没有天然语义。

这会引发一个问题：

> 网络很容易学到“第 i 个槽位大致对应某类结构”，而不是学会灵活地生成任意场景需要的无序几何集合。

### 7.2 所以很多方法会转向更适合集合的表示方式

例如：

- proposal-based：先产出很多候选高斯，再用置信度或 mask 选择一部分
- codebook / token-based：先预测离散原型或 token，再解码成高斯
- point-set style 输出：让输出层天然更像集合而不是网格
- coarse-to-fine：先预测粗集合，再局部细化

### 7.3 这一步的真正意义是：输出结构必须和目标对象的数学结构相匹配

这一点和整本教程的主线其实完全一致。

第 3 章里我们强调：

```text
表示形式要和任务结构匹配
```

到了第 11 章，这句话只是换了一个层面继续成立：

> 如果目标是一个无序、可变大小的 Gaussian 集合，那输出头和监督方式也必须尊重这个结构，而不是偷懒把它硬塞回定长有序向量。

---

## 八、几个典型应用场景：为什么有些地方非常需要 feed-forward，而有些地方没那么急

### 8.1 即时预览 / 交互式重建

如果用户拿手机拍一圈，想几秒内就看到一个可旋转的粗 3D 结果，那么 per-scene optimization 的等待时间就会非常刺眼。

这种场景下，feed-forward 或 hybrid 的价值特别直接。

### 8.2 机器人 / AR / SLAM

这些系统往往不能等很久。

它们更需要：

- 快速得到一个可用几何表示
- 后面再边运行边细化

这时 warm-start 路线会非常自然，因为它既保留了速度，又允许在线 refinement。

### 8.3 高质量离线资产生产

如果目标是做一个高质量数字资产，用户能接受更长等待时间，那 fully feed-forward 的压力就没那么大。

这类场景下，传统优化式方法往往仍然非常有竞争力，因为它更愿意为单场景质量付出时间。

### 8.4 这说明 feed-forward 不是“传统 3DGS 马上会被淘汰”的结论

更准确的判断是：

```text
不同应用，对 amortization 的需求强弱不同
```

- 有些地方时间是第一约束
- 有些地方质量是第一约束
- 有些地方则非常适合两者混合

这也是为什么第 11 章的重点不是“宣布新范式全面胜利”，而是帮你看清：

> 什么时候值得为一次前向预测付出更多建模复杂度，什么时候传统 per-scene optimization 仍然是更值的选择。

---

## 九、一个最小可运行实验：冷启动优化、warm-start 和理想 fully feed-forward 的收敛曲线对比

下面这段代码不跑真正的 Gaussian 网络，它只做一件对理解第 11 章很有价值的事：

> 用 toy 曲线把三种范式的“时间 - 质量”关系画出来：传统冷启动优化、feed-forward warm-start + 少量 refinement、以及理想化的一次前向预测。

```python
import numpy as np
import matplotlib.pyplot as plt


steps = np.arange(0, 3001)

# 传统从粗初始化开始：早期慢，后面继续磨
psnr_cold = 10.0 + 20.0 * (1 - np.exp(-steps / 800.0))
psnr_cold += 1.2 * (1 - np.exp(-np.maximum(steps - 1800, 0) / 700.0))

# warm-start：一开始就站在更高位置，然后用少量优化补最后几步
psnr_warm = 24.0 + 6.2 * (1 - np.exp(-steps / 260.0))
psnr_warm += 0.8 * (1 - np.exp(-np.maximum(steps - 800, 0) / 500.0))

# 理想一次前向：几乎没有 refinement，但上限略低于充分优化
psnr_ff = np.full_like(steps, 27.8, dtype=np.float64)

# 近似时间成本（toy numbers）
time_cold = 0.0025 * steps              # 每步都很贵
warm_forward_time = 1.2                 # 初次前向一次性成本
ff_forward_time = 0.7                   # fully feed-forward 一次性成本

plt.figure(figsize=(9.2, 5.2))
plt.plot(time_cold, psnr_cold, label='cold-start optimization')
plt.plot(warm_forward_time + 0.0025 * steps, psnr_warm, label='warm-start + refinement')
plt.axhline(psnr_ff[0], linestyle='--', label='fully feed-forward')
plt.axvline(ff_forward_time, linestyle=':', color='gray')

plt.xlabel('relative time')
plt.ylabel('PSNR (toy dB)')
plt.title('Cold-start vs warm-start vs feed-forward')
plt.legend()
plt.tight_layout()
plt.show()
```

你应该观察到：

- `cold-start optimization` 需要很久才能爬到高质量区域
- `warm-start + refinement` 一开始就在更高位置，然后用较少步数补尾部质量
- `fully feed-forward` 速度最直接，但质量上限未必最高

这段实验最重要的作用不是给你真实 benchmark，而是帮助你建立一个非常重要的判断框架：

```text
feed-forward 研究不只是拼命追求“绝对不要优化”
很多时候更有价值的问题是：
我能不能用一个共享模型
把最昂贵的冷启动阶段尽量替掉
```

---

## 十、如果你今天要自己做一个 MVP，最合理的路线通常不是 fully feed-forward，而是 hybrid

如果你真的准备动手做第一个 feed-forward Gaussian 系统，最稳的切入点通常不是：

```text
一口气做出一个对任意新场景都能单次前向直接输出最优 Gaussian 的大模型
```

而更像是：

```text
先做一个能明显加速冷启动的 warm-start 系统
```

### 10.1 一个非常现实的 MVP 路线可以是

#### Step 1
先保留第 7 章的 renderer、loss 和 refinement 逻辑不动。

#### Step 2
训练一个初始化网络：

```text
X_s -> Theta_s^(0)
```

让它输出一组比 SfM 初始化更好的 Gaussian 脚手架。

#### Step 3
再从这个起点出发，只做少量 refinement：

```text
Theta_s^* ≈ refine(Theta_s^(0), X_s, K steps)
```

其中 `K` 比传统方法小很多。

### 10.2 为什么这条路线最值钱

因为它：

- 直接复用你已经有的静态 3DGS 系统
- 不要求一开始就完美解决集合无序和 variable `N` 的全部难题
- 也不要求一次前向就吞掉全部场景歧义
- 但已经可以显著减少总等待时间

### 10.3 这条路线也最容易验证价值

因为你可以直接比较：

- 从 SfM 初始化出发，收敛到某个 PSNR 需要多少步
- 从 learned warm-start 出发，需要多少步

如果后者明显更少，那这个共享模型就已经创造了真实价值。

所以第 11 章的一个特别现实的工程结论是：

> 最有希望先落地的，不一定是 fully feed-forward，而往往是“feed-forward warm-start + few-step refinement”。

---

## 十一、把整本书收束回来：静态、动态、feed-forward 其实是同一个框架的三种展开

如果读到这里再退远一点看，会发现第 3 章到第 11 章其实一直在讲同一个框架，只是每一章在替这个框架解决不同的问题。

### 11.1 第 3-4 章回答的是表示和渲染

- 为什么选 Gaussian
- 怎样从 Gaussian 变成像素

### 11.2 第 5-8 章回答的是怎样把它训练好、跑快

- 什么叫学对
- 初始化怎样给脚手架
- 训练闭环怎样收敛
- 推理怎样打掉带宽和排序瓶颈

### 11.3 第 10 章回答的是时间扩展

- 如果场景会动，参数怎样变成时间函数

### 11.4 第 11 章回答的是求解范式扩展

- 如果不想每个场景都重新优化，能不能把求解过程尽量摊销成一个共享前向模型

所以你可以把全书最后压成下面这条链：

```text
静态 3DGS：
学一个场景自己的 Theta

动态 4DGS：
学一个场景自己的 Theta(t)

Feed-forward GS：
学一个跨场景共享函数 F_phi
让它尽量直接产出 Theta 或 Theta(t) 的好起点
```

也就是说：

> 静态、动态、feed-forward 并不是三套互不相干的世界，而是同一个 Gaussian-render-optimize 框架，在“时间维”和“求解范式”两个方向上的继续展开。

---

## 十二、把整章压成一个最短心智模型

如果你只想记一条链，就记这个：

```text
传统 3DGS 的慢
不是慢在渲染
而是慢在每个新场景都要重新做很长的 per-scene optimization

    ↓

Feed-forward 想做的
不是替换第 4 章的 renderer
而是把“逐场景求解”改成“跨场景摊销学习”

    ↓

形式上就是：
传统：Theta_s^* = argmin_Theta L_s(Theta)
前向：Theta_hat_s = F_phi(X_s)
混合：Theta_s^(0) = F_phi(X_s)，再少量 refinement

    ↓

真正困难的地方在于：
输出是可变大小、无序的 Gaussian 集合
而从图像到 3D 本来又是病态逆问题

    ↓

所以最现实的路线往往不是一步到位 fully feed-forward
而是先做 warm-start / hybrid
替掉最贵的冷启动阶段

    ↓

静态、动态、feed-forward
其实都是同一个 Gaussian-render-optimize 框架的不同展开
只是它们分别在回答：
“场景是什么”
“场景怎样随时间变”
“这组 Gaussian 能不能被直接预测出来”
```

这就是第 11 章真正想让你建立起来的 feed-forward Gaussian 视角。

---

## 十三、本章你真正应该能自己重建的几个问题

读完以后，遮住正文，你至少应该能自己回答：

1. Feed-forward Gaussian 真正想替代的，到底是 3DGS 里的哪一段成本？
2. 为什么“实时渲染”不等于“即时重建”？
3. 为什么从图像直接回归 Gaussian 集合，会立刻遇到 variable `N`、集合无序和病态逆问题？
4. 为什么 warm-start / hybrid 往往是比 fully feed-forward 更现实的第一步？
5. render-space supervision、set matching 和 optimization distillation 分别在监督什么？
6. 为什么输出结构必须尊重 Gaussian 集合作为无序对象的数学性质？
7. 哪些应用最急需 amortization，哪些场景里传统 per-scene optimization 仍然很值？
8. 为什么说静态、动态、feed-forward 其实属于同一个 Gaussian-render-optimize 大框架？

如果这些问题你能自己从头讲回来，这一章就真的进入你的脑子了。

---

## 十四、全书最后的收束

到这里，这套教程真正想给你的不是某几个论文名词，而是一条完整的工程视角：

- 你知道为什么 primitive 最后选 Gaussian，而不是点、体素或 mesh 片段
- 你知道 3D Gaussian 怎样被投影、筛选、排序并混成图像
- 你知道什么叫“学对”，以及为什么 3DGS 的训练不只是一个 loss.backward()
- 你知道第一批 Gaussian 为什么必须借助 SfM 或别的几何脚手架
- 你知道一个静态场景怎样通过训练闭环走到收敛
- 你知道为什么训练完以后，推理阶段仍然会暴露 projection、sorting、tile 和带宽瓶颈
- 你知道如果自己实现，应该按什么顺序把这套系统一层层落下来
- 你知道当场景开始动起来时，参数怎样被扩展成时间函数
- 你也知道当每场景单独优化太慢时，为什么会自然走向 amortized / feed-forward 的方向

如果把整本书最后压成一句话，它大概就是：

```text
Gaussian Splatting 的真正价值
不只是“它很快”
而是它把三维重建、可微渲染、结构优化、时间扩展和摊销推理
都压进了一套足够规则、足够工程化、又足够可扩展的表示框架里
```

到这里，这条主线就闭合了。
