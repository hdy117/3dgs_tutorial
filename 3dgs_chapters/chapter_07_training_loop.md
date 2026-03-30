# 第 7 章：训练闭环——从第一批高斯到收敛场景，中间到底发生了什么

**本章核心问题**：第 6 章已经解释了第一批 Gaussian 从哪来。现在问题变成：

> 这些还很粗糙的 Gaussian，怎样在一轮又一轮训练里，慢慢学成正确的位置、形状、透明度和外观？又为什么 3DGS 的训练不只是“反向传播 + Adam”这么简单？

如果前面几章已经分别回答了：

- 第 3 章：为什么 primitive 选 Gaussian
- 第 4 章：这些 Gaussian 怎样被渲染成图
- 第 5 章：什么叫“学对”，loss 为什么这样设计
- 第 6 章：第一批 Gaussian 从哪来

那么这一章要回答的就是：

```text
这些东西怎样真正被接成一条可运行、可诊断、可收敛的训练闭环
```

先把主线写在前面：

```text
初始化给你的是脚手架
渲染给你的是预测图
图像误差给你的是修正方向
optimizer 负责连续调参
periodic densify / split / prune 负责重分配表示容量
日志和可视化负责告诉你：系统是在收敛，还是已经开始出事
```

这就是 3DGS 训练循环最短的工程主线。

---

## 一、先把训练看成一条真正闭合的回路

如果只看一帧，你会以为 3DGS 和普通可微渲染没有本质区别：

```text
输入高斯 + 相机
-> 渲染
-> 图像损失
-> 反向传播
```

但 3DGS 真正的训练不是一帧，而是一条闭环：

```text
初始化高斯
-> 采样一个训练视角
-> 渲染当前图像
-> 和 GT 比较得到损失
-> 反向传播得到梯度
-> optimizer 更新参数
-> 周期性 densify / split / prune
-> 继续采样下一个视角
-> 重复很多轮，直到收敛
```

这条链里有两件事必须同时成立：

- 连续参数得往正确方向收敛
- 表示容量得在训练过程中被持续重分配

如果只有前者，没有后者，模型经常会“想学细节，但手里刷子不够”。
如果只有后者，没有前者，系统又会一直长结构，却不知道怎么把每个结构调准。

所以这一章真正讲的是：

> 3DGS 的训练为什么是一条“连续优化 + 结构编辑”联合驱动的闭环。

---

## 二、把训练循环先写成最核心的伪代码

如果把所有细节先压掉，3DGS 的训练主循环可以写成：

```python
for step in range(total_steps):
    gt_image, camera = sample_training_view(dataset)

    rendered, radii = render(gaussians, camera)

    l1 = l1_loss(rendered, gt_image)
    l_ssim = 1 - ssim(rendered, gt_image)
    l_img = (1 - lambda_dssim) * l1 + lambda_dssim * l_ssim

    optimizer.zero_grad()
    l_img.backward()
    optimizer.step()

    if step % densify_interval == 0:
        update_density_control(gaussians, radii)

    log_metrics(step, l1, l_img, gaussians)
```

这段伪代码已经足够说明本章最关键的分工：

- `render(...)`：把当前表示变成图像
- `l_img`：告诉系统“哪里不像”
- `optimizer.step()`：做连续参数更新
- `update_density_control(...)`：做离散结构编辑
- `log_metrics(...)`：检查系统是在学，还是在崩

你如果想理解一章训练循环，这就是骨架。

---

## 三、从第 6 章接过来：初始化给的是脚手架，不是答案

第 6 章已经讲过，初始化的目标不是“一开始就真”，而是：

> 把系统送进一个可训练区间。

所以一开始的 Gaussian 虽然已经不再是随机撒的，但仍然常常有下面这些问题：

- 位置只是大致对
- 形状还比较保守，往往偏各向同性
- 不透明度只是中等起点
- 细节远远不够
- 有些区域还没被充分覆盖

你可以把初始状态想成：

```text
场景的第一层脚手架已经搭起来了
但离真正可用的高质量表示还差很远
```

所以训练前期真正要做的，不是“精修到发丝级别”，而是先把系统从“粗糙但可训”推到“结构已经成形”。

---

## 四、每一步训练里，误差到底在推动什么

### 4.1 图像误差先在屏幕上暴露问题

对某个采样视角 `cam_k`，当前预测图像是：

```text
I_pred^k = render(Theta, cam_k)
```

真实图像是：

```text
I_gt^k
```

于是最核心图像项仍然是第 5 章那条：

```text
L_img = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)
```

常见设置仍然是：

```text
lambda_dssim = 0.2
```

也就是：

```text
L_img ≈ 0.8 * L1 + 0.2 * (1 - SSIM)
```

### 4.2 误差不会直接说“该加几个高斯”，它只会先说“哪里不像”

这点很重要。loss 本身只能表达：

- 哪些像素颜色不对
- 哪些边缘还糊
- 哪些局部结构还没守住

它不会直接说：

```text
这里该 clone 两个高斯
那里该 split 一个
那边该 prune 掉三个
```

所以训练闭环里必须再多一层解释：

> 先让图像误差生成梯度信号，再由训练规则把这些信号翻译成“连续调参”或“结构编辑”。

这正是第 5 章里“loss 和训练规则分工”的具体落地。

---

## 五、连续参数更新：optimizer 每一步到底在改什么

设当前参数集合是：

```text
Theta = {mu_i, Sigma_i, alpha_i, sh_i}_i
```

更贴近真实实现时，也常常会写成：

```text
Theta = {mu_i, scale_i, rotation_i, opacity_i, sh_i}_i
```

不管参数化怎样变，优化器每一步做的事情本质都一样：

### 5.1 `mu` 在学“东西该在哪”

如果渲染边缘总偏、局部结构总错位，位置梯度会推动 `mu` 调整。

### 5.2 `Sigma / scale / rotation` 在学“局部形状该怎样贴合结构”

如果 footprint 太胖、太窄、朝向不对，形状相关梯度会推动它变细、变宽、变斜。

### 5.3 `alpha / opacity` 在学“该遮多少，透多少”

太透明会导致区域总画不出来，太不透明又会压死后面的层。opacity 参数就是在平衡这件事。

### 5.4 `color / sh` 在学“看起来该是什么外观”

这部分最直接，就是把颜色和视角相关外观往 GT 拉近。

所以如果你把一整个 step 的连续优化说成一句话，就是：

> optimizer 在持续修正“放哪、长什么形、遮多少、看起来怎样”。

---

## 六、为什么训练里不能只靠连续调参

这一步是理解 3DGS 的真正分水岭。

假设某片细叶子区域本来只放了几个 Gaussian。你可以一直对这几个 Gaussian 做梯度下降，但它们能做的也就只有：

- 挪位置
- 改形状
- 调透明度和颜色

它们做不到的是：

```text
“这片区域其实本来就需要更多局部自由度，我自己长出更多 Gaussian 来”
```

所以训练里一定会出现这样的时刻：

> 误差已经在告诉你“这里还不够”，但当前这套表示本身没有足够容量把这个区域解释细。

这就是 densify / split / clone / prune 要登场的原因。

---

## 七、密度控制到底依赖什么信号

3DGS 里最常见的结构编辑信号，核心只有两类。

### 7.1 信号一：梯度长期偏大

如果某个 Gaussian 相关的梯度长期偏大，通常说明：

```text
这片区域持续没学好
```

它的含义不是“这一帧刚好有点误差”，而更像：

> 这里的表示容量可能不够，或者局部结构还没被合适拆开。

### 7.2 信号二：屏幕空间 footprint 太大

训练里常常会记录每个 Gaussian 投影后的 2D 半径或覆盖范围，记作：

```text
r_i
```

它反映的不是 3D 尺度本身，而是：

> 这个 Gaussian 在当前视图里到底在屏幕上铺得有多开。

这里一定要分清三件不同的东西：

- **3D 尺度**：Gaussian 在世界空间里有多大
- **2D 半径 / footprint**：它投到当前屏幕上覆盖多大
- **梯度阈值**：损失对这个 Gaussian 参数的敏感度有多强

这三者不是一回事，但它们会在训练决策里被一起使用。

### 7.3 一个特别实用的判断逻辑

你可以把密度控制最常见的逻辑压成：

```text
梯度大 -> 这里还没学好
footprint 大 -> 这里太粗了
两者都成立 -> 倾向 split / densify
```

这正是第 5 章里结构编辑思想在训练循环中的具体落地。

---

## 八、clone、split、prune 在训练循环里到底怎样分工

### 8.1 Clone：当前 Gaussian 不算太粗，但人数不够

如果某个 Gaussian：

- 误差信号大
- 但 footprint 并不算大

那常见判断是：

> 不是它太大，而是这个区域需要更多相近的局部自由度。

这时 clone 更合理。

### 8.2 Split：这个 Gaussian 太大，又一直学不好

如果某个 Gaussian：

- footprint 很大
- 误差信号又长期下不去

那通常说明：

> 一个 Gaussian 正在试图解释太大一块区域，应该拆成更细的几个。

这时 split 更合理。

### 8.3 Prune：它几乎没贡献了

如果某个 Gaussian 长期表现成：

- `alpha` 很低
- 可见贡献几乎没有
- 或者尺度已经退化

那删掉它通常更好，因为它会白白增加：

- 显存压力
- 排序和 tile 分配开销
- 每像素 blending 负担

所以 prune 的真正作用是：

> 把表示预算收回来，交给更有用的区域。

---

## 九、为什么 densify / prune 不该每一步都做

### 9.1 单步梯度太噪

每个 step 常常只采一个视角，或者只看一个很小 batch。于是该步梯度会强烈受到：

- 当前视角
- 当前遮挡关系
- 当前局部纹理

影响。

所以如果你每一步都根据瞬时梯度做结构增删，系统会非常抖。

### 9.2 更稳的做法：缓存梯度统计，再周期性决策

工程上更常见的是：

- 连续训练若干步
- 缓存或累计梯度统计
- 每隔 `densify_interval` 才做一次结构编辑

例如可以粗略写成：

```python
if step % densify_interval == 0:
    grads_mu = gaussians.mu.grad.detach().norm(dim=1)
    radii_cache = radii.detach()
    densify_and_prune(gaussians, grads_mu, radii_cache)
```

这不是在说“只看一次梯度就够了”，而是在强调：

> 结构编辑必须建立在比单步更稳的信号上。

### 9.3 还有一个现实问题：参数数量一变，优化器状态就变了

一旦高斯数目变化，参数张量就跟着变化。

这会影响 Adam 一类优化器内部的：

- 一阶动量
- 二阶统计

所以结构编辑不是“免费操作”，而是会打断一部分连续优化节奏。

这也是为什么它更适合作为周期性阶段动作，而不是每一步都插进去。

---

## 十、为什么学习率调度在训练闭环里也很重要

训练前期和后期，系统面临的问题并不一样。

### 10.1 前期：主要任务是把结构拉到位

这时常常：

- 位置还比较粗
- 表示容量在增长
- 很多区域还没进入细调阶段

所以通常更适合相对积极的学习率。

### 10.2 后期：主要任务是收敛和清理

到后面，场景大结构已经差不多成形，再用前期那种学习率，常见问题就是：

- 抖
- 收敛不稳
- 细节总是在最佳点附近来回晃

所以一个常见做法是阶段式衰减：

```text
step in decay_steps -> lr *= gamma
```

例如：

```text
7500 步、15000 步时各衰减一次
```

这背后的直觉很简单：

> 前期要快，后期要稳。

---

## 十一、一个更完整的训练伪代码

下面这段伪代码，把前面分散的东西串起来：

```python
config = {
    'total_steps': 30000,
    'densify_interval': 1000,
    'densify_from': 500,
    'prune_from': 15000,
    'lambda_dssim': 0.2,
    'lr_decay_steps': [7500, 15000],
    'lr_decay_factor': 0.1,
}

gaussians = init_from_sfm(...)
optimizer = build_adam_for_gaussians(gaussians)

for step in range(config['total_steps']):
    gt_image, camera = dataset.sample()

    rendered, radii = render(gaussians, camera)

    l1 = l1_loss(rendered, gt_image)
    l_ssim = 1 - ssim(rendered, gt_image)
    l_img = (1 - config['lambda_dssim']) * l1 + config['lambda_dssim'] * l_ssim

    optimizer.zero_grad()
    l_img.backward()
    optimizer.step()

    if step % config['densify_interval'] == 0 and step >= config['densify_from']:
        grads_mu = gaussians.mu.grad.detach().norm(dim=1)
        gaussians = densify_or_split_if_needed(gaussians, grads_mu, radii)

    if step >= config['prune_from']:
        gaussians = prune_inactive_gaussians(gaussians)

    if step in config['lr_decay_steps']:
        decay_learning_rate(optimizer, config['lr_decay_factor'])

    if step % 100 == 0:
        log_training_state(step, l1, l_img, gaussians)
```

注意这段伪代码最重要的不是具体函数名，而是结构顺序：

```text
采样
-> 渲染
-> 损失
-> 反向
-> 连续更新
-> 周期性结构编辑
-> 学习率调度
-> 日志诊断
```

这就是第 7 章最想让你抓住的骨架。

---

## 十二、训练循环里最常见的四种失败模式

真正做训练时，最可怕的不是公式不懂，而是系统已经出事，你却不知道它是在什么层面出的问题。

### 12.1 症状一：loss 不怎么降，PSNR 很快卡住

常见原因：

- 初始化太差，脚手架本身不在可训练区间
- 学习率太小，系统根本不动
- densify 触发太少，表示容量不够

这类问题本质上是在说：

> 系统想学，但当前自由度不够，或者根本没被推起来。

### 12.2 症状二：Gaussian 数量疯狂增长

常见原因：

- densify 阈值太激进
- prune 太慢或根本没起作用
- 训练一直在“补表示”，却没进入收敛阶段

这类问题的本质是：

> 系统一直在扩容，却没有把旧表示清理掉。

### 12.3 症状三：图像发白、发糊，或者 opacity collapse

常见原因：

- `alpha` 整体偏高，前排高斯吃掉太多透射率
- footprint 太大，局部结构全被糊掉
- 学习率过大导致透明度不稳定

这说明问题更多出在：

```text
遮挡和混合层面
```

### 12.4 症状四：协方差爆炸或数值不稳

常见原因：

- scale 参数跑飞
- 极端 footprint 让局部线性近似变差
- `Sigma_2d` 变得接近奇异

这类问题提醒你：

> 第 4 章那条可微渲染链虽然成立，但它需要被数值稳定地运行。

---

## 十三、训练时最值得盯的四类曲线

如果你只看一张最终图，很容易误判系统状态。真正有用的是看过程指标。

### 13.1 `PSNR / L1 / L_img`

它们告诉你：

- 图像质量是不是还在整体变好
- 收敛是在继续，还是已经平台期

### 13.2 Gaussian 数量 `N`

它告诉你：

- densify 是否真的在工作
- prune 有没有把冗余结构收回来
- 模型是不是在失控膨胀

### 13.3 `alpha` 分布

它告诉你：

- 系统是不是在大量生成几乎透明的“死 Gaussian”
- opacity 是否整体过高，导致前排压死后排

### 13.4 `scale` 分布

它告诉你：

- Gaussian 是否越来越细化
- 有没有出现一批异常大的 footprint
- 是否有大量退化到几乎不可见的结构

如果把最常见的训练图景压成一句话，就是：

```text
PSNR 慢慢上去
N 前期上升、中后期趋稳
alpha 和 scale 分布逐渐从粗糙走向分化和收敛
```

---

## 十四、一个最小可运行实验：把训练曲线直觉先画出来

下面这段代码不跑真实 3DGS，而是先把“训练闭环里最值得监控的曲线长什么样”可视化出来。

```python
import numpy as np
import matplotlib.pyplot as plt

steps = np.arange(30000)

# 造一组很像 3DGS 训练过程的 toy 曲线
psnr = 12 + 19 * (1 - np.exp(-steps / 6000))
psnr += 1.2 * (1 - np.exp(-np.maximum(steps - 12000, 0) / 5000))

loss = 0.55 * np.exp(-steps / 5000) + 0.08

N = 12000 + 90000 * (1 - np.exp(-steps / 8000))
N -= 18000 * (1 - np.exp(-np.maximum(steps - 18000, 0) / 4000))

alpha_mean = 0.45 - 0.12 * (1 - np.exp(-steps / 12000))
scale_mean = 0.35 - 0.20 * (1 - np.exp(-steps / 9000))

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

axes[0, 0].plot(steps, psnr)
axes[0, 0].set_title('PSNR vs step')
axes[0, 0].set_xlabel('step')
axes[0, 0].set_ylabel('PSNR (dB)')

axes[0, 1].plot(steps, loss)
axes[0, 1].set_title('loss vs step')
axes[0, 1].set_xlabel('step')
axes[0, 1].set_ylabel('loss')

axes[1, 0].plot(steps, N)
axes[1, 0].set_title('number of gaussians')
axes[1, 0].set_xlabel('step')
axes[1, 0].set_ylabel('N')

axes[1, 1].plot(steps, alpha_mean, label='alpha mean')
axes[1, 1].plot(steps, scale_mean, label='scale mean')
axes[1, 1].set_title('alpha / scale trends')
axes[1, 1].set_xlabel('step')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

你应该观察到：

- `PSNR` 前期上升快，后期趋缓，这是典型收敛曲线
- `loss` 应该整体下降，而不是长期剧烈震荡
- `N` 往往前期增长，后期趋稳甚至略回落，对应 densify 后再 prune
- `alpha` 和 `scale` 的统计趋势能帮助你看出系统是不是在“越学越稳”还是“已经开始跑飞”

这段代码虽然是 toy，但它非常适合作为第 7 章的第一层直觉训练。

---

## 十五、把整章压成一个最短心智模型

如果你只想记一条链，就记这个：

```text
第 6 章给的是第一批可训练 Gaussian 脚手架

    ↓

每个训练 step 都在做：
采样视角 -> 渲染 -> 图像损失 -> 反向传播 -> optimizer 更新

    ↓

但连续调参还不够
因为表示容量本身也要在训练中被不断重分配

    ↓

所以每隔一段时间，系统还会根据梯度和 footprint 做 densify / split / prune

    ↓

前期更像“搭结构”
中期更像“细化”
后期更像“收敛 + 清理”

    ↓

日志、曲线和可视化不是附属品
而是判断系统正在学、还是已经出事的主要窗口
```

这就是 3DGS 训练闭环最值得记住的主线。

---

## 十六、本章你真正应该能自己重建的几个问题

读完以后，遮住正文，你至少应该能自己回答：

1. 为什么 3DGS 训练循环不是“普通可微渲染 + Adam”这么简单？
2. 为什么连续参数更新和结构编辑必须同时存在？
3. 为什么图像误差只能回答“哪里不像”，却不能直接回答“该加几个 Gaussian”？
4. 为什么 3D 尺度、2D footprint 半径和梯度阈值不是一回事？
5. 为什么 densify / prune 必须建立在更稳的统计信号上，而不是单步梯度上？
6. 为什么训练前期和后期需要不同的学习率和不同的结构策略？
7. 为什么曲线和可视化在 3DGS 训练里不是附属品，而是核心诊断工具？
8. 如果 PSNR 不升、N 爆炸、图像发糊，你应该首先怀疑训练闭环的哪一层？

如果这些问题你能自己讲回来，这一章就真的进入你的脑子了。

---

## 十七、下一章接什么

现在你已经知道：

- 第一批 Gaussian 怎样被放进场景
- 训练循环怎样让它们逐步收敛
- densify / split / prune 怎样在训练中重分配表示容量

下一章 [chapter_08_inference_optimization.md](chapter_08_inference_optimization.md) 会自然接到另一个工程问题：

> 训练完以后，这套 Gaussian 已经学会了场景。但为什么直接拿训练时那套前向代码去渲染，常常还是慢得没法实时用？

也就是从：

```text
“怎么学会”
```

走到：

```text
“学会之后，怎么跑快”
```
