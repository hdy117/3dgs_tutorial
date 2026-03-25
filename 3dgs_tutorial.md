# 3D Gaussian Splatting (3DGS) 教程

_从问题出发，到 feed-forward 推理的完整学习路径_

---

## 教程导览

本教程采用 **第一性原理学习框架**：`problem → starting point → invention → verification → example`，带你从 3D 重建的痛点出发，理解 3D Gaussian Splatting 为什么被发明、如何工作、以及如何实现。

**核心脉络**：
- **问题根源**：为什么需要 3DGS？（NeRF 的渲染瓶颈）
- **发明动机**：从体素到点云再到高斯椭球，每一步的必然性
- **技术实现**：可微分渲染 + 高斯优化的完整流程
- **实战验证**：从数据采集到实时推理的完整 pipeline
- **前沿扩展**：4DGS（动态场景）+ Feed-Forward GS（即时重建）

---

## 章节规划

### 第 1 章：为什么我们需要 3DGS？——从第一性原理推导视图合成

**核心问题**：如果你只有几张照片，怎么"看到"没拍过的角度？

内容：
- Novel View Synthesis 问题形式化
- 传统方法回顾（MVS、NeRF）及其瓶颈
- 为什么需要新思路？

文件：`chapters/chapter_01_background.md`

---

### 第 2 章：3D 表示的演化——为什么每一步都是必然的？

**核心问题**：体素、点云、NeRF、3DGS——这四个方法是怎么一步步"逼"出来的？

内容：
- Volume Rendering Equation 的本质
- 体素网格 → 点云 → MLP → 高斯的演进路径
- 每个阶段的矛盾如何驱动下一步发明

文件：`chapters/chapter_02_representation_evolution.md`

---

### 第 3 章：核心发明——为什么是高斯？从数学必然到工程实现

**核心问题**：点云需要体积 → 什么形状能在 3D 定义且投影到 2D 有解析解？

内容：
- **Axioms**：3D 空间中的位置、方向、尺度、颜色
- **Contradictions**：点是零维的，体素太密集 → 需要"软"表示
- **Solution Path**：协方差矩阵定义高斯椭球 + Splatting 投影
- **为什么高斯是唯一答案**（旋转不变性 + 闭合解）

文件：`chapters/chapter_03_gaussian_splatting.md`

---

### 第 4 章：从数学到图像——可微分渲染的完整推导

**核心问题**：投影公式怎么变成实际可计算的管线？梯度如何流回高斯参数？

内容：
- 符号约定与坐标系转换（世界→相机→屏幕）
- 3D 高斯 → 2D 高斯的投影推导（Jacobian 变换）
- Alpha blending 的排序问题与深度缓冲
- 反向传播通过渲染管线（为什么是可微的）

文件：`chapters/chapter_04_differentiable_rendering.md`

---

### 第 5 章：让高斯学会"正确"的事——优化目标的完整设计

**核心问题**："正确"到底是什么？像素颜色接近、边缘锐利、还是几何准确？

内容：
- L1 loss + SSIM 的组合动机（为什么不是 MSE？）
- 正则化项设计：尺度约束、透明度约束
- 自适应密度控制（densification/pruning）的触发条件
- 损失函数权重调参的经验法则

文件：`chapters/chapter_05_optimization.md`

---

### 第 6 章：数据准备与初始化——从 SfM 点云到高斯集合

**核心问题**：高素应该放在哪里？为什么不能随机初始化？

内容：
- COLMAP SfM 流程解析（特征匹配 → 稀疏重建）
- 从 sparse point cloud 提取初始高斯参数
- 尺度估计与协方差初始化（各向同性 vs 各向异性）
- 数据集格式适配（nerf_synthetic、real_blender）

文件：`chapters/chapter_06_data_preparation.md`

---

### 第 7 章：训练闭环——从初始化到收敛的完整迭代

**核心问题**：怎么让高斯"自动调整"到正确的值？什么时候算收敛？

内容：
- 训练循环设计（渲染 → loss → backward → step）
- 密度控制策略（何时 split/clone/prune）
- 学习率调度与收敛判断
- PSNR/SSIM/LPIPS监控与诊断工具

文件：`chapters/chapter_07_training_loop.md`

---

### 第 8 章：为什么训练完的模型跑不动？——实时推理优化

**核心问题**：推理不是"训练代码去掉反向传播"，而是完全不同的优化目标。

内容：
- 三个硬约束：<10ms 延迟、确定性输出、带宽友好
- 缓存策略发明：排序结果 + Tile 映射复用（相机移动很小时的洞察）
- 混合精度推理（float16 + Tensor Core 加速）
- Kernel Fusion 与 SoA 内存布局

文件：`chapters/chapter_08_inference_optimization.md`

---

### 第 9 章：从零到可运行——为什么"调库跑通"不等于理解？

**核心问题**：clone 官方代码 30 分钟 vs 从零实现 3 周，哪种真的懂？

内容：
- **Week 1: 让它动**（COLMAP 加载 + 慢速渲染 + 看到模糊图像）
- **Week 2: 让它快 + 好**（Tile-based 加速 + 密度控制 + PSNR>25）
- **Week 3: 让它飞**（float16 + 缓存策略 + CUDA，可选）
- 调试清单与常见问题速查表

文件：`chapters/chapter_09_practice_path.md`

---

### 第 10 章：为什么静态高斯跑不动视频？——4D Gaussian Splatting

**核心问题**：每帧独立训练要 25 小时，动态场景怎么办？

内容：
- **问题本质**：动态 ≠ 多帧静态拼接，需要时空统一表示
- **核心洞察**：动态 = 静态基底 + 时序变形场
- **混合变形架构**：静态高斯（70%）+ 刚体 SE(3) 变换（20%）+ MLP 非刚体（10%）
- **时序约束利用**：motion parallax 辅助深度估计

文件：`chapters/chapter_10_4d_gaussian.md`

---

### 第 11 章：为什么还要训练 30k 步？——Feed-Forward Gaussian Splatting

**核心问题**："离线优化 + 在线推理"两段式在即时重建场景失效。

内容：
- **三种方案对比**：Naive Image→GS ❌ / Iterative Refinement ✅ / Neural Codebook ⚠️
- **Learned Optimizer**：用网络拟合梯度下降轨迹（10 步替代 30k 步）
- **混合范式**：feed-forward warm start + few-shot fine-tuning（2s+5min ≈ 传统 30min 质量）
- 终极形态展望：One Model Fits All

文件：`chapters/chapter_11_feedforward_gaussian.md`

---

## 使用建议

1. **按顺序阅读**：每章都建立在前一章的基础上，跳跃阅读会丢失推导链条
2. **主动推导**：遇到公式先自己推，再看答案（用"我能独立重推吗？"检验理解）
3. **动手实践**：第 9 章的 3 周实战路径是检验理解的唯一标准
4. **查漏补缺**：利用每章的"关键概念细化"折叠块按需展开细节

---

## 学习路径图

```
基础层（必须掌握）:
  Ch1→Ch2→Ch3: 为什么是高斯？（问题定义 + 表示演化 + 核心发明）
  
中间层（理解实现）:
  Ch4→Ch5→Ch6→Ch7: 怎么训练？（渲染管线 + 优化目标 + 数据准备 + 训练闭环）

应用层（实战部署）:
  Ch8: 推理优化（实时渲染的关键技巧）
  Ch9: 从零实现（3 周完整实践路径）

进阶层（前沿扩展）:
  Ch10: 4DGS（动态场景）
  Ch11: Feed-Forward GS（即时重建）
```

---

**开始你的第一性原理之旅吧！🔥**
