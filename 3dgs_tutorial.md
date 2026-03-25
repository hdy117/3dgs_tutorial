# 3D Gaussian Splatting (3DGS) 教程

_从问题出发，到 feed-forward 推理的完整学习路径_

---

## 教程导览

本教程采用 **第一性原理学习框架**：`problem → starting point → invention → verification → example`，带你从 3D 重建的痛点出发，理解 3D Gaussian Splatting 为什么被发明、如何工作、以及如何实现。

**核心脉络**：
- **问题根源**：为什么需要 3DGS？（NeRF 的渲染瓶颈）
- **发明动机**：从体素到点云再到高斯椭球，每一步的必然性
- **技术实现**：可微分渲染 + 高斯优化的完整流程
- **实战验证**：从数据采集到实时渲染的完整 pipeline
- **前沿扩展**：4DGS（动态场景）+ Feed-Forward GS（即时重建）

---

## 章节规划

### 第 1 章：背景与问题定义 - 为什么 3D 重建需要新思路？
**学习路径**：`problem → starting point`

内容：
- 3D 重建的传统方法回顾（MVS、NeRF）
- NeRF 的核心贡献与致命缺陷（渲染速度慢）
- 问题本质：如何在保持质量的同时实现实时渲染？

文件：`chapters/chapter_01_background.md`

---

### 第 2 章：从体素到点云 - 表示方法的演变
**学习路径**：`starting point → invention`

内容：
- 体素网格：空间离散化的朴素尝试
- 点云表示：稀疏性的引入
- 问题：点云如何渲染？如何建模表面？

文件：`chapters/chapter_02_representation_evolution.md`

---

### 第 3 章：核心发明 - 高斯椭球体作为 3D 表示
**学习路径**：`invention`（完整的第一性推导）

内容：
- **Axioms**：3D 空间中的位置、方向、尺度、颜色
- **Contradictions**：点云无法渲染，体素太密集 → 需要"软"表示
- **Solution Path**：用 2D 协方差矩阵定义高斯椭球
- **Compression Mechanisms**：Splatting 投影 + 可微分排序
- **Verification**：为什么高斯是必然选择？

文件：`chapters/chapter_03_gaussian_splatting.md`

---

### 第 4 章：可微分渲染管线
**学习路径**：`invention → verification`

内容：
- 投影：3D 高斯 → 2D 屏幕空间
- Alpha blending 与排序问题
- 梯度反向传播通过渲染管线
- 为什么这是可微的 NeRF 替代方案？

文件：`chapters/chapter_04_differentiable_rendering.md`

---

### 第 5 章：优化目标与损失函数
**学习路径**：`verification`

内容：
- 图像重建损失（L1 + SSIM）
- 正则化项：尺度约束、透明度约束
- 自适应密度控制（densification/pruning）
- 优化循环：迭代调整高斯参数

文件：`chapters/chapter_05_optimization.md`

---

### 第 6 章：数据采集与初始化
**学习路径**：`example`（最小可行案例）

内容：
- COLMAP SfM 提取初始点云
- 从 SfM 参数初始化高斯属性
- 相机参数处理
- 数据集结构（nerf_synthetic 格式适配）

文件：`chapters/chapter_06_data_preparation.md`

---

### 第 7 章：完整训练流程
**学习路径**：`example`

内容：
- 逐帧渲染 + 损失计算
- 梯度更新 + 密度控制
- 训练监控（PSNR、SSIM、LPIPS）
- 收敛判断与调参技巧

文件：`chapters/chapter_07_training_loop.md`

---

### 第 8 章：为什么训练完的模型跑不动？——实时推理优化

**学习路径**：`verification → example`

内容：
- 推理 vs 训练的本质区别（延迟约束 vs 精度约束）
- 缓存策略：排序结果 + Tile 映射复用
- 混合精度推理（float16）
- Kernel Fusion 与内存布局优化

文件：`chapters/chapter_08_inference_optimization.md`

---

### 第 9 章：从零到可运行——为什么"调库跑通"不等于理解？

**学习路径**：`example`（完整实践指南）

内容：
- 3 周实战路径设计（Week 1: 让它动 / Week 2: 让它快 + 好 / Week 3: 性能优化）
- COLMAP 数据加载 + 慢速渲染实现
- Tile-based 加速 + 密度控制
- 调试技巧与常见问题速查

文件：`chapters/chapter_09_practice_path.md`

---

### 第 10 章：为什么静态高斯跑不动视频？——4D Gaussian Splatting

**学习路径**：`problem → invention`（动态场景）

内容：
- **问题本质**：每帧独立训练失效，需要时空统一表示
- **核心洞察**：动态 = 静态基底 + 时序变形场
- **混合变形架构**：静态高斯 + 刚体 SE(3) 变换 + MLP 非刚体
- **时序约束利用**：motion parallax 辅助深度估计

文件：`chapters/chapter_10_4d_gaussian.md`

---

### 第 11 章：为什么还要训练 30k 步？——Feed-Forward Gaussian Splatting

**学习路径**：`problem → invention`（范式转移）

内容：
- **问题本质**："离线优化 + 在线推理"两段式在即时重建场景失效
- **三种方案对比**：Naive Image→GS ❌ / Iterative Refinement ✅ / Neural Codebook ⚠️
- **Learned Optimizer**：用网络拟合梯度下降的轨迹
- **混合范式**：feed-forward warm start + few-shot fine-tuning

文件：`chapters/chapter_11_feedforward_gaussian.md`

---

## 使用建议

1. **按顺序阅读**：每章都建立在前一章的基础上
2. **主动推导**：遇到公式先自己推，再看答案
3. **动手实践**：每章都有对应的代码片段（伪代码 + PyTorch 实现提示）
4. **查漏补缺**：用"我能独立重推吗？"检验理解

---

**开始你的第一性原理之旅吧！🔥**
