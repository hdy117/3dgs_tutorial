# 3D Gaussian Splatting (3DGS) 教程

_从问题出发，到feed-forward推理的完整学习路径_

---

## 教程导览

本教程采用 **第一性原理学习框架**：`problem → starting point → invention → verification → example`，带你从3D重建的痛点出发，理解3D Gaussian Splatting 为什么被发明、如何工作、以及如何实现。

**核心脉络**：
- **问题根源**：为什么需要3DGS？（NeRF的渲染瓶颈）
- **发明动机**：从体素到点云再到高斯椭球，每一步的必然性
- **技术实现**：可微分渲染 + 高斯优化的完整流程
- **实战验证**：从数据采集到实时渲染的完整pipeline

---

## 章节规划

### 第1章：背景与问题定义 - 为什么3D重建需要新思路？
**学习路径**：`problem → starting point`

内容：
- 3D重建的传统方法回顾（MVS、NeRF）
- NeRF的核心贡献与致命缺陷（渲染速度慢）
- 问题本质：如何在保持质量的同时实现实时渲染？

文件：`chapters/chapter_01_background.md`

---

### 第2章：从体素到点云 - 表示方法的演变
**学习路径**：`starting point → invention`

内容：
- 体素网格：空间离散化的朴素尝试
- 点云表示：稀疏性的引入
- 问题：点云如何渲染？如何建模表面？

文件：`chapters/chapter_02_representation_evolution.md`

---

### 第3章：核心发明 - 高斯椭球体作为3D表示
**学习路径**：`invention`（完整的第一性推导）

内容：
- **Axioms**：3D空间中的位置、方向、尺度、颜色
- **Contradictions**：点云无法渲染，体素太密集 → 需要"软"表示
- **Solution Path**：用2D协方差矩阵定义高斯椭球
- **Compression Mechanisms**：Splatting 投影 + 可微分排序
- **Verification**：为什么高斯是必然选择？

文件：`chapters/chapter_03_gaussian_splatting.md`

---

### 第4章：可微分渲染管线
**学习路径**：`invention → verification`

内容：
- 投影：3D高斯 → 2D屏幕空间
- Alpha blending 与排序问题
- 梯度反向传播通过渲染管线
- 为什么这是可微的NeRF替代方案？

文件：`chapters/chapter_04_differentiable_rendering.md`

---

### 第5章：优化目标与损失函数
**学习路径**：`verification`

内容：
- 图像重建损失（L1 + SSIM）
- 正则化项：尺度约束、透明度约束
- 自适应密度控制（densification/pruning）
- 优化循环：迭代调整高斯参数

文件：`chapters/chapter_05_optimization.md`

---

### 第6章：数据采集与初始化
**学习路径**：`example`（最小可行案例）

内容：
- COLMAP SfM 提取初始点云
- 从SfM参数初始化高斯属性
- 相机参数处理
- 数据集结构（nerf_synthetic格式适配）

文件：`chapters/chapter_06_data_preparation.md`

---

### 第7章：完整训练流程
**学习路径**：`example`

内容：
- 逐帧渲染 + 损失计算
- 梯度更新 + 密度控制
- 训练监控（PSNR、SSIM、LPIPS）
- 收敛判断与调参技巧

文件：`chapters/chapter_07_training_loop.md`

---

### 第8章：Feed-Forward推理 - 实时渲染
**学习路径**：`verification → example`

内容：
- 训练完成后：固定高斯集
- 实时投影与排序（CUDA实现）
- 与训练pipeline的区别
- 性能对比（NeRF vs 3DGS）

文件：`chapters/chapter_08_feed_forward.md`

---

### 第9章：扩展与变体
**学习路径**：`problem → invention`（进阶）

内容：
- 动态3DGS（4D Gaussian）
- 压缩与流式传输
- 与其他表示的结合（神经辐射场 + 高斯）

文件：`chapters/chapter_09_extensions.md`

### 第10章：实践路径 - 从零到可运行的实现
**学习路径**：`example`（完整实践指南）

内容：
- **阶段0**：环境准备（Python/CUDA/数据集）
- **阶段1**：数据加载模块（COLMAP解析 + Dataset类）
- **阶段2**：高斯初始化（从SfM点云 + 尺度估计）
- **阶段3**：投影渲染（慢速版调试 + tile优化）
- **阶段4**：训练循环（损失函数 + 优化器）
- **阶段5**：密度控制（densify/prune 实现）
- **阶段6**：Tile优化（预筛选 + 并行）
- **阶段7**：完整训练（30k步 + 评估）
- 调试清单 + 常见问题速查 + 实现检查清单

文件：`chapters/chapter_10_practice_path.md`

---

## 使用建议

1. **按顺序阅读**：每章都建立在前一章的基础上
2. **主动推导**：遇到公式先自己推，再看答案
3. **动手实践**：每章都有对应的代码片段（伪代码 + PyTorch实现提示）
4. **查漏补缺**：用"我能独立重推吗？"检验理解

---

**开始你的第一性原理之旅吧！🔥**
