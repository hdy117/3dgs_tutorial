# 线性代数教程 - Jupyter Notebook 代码库

## 📚 目录结构

```
code/notebooks/
├── README.md                    # 使用说明和索引
├── chapter_01_vectors.ipynb     # 第 3 章：向量基础实验
├── chapter_02_matrices.ipynb    # 第 5 章：矩阵变换可视化
├── chapter_03_covariance.ipynb  # 第 6 章：协方差与椭球
├── chapter_04_determinant_rank.ipynb  # 第 7 章：行列式与秩
├── chapter_05_eigenvalue.ipynb   # 第 8 章：特征值与特征向量
├── chapter_06_svd.ipynb          # 第 9 章：SVD 三步拆解
├── chapter_07_lowrank_pca.ipynb  # 第 10 章：低秩近似与 PCA
├── chapter_08_3dgs_practical.ipynb   # 第 11 章：3DGS 实战应用
├── chapter_09_common_mistakes.ipynb # 第 12 章：常见误解验证
└── chapter_10_all_experiments.ipynb   # 第 13 章：5 大实验合集
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install numpy matplotlib jupyter ipywidgets pandas
```

### 2. 运行 Notebook
```bash
jupyter notebook code/notebooks/
# 或
jupyter lab code/notebooks/
```

## 📖 每章对应关系

| Notebook | 对应章节 | 核心内容 |
|----------|---------|---------|
| `chapter_01_vectors.ipynb` | 第 3-4 章 | 向量运算、基变换、投影可视化 |
| `chapter_02_matrices.ipynb` | 第 5 章 | 矩阵推网格、缩放/旋转/剪切 |
| `chapter_03_covariance.ipynb` | 第 6 章 | 协方差椭球、特征分解验证 |
| `chapter_04_determinant_rank.ipynb` | 第 7 章 | 行列式几何意义、秩与奇异值 |
| `chapter_05_eigenvalue.ipynb` | 第 8 章 | 特征方向"只缩放不转弯"验证 |
| `chapter_06_svd.ipynb` | 第 9 章 | SVD三步拆解(Vᵀ→Σ→U)可视化 |
| `chapter_07_lowrank_pca.ipynb` | 第 10 章 | Eckart-Young定理、图像压缩、PCA=SVD |
| `chapter_08_3dgs_practical.ipynb` | 第 11 章 | 协方差传播、Jacobian 局部线性化 |
| `chapter_09_common_mistakes.ipynb` | 第 12 章 | 5 个误解的数值验证 (线性 vs 仿射等) |
| `chapter_10_all_experiments.ipynb` | 第 13 章 | 全部实验合集 (向量/矩阵/椭圆/特征/SVD) |

## 🔥 推荐学习顺序

1. **先跑实验**:从 `chapter_10_all_experiments.ipynb`开始，感受直观现象
2. **查缺补漏**:对照章节文档理解背后的数学原理
3. **深入实战**:运行 `chapter_08_3dgs_practical.ipynb`,连接 3DGS

## 📝 使用说明

每个 Notebook 都包含:
- ✅ 可运行的 Python/Matplotlib 代码块
- ✅ 预期输出示例 (以注释形式)
- ✅ 关键洞察说明 (Markdown 单元格)
- ✅ 与章节内容的对应关系链接

---

**作者**: Ember 🔥  
**版本**: v1.0 (2026-03-31)
