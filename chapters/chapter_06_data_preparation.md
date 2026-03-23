# 第6章：数据采集与初始化

**学习路径**：`example`（最小可行案例）

---

## 引言：从照片到可训练的高斯

**目标**：给定一组多视角照片，输出可训练的3D高斯集合

**完整流程**：
```
照片 + 相机位姿 → SfM → 点云 → 初始化高斯 → 训练
```

本章聚焦：SfM输出如何转化为高斯参数

---

## 1. SfM（Structure from Motion）基础

### 1.1 什么是SfM？

从多张无序照片中恢复：
- 相机位姿（外参 R, t，内参 K）
- 稀疏3D点云（tracks）

**典型工具**：
- COLMAP（最常用，C++/Python）
- OpenMVG
- VisualSfM

### 1.2 COLMAP输出格式

```
images.txt      # 图像列表 + 相机内参
cameras.txt     # 相机模型定义
points3D.txt    # 3D点云 + 颜色
```

**points3D.txt 示例**：
```
POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
```

- (X, Y, Z)：世界坐标
- (R, G, B)：观测到的平均颜色（0-255）
- TRACK[]：观测该点的图像和像素坐标

---

## 2. 从SfM点云初始化高斯

### 2.1 逐点初始化

对于每个SfM点 p = (X, Y, Z, R, G, B)：

**高斯参数**：
- 位置 μ = p.xyz（直接复制）
- 颜色 c = (R/255, G/255, B/255)（归一化到0-1）
- 协方差 Σ = ?（需要估计）
- 不透明度 α = 1（初始全不透明，后续优化）

---

### 2.2 协方差初始化策略

**问题**：SfM点只有位置，没有"尺度"或"方向"

**朴素方案1：各向同性球**
```
Σ = σ² * I，其中 σ = 0.01（初始很小）
```
- 问题：太小的σ会导致渲染时几乎看不见
- 需要快速densify来扩大，不稳定

**朴素方案2：从点云邻域估计**
- 对每个点，找K近邻（如K=5）
- 计算邻域点集的协方差矩阵
- 用该协方差作为初始Σ

**问题**：
- SfM点云稀疏，近邻可能距离很远 → Σ过大
- 边缘点近邻少 → 估计不准

**3DGS论文方案**：**从相机几何估计**

**核心洞察**：
- 每个点被多视角观测
- 每个视角下，该点在图像上的**像素不确定性**可以反推3D尺度

**公式**（简化）：
对于点 p，在相机 j 下的投影误差 e_j（重投影误差）
令：
```
scale = median(|e_j| for all j)  # 中位数重投影误差
Σ = scale² * I  # 初始各向同性
```

**为什么用中位数？**
- 对抗离群点（某些视角误差大）
- 稳健估计

---

### 2.3 不透明度初始化

**策略**：
- 所有 α = 0.5（半透明）
- 理由：给优化留空间（可增可减）
- 如果初始α=1，优化时很难降下来；α=0则无法上升

---

### 2.4 初始化伪代码

```python
def init_gaussians_from_sfm(sfm_points, sfm_cameras):
    gaussians = []
    for p in sfm_points:
        # 1. 位置和颜色
        mu = p.xyz
        c = p.rgb / 255.0

        # 2. 估计尺度（从重投影误差）
        reprojection_errors = []
        for obs in p.observations:
            camera = sfm_cameras[obs.camera_id]
            # 用相机参数投影mu到像素
            proj = project(mu, camera.R, camera.t, camera.K)
            error = distance(proj, obs.pixel)
            reprojection_errors.append(error)
        scale = median(reprojection_errors) * 1.0  # 乘子可调
        sigma = scale * 0.5  # 保守一点

        # 3. 协方差（各向同性）
        Sigma = sigma**2 * np.eye(3)

        # 4. 不透明度
        alpha = 0.5

        gaussians.append(Gaussian(mu, Sigma, alpha, c))
    return gaussians
```

---

## 3. 相机参数处理

### 3.1 内参矩阵K

从COLMAP cameras.txt：
```
CAMERA_ID, MODEL, WIDTH, HEIGHT, params[]
```

对于PINHOLE模型：
```
fx, fy, cx, cy = params
K = [[fx, 0, cx],
     [0, fy, cy],
     [0,  0,  1]]
```

### 3.2 外参（R, t）

从images.txt（每张图的外参）：
```
IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
```

四元数 Q → 旋转矩阵 R：
```
R = quaternion_to_matrix(Q)
```

平移 t = (TX, TY, TZ)

注意：COLMAP的坐标系和OpenGL/NeRF可能不同，需确认：
- 通常COLMAP：Y轴向上，Z向前
- 3DGS/NeRF：Z轴向上，Y向下（图像坐标）
可能需要调整：R = R * diag(1, -1, -1)

---

## 4. 数据集格式

### 4.1 标准结构

参考NeRF数据集格式（transforms.json）：
```
{
  "camera_angle_x": 0.6911112070083618,
  "frames": [
    {
      "file_path": "images/0001.png",
      "rotation": [0,0,0],   # 可选
      "translation": [0,0,0],
      "transform_matrix": [4x4矩阵]
    },
    ...
  ]
}
```

3DGS也支持类似格式，或直接用COLMAP输出。

---

### 4.2 数据加载流程

```python
class Dataset:
    def __init__(self, path):
        # 读取COLMAP输出
        self.cameras = load_cameras(path / "cameras.txt")
        self.images = load_images(path / "images.txt")  # 包含外参
        self.points3d = load_points3d(path / "points3D.txt")

        # 加载图像
        self.image_paths = [path / "images" / img.name for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        camera = self.cameras[img.camera_id]
        pose = get_pose(img)  # 4x4变换矩阵
        return img, camera, pose
```

---

## 5. 初始化高斯集合

**流程整合**：

```python
# 1. 加载SfM数据
dataset = Dataset("/path/to/sfm_output")
sfm_points = dataset.points3d

# 2. 初始化高斯
gaussians = init_gaussians_from_sfm(sfm_points, dataset.cameras)

# 3. 设置优化器
optimizer = torch.optim.Adam([
    {'params': gaussians.mu, 'lr': 1e-3},
    {'params': gaussians.Sigma, 'lr': 1e-3},
    {'params': gaussians.alpha, 'lr': 1e-2},
    {'params': gaussians.color, 'lr': 1e-2}
])

# 4. 开始训练循环（见第5、7章）
```

---

## 6. 常见陷阱

### 陷阱1：坐标系不一致
- COLMAP坐标系 vs 渲染坐标系
- 检查：用初始点云渲染，是否和输入图像对齐？
- 解决：调整旋转矩阵符号（通常 Y 和 Z 轴翻转）

### 陷阱2：尺度初始化过小
- 现象：训练初期画面全黑（高斯太小，投影后值接近0）
- 解决：scale 乘子调大（如 1.0 → 5.0）

### 陷阱3：不透明度初始化过低
- 现象：画面暗淡（α=0.5 但高斯小，叠加后仍暗）
- 解决：α 初始化为 0.8 或 1.0

### 陷阱4：SfM点云噪声
- 现象：某些点明显是outlier，初始化后成为坏高斯
- 解决：SfM后处理（filter points by reprojection error）

---

## 思考题（独立重推检验）

1. **为什么SfM点云的尺度估计这么关键？** 如果所有高斯初始尺度都是0.001，会发生什么？
2. **重投影误差和3D高斯的尺度有什么几何关系？** 画图说明一个点在两个视角下的误差如何反映3D位置不确定性。
3. **为什么不直接用SfM点云作为"点"渲染，而要加高斯？** 尝试用点精灵渲染初始点云，看看效果缺什么？
4. **坐标系转换**：如果COLMAP用右手Y-up，而渲染用右手Z-up，你需要哪些矩阵变换？写出变换矩阵。

---

**下一章**：完整训练流程 - 将所有组件（渲染、损失、优化、密度控制）整合成闭环。
