# 第6章：数据采集与初始化

**学习路径**：`example`（最小可行案例）

**核心目标**：从多视角照片和SfM输出，得到可训练的3D高斯集合

---

## 一、引言：从照片到可训练的高斯

**完整流程**：
```
照片 + 相机位姿 → SfM → 点云 → 初始化高斯 → 训练
```

本章聚焦：**SfM输出如何转化为高斯参数**

---

## 二、SfM（Structure from Motion）基础

### 2.1 什么是SfM？

从多张无序照片中恢复：
- 相机位姿（外参 R, t，内参 K）
- 稀疏3D点云（tracks）

**典型工具**：
- **COLMAP**（最常用，C++/Python，精度高）
- OpenMVG
- VisualSfM

### 2.2 COLMAP输出格式

```
database.db          # SQLite数据库（可选）
cameras.bin         # 相机模型 + 内参
images.bin          # 图像列表 + 外参
points3D.bin        # 3D点云 + 颜色 + tracks
```

**二进制格式**（推荐，更高效）：
- COLMAP提供Python API读取：`pycolmap`
- 或使用官方命令行工具转换：`colmap model_converter`

**points3D.bin 结构**：
```python
Point3D {
    id: int
    xyz: (3,) float32
    rgb: (3,) uint8
    error: float  # 重投影误差
    track: TrackElement[]  # 观测该点的所有图像
}
```

**TrackElement**：
```python
TrackElement {
    image_id: int
    point2D_idx: int  # 指向images.bin中的关键点
}
```

---

### 2.3 快速数据准备流程

```bash
# 1. 用COLMAP进行SfM
colmap feature_extractor --database_path db/ --image_path images/
colmap exhaustive_matcher --database_path db/
colmap mapper --database_path db/ --image_path images/ --output_path sparse/

# 2. 转换为文本格式（可选）
colmap model_converter --input_path sparse/ --output_path sparse_txt/ --output_type TXT

# 3. 验证（可视化点云）
colmap model_analyzer --path sparse/
```

---

## 三、从SfM点云初始化高斯

### 3.1 逐点初始化

对于每个SfM点 p = (X, Y, Z, R, G, B, error, track)：

**高斯参数**：
- 位置 μ = p.xyz（直接复制）
- 颜色 c = (R/255, G/255, B/255)（归一化到0-1）
- 协方差 Σ = ?（需要估计）
- 不透明度 α = 1（初始全不透明，后续优化）

---

### 3.2 协方差初始化策略

**问题**：SfM点只有位置，没有"尺度"或"方向"

#### 策略1：各向同性球（简单但效果差）

```
Σ = σ² * I，其中 σ = 0.01（初始很小）
```

**问题**：
- 太小的σ会导致渲染时几乎看不见（投影后值接近0）
- 需要快速densify来扩大，不稳定

---

#### 策略2：从点云邻域估计

**思路**：利用局部几何，对每个点找K近邻（如K=5），计算邻域点集的协方差矩阵

```python
def estimate_scale_from_neighbors(p, all_points, K=5):
    # 计算到所有点的距离
    dists = np.linalg.norm(all_points - p.xyz, axis=1)
    neighbor_ids = np.argsort(dists)[:K]
    neighbors = all_points[neighbor_ids]
    # 计算协方差
    centered = neighbors - neighbors.mean(axis=0)
    cov = (centered.T @ centered) / K
    # 取平均方差作为尺度
    scale = np.sqrt(cov.diagonal()).mean()
    return scale
```

**问题**：
- SfM点云稀疏，近邻可能距离很远 → Σ过大
- 边缘点近邻少 → 估计不准

---

#### 策略3：从相机几何估计（3DGS论文方案）

**核心洞察**：
- 每个点被多视角观测
- 每个视角下，该点在图像上的**像素不确定性**可以反推3D尺度

**公式**：
```python
def estimate_scale_from_reprojection(p, cameras, images):
    """
    p: SfM点，包含 track（观测该点的所有图像）
    cameras: 相机参数字典
    images: 图像外参列表
    """
    reprojection_errors = []
    for obs in p.track:
        img = images[obs.image_id]
        cam = cameras[img.camera_id]
        # 用相机参数投影p.xyz到像素
        R, t = img.R, img.t
        K = cam.K
        # 世界→相机
        X_cam = R @ p.xyz + t
        # 投影
        proj = K @ X_cam
        proj = proj[:2] / proj[2]
        # 重投影误差
        err = np.linalg.norm(proj - obs.pixel)
        reprojection_errors.append(err)
    # 取中位数（稳健估计）
    scale = np.median(reprojection_errors) * 0.5  # 保守系数
    return scale
```

**为什么用中位数？**
- 对抗离群点（某些视角误差大）
- 稳健估计

**尺度与协方差关系**：
- 假设各向同性：Σ = scale² · I
- 实际可稍大一点：Σ = (scale·1.5)² · I

---

### 3.3 不透明度初始化

**策略**：
- 所有 α = 0.5（半透明）
- 理由：给优化留空间（可增可减）
- 如果初始α=1，优化时很难降下来；α=0则无法上升

---

### 3.4 初始化伪代码

```python
def init_gaussians_from_sfm(sfm_points, cameras, images):
    """
    sfm_points: list of Point3D（从points3D.bin加载）
    cameras: dict{camera_id: Camera}
    images: list of Image（从images.bin加载）
    """
    gaussians = []
    for p in sfm_points:
        # 1. 位置和颜色
        mu = torch.tensor(p.xyz, dtype=torch.float32)
        c = torch.tensor(p.rgb / 255.0, dtype=torch.float32)

        # 2. 估计尺度（从重投影误差）
        scale = estimate_scale_from_reprojection(p, cameras, images)
        sigma = scale * 0.5  # 保守系数

        # 3. 协方差（各向同性）
        Sigma = torch.eye(3) * sigma**2

        # 4. 不透明度
        alpha = torch.tensor([0.5])

        gaussians.append(Gaussian(mu, Sigma, alpha, c))

    return GaussiansList(gaussians)  # 或合并为张量
```

---

## 四、相机参数处理

### 4.1 从COLMAP读取相机

**使用pycolmap**（推荐）：
```python
import pycolmap

recon = pycolmap.Reconstruction("sparse/0")
for camera_id, camera in recon.cameras.items():
    print(f"Camera {camera_id}: model={camera.model}, "
          f"params={camera.params}, width={camera.width}, height={camera.height}")

for image_id, image in recon.images.items():
    print(f"Image {image.name}: qvec={image.qvec}, tvec={image.tvec}, "
          f"camera_id={image.camera_id}")
```

**相机模型**：
- `PINHOLE`：最常用，4参数（fx, fy, cx, cy）
- `SIMPLE_PINHOLE`：3参数（fx, cx, cy），fy=fx
- `OPENCV`：8参数（含畸变）

**内参矩阵K**：
```python
if camera.model == "PINHOLE":
    fx, fy, cx, cy = camera.params
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
```

---

### 4.2 外参（R, t）从四元数转换

**四元数 → 旋转矩阵**：
```python
def qvec2rotmat(qvec):
    """qvec = (w, x, y, z)"""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R
```

**平移**：
- t = image.tvec（直接使用）

---

### 4.3 坐标系转换

**问题**：COLMAP坐标系 vs 渲染坐标系

- COLMAP：通常Y轴向上，Z向前（右手系）
- 3DGS/NeRF：Z轴向上，Y向下（图像坐标）

**转换**（可能需要）：
```python
# COLMAP → NeRF坐标系
transform = np.diag([1, -1, -1])  # Y和Z翻转
R = R @ transform
t = t @ transform  # 或 t = transform @ t（根据约定）
```

**验证方法**：
- 用初始点云渲染，看是否与输入图像对齐
- 如果图像左右颠倒或上下颠倒，调整R的符号

---

## 五、数据集结构

### 5.1 标准结构（参考NeRF）

```
dataset/
├── images/          # 所有图像
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── sparse/          # COLMAP输出
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── transforms.json  # 可选，NeRF格式
```

**transforms.json 格式**：
```json
{
  "camera_angle_x": 0.6911112070083618,
  "frames": [
    {
      "file_path": "images/0001",
      "transform_matrix": [[...], [...], [...], [...]]
    }
  ]
}
```

---

### 5.2 Dataset类实现

```python
class SfMDataset(Dataset):
    def __init__(self, sparse_path, image_path, split='train'):
        self.sparse_path = Path(sparse_path)
        self.image_path = Path(image_path)

        # 加载COLMAP reconstruction
        self.recon = pycolmap.Reconstruction(str(sparse_path))

        # 提取所有图像
        self.images = list(self.recon.images.values())
        # 过滤训练/测试（可按文件名或随机）
        if split == 'train':
            self.images = [img for img in self.images if 'train' in img.name]
        else:
            self.images = [img for img in self.images if 'test' in img.name]

        # 提取点云
        self.points3d = list(self.recon.points3D.values())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # 1. 加载图像
        img_file = self.image_path / img.name
        image = Image.open(img_file).convert("RGB")
        image = torch.from_numpy(np.array(image) / 255.0).float()
        image = image.permute(2, 0, 1)  # HWC → CHW

        # 2. 相机参数
        camera = self.recon.cameras[img.camera_id]
        K = build_K(camera)  # 见4.1节

        R = qvec2rotmat(img.qvec)
        t = img.tvec

        # 3. 坐标系转换（如需要）
        # R = R @ np.diag([1, -1, -1])

        return image, {
            'R': torch.tensor(R, dtype=torch.float32),
            'T': torch.tensor(t, dtype=torch.float32),
            'K': torch.tensor(K, dtype=torch.float32),
            'width': camera.width,
            'height': camera.height,
            'image_name': img.name
        }
```

---

## 六、初始化高斯集合

**流程整合**：

```python
# 1. 加载数据集
dataset = SfMDataset("data/sparse", "data/images", split='train')

# 2. 从SfM点云初始化高斯
gaussians = init_gaussians_from_sfm(
    dataset.points3d,
    dataset.recon.cameras,
    dataset.images
)

print(f"Initialized {len(gaussians)} gaussians")
# 典型：10k-100k

# 3. 移动到GPU
gaussians.to('cuda')

# 4. 设置优化器
optimizer = torch.optim.Adam([
    {'params': gaussians.mu, 'lr': 1.6e-4},
    {'params': gaussians.Sigma, 'lr': 1e-3},
    {'params': gaussians.alpha, 'lr': 5e-2},
    {'params': gaussians.color, 'lr': 5e-3}
])
```

---

## 七、常见陷阱与解决

### 陷阱1：坐标系不一致

**症状**：渲染图像与GT图像不匹配（旋转/翻转）

**诊断**：
- 用初始高斯渲染一帧，与GT并排显示
- 如果左右颠倒：R的y或z轴符号错
- 如果上下颠倒：R的y轴符号错

**解决**：
```python
# 尝试不同变换
transforms = [
    np.eye(3),
    np.diag([1, -1, -1]),
    np.diag([-1, 1, -1]),
    np.diag([-1, -1, 1])
]
```

---

### 陷阱2：尺度初始化过小

**症状**：训练初期画面全黑（高斯太小，投影后值接近0）

**诊断**：
- 检查 `gaussians.Sigma.diagonal().mean()` 是否 < 0.001
- 渲染时所有高斯值接近0

**解决**：
- 增大尺度估计的乘子：`scale = median_error * 5.0`（原0.5）
- 或直接设置最小初始尺度：`sigma = max(sigma, 0.1)`

---

### 陷阱3：不透明度初始化过低

**症状**：画面暗淡（α=0.5但高斯小，叠加后仍暗）

**解决**：
- α 初始化为 0.8 或 1.0

---

### 陷阱4：SfM点云噪声

**症状**：某些点明显是outlier，初始化后成为坏高斯

**诊断**：
- 检查重投影误差分布：`errors = [p.error for p in points3d]`
- 如果存在误差 > 10像素的点，考虑过滤

**解决**：
```python
# 过滤重投影误差大的点
max_error = 5.0  # 像素
filtered_points = [p for p in points3d if p.error < max_error]
```

---

## 八、思考题（独立重推检验）

1. **为什么SfM点云的尺度估计这么关键**？如果所有高斯初始尺度都是0.001，会发生什么？
2. **重投影误差和3D高斯的尺度有什么几何关系**？画图说明一个点在两个视角下的误差如何反映3D位置不确定性。
3. **为什么不直接用SfM点云作为"点"渲染**，而要加高斯？尝试用点精灵渲染初始点云，看看效果缺什么？
4. **坐标系转换**：如果COLMAP用右手Y-up，而渲染用右手Z-up，你需要哪些矩阵变换？写出变换矩阵。

---

## 九、下一章预告

**第7章**：完整训练流程 - 将所有组件（渲染、损失、优化、密度控制）整合成闭环，详细讲解训练循环、学习率调度、收敛判断。

---

**关键记忆点**：
- ✅ SfM工具：COLMAP（二进制格式 + pycolmap API）
- ✅ 尺度估计：从重投影误差的中位数
- ✅ 初始化：μ=点云xyz, c=RGB/255, Σ=scale²·I, α=0.5
- ✅ 坐标系：注意COLMAP与渲染坐标系的差异
- 🎯 **初始化质量决定训练稳定性**
