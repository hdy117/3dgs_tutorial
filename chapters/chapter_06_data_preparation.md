# 第6章：数据采集与初始化

**学习路径**：`example`（最小可行案例）

**核心目标**：从多视角照片和SfM输出，得到可训练的3D高斯集合

---

## 一、数据流程总览

```
多视角照片
    ↓
SfM (COLMAP)
    ↓
稀疏点云 + 相机位姿
    ↓
初始化高斯
    ↓
训练
```

---

## 二、SfM（Structure from Motion）详解

### 2.1 SfM能输出什么？

```
+----------------+-----------------+----------------------+
| 输出           | 格式            | 用途                 |
+----------------+-----------------+----------------------+
| 相机内参       | K矩阵 (fx,fy,cx,cy)| 投影计算           |
| 相机外参       | R(旋转), t(平移) | 世界→相机变换        |
| 稀疏点云       | (X,Y,Z,R,G,B,error,track)| 高斯初始化 |
| 重投影误差     | 每点每视角误差  | 尺度估计             |
+----------------+-----------------+----------------------+
```

---

### 2.2 COLMAP工作流（命令行）

**步骤1: 特征提取**
```bash
colmap feature_extractor \
  --database_path database.db \
  --image_path images/
```
输出: SIFT特征点 + descriptors

**步骤2: 特征匹配**
```bash
colmap exhaustive_matcher \
  --database_path database.db
```
输出: 特征点匹配对

**步骤3: 稀疏重建**
```bash
colmap mapper \
  --database_path database.db \
  --image_path images/ \
  --output_path sparse/
```
输出:
```
sparse/0/
├── cameras.bin    # 相机模型 + 内参
├── images.bin     # 图像外参（四元数 + 平移）
└── points3D.bin   # 3D点云 + RGB + tracks
```

**步骤4: 验证**
```bash
colmap model_analyzer --path sparse/0/
```
输出: 点云统计、重投影误差分布

---

### 2.3 重投影误差与尺度估计

**几何关系**:

```
3D点 P
    ↓ 投影到视角1
像素 p1 (观测值)
    ↓ 投影到视角2
像素 p2 (观测值)

重投影误差:
  e₁ = ‖p₁ - proj(P, camera₁)‖
  e₂ = ‖p₂ - proj(P, camera₂)‖
  ...

尺度估计: scale = median(e₁, e₂, ...)
```

**为什么用中位数?**
- 对抗离群点（某些视角误差大）
- 稳健估计

**尺度与协方差关系**:
```
Σ = (scale × factor)² · I
factor = 0.5 (保守) 或 1.0 (激进)
```

---

## 三、从SfM点云初始化高斯

### 3.1 初始化参数表

```
+--------+------+--------+----------------------+
| 参数   | 来源 | 初始值 | 说明                 |
+--------+------+--------+----------------------+
| μ      | SfM点云xyz | 直接复制 | 位置              |
| c      | SfM点云RGB | RGB/255 | 颜色（归一化）     |
| Σ      | 估计（见下）| scale²·I | 协方差（各向同性）|
| α      | 固定值 | 0.5    | 不透明度（半透明）  |
+--------+------+--------+----------------------+
```

---

### 3.2 尺度估计算法

**朴素方案**: Σ = 0.01²·I
- ❌ 太小 → 渲染几乎不可见

**正确方案**: 从重投影误差估计

**算法**:
```python
def estimate_scale_from_reprojection(p, cameras, images, factor=0.5):
    """
    p: SfM点，包含 track（观测该点的所有图像）
    cameras: 相机参数字典
    images: 图像外参列表
    """
    reproj_errors = []
    for track in p.track:
        img = images[track.image_id]
        cam = cameras[img.camera_id]
        
        # 投影
        R = qvec2rotmat(img.qvec)
        T = img.tvec
        K = build_K(cam)
        X_cam = R @ p.xyz + T
        proj = K @ X_cam
        proj = proj[:2] / proj[2]
        
        err = np.linalg.norm(proj - track.point2D)
        reproj_errors.append(err)
    
    scale = np.median(reproj_errors) if reproj_errors else 0.01
    return scale * factor
```

---

### 3.3 初始化伪代码

```python
def init_gaussians_from_sfm(points3d, cameras, images, scale_factor=0.5):
    gaussians = []
    
    for p in points3d:
        # 1. 位置和颜色
        mu = torch.tensor(p.xyz, dtype=torch.float32)
        color = torch.tensor(p.rgb / 255.0, dtype=torch.float32)
        
        # 2. 估计尺度
        scale = estimate_scale_from_reprojection(p, cameras, images, scale_factor)
        sigma = scale
        
        # 3. 协方差（各向同性）
        Sigma = torch.eye(3) * (sigma**2)
        
        # 4. 不透明度
        alpha = torch.tensor([0.5])
        
        gaussians.append(Gaussian(mu, Sigma, alpha, color))
    
    return GaussiansList(gaussians)
```

---

## 四、相机参数处理

### 4.1 坐标系转换问题

**COLMAP坐标系**（通常）:
- 右手系
- Y轴向上
- Z轴向前（相机光轴）

**NeRF/3DGS坐标系**（常用）:
- 右手系
- Z轴向上
- Y轴向下（图像坐标y向下）

**转换矩阵**:
```python
# COLMAP → NeRF
transform = np.diag([1, -1, -1])  # Y和Z翻转

R = R @ transform
T = transform @ T  # 或 T = T @ transform（根据约定）
```

**验证方法**:
1. 用初始高斯渲染一帧
2. 与GT图像并排显示
3. 如果左右/上下颠倒，调整R/T符号

---

### 4.2 内参矩阵K

**从COLMAP相机参数**:
```python
def build_K(camera):
    """camera: pycolmap.Camera对象"""
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
    elif camera.model == "SIMPLE_PINHOLE":
        fx, cx, cy = camera.params
        fy = fx
        K = np.array([[fx, 0, cx],
                      [0, fx, cy],
                      [0,  0,  1]])
    else:
        raise NotImplementedError(f"Model {camera.model}")
    return K
```

---

## 五、常见陷阱与诊断

### 5.1 陷阱诊断表

```
+--------+----------+----------+----------+
| 症状   | 可能原因 | 检查方法 | 解决方案  |
+--------+----------+----------+----------+
| 渲染全黑 | Σ太小    | Sigma.diag().mean() | scale_factor×5-10 |
| 渲染全白 | α太大    | alpha.mean() | α调至0.3 |
| 图像偏移 | 坐标系错 | 渲染vs GT对比 | 调整R/T符号 |
| 空洞多   | 初始点云稀疏 | len(points3d) | SfM调参数增加点 |
| 条纹伪影 | Σ奇异    | torch.det(Sigma) | Σ正则化 |
+--------+----------+----------+----------+
```

---

### 5.2 调试可视化代码

```python
def debug_initialization(gaussians, camera, H, W):
    """检查初始化质量"""
    # 1. 渲染
    rendered = render(gaussians, camera, H, W)
    
    # 2. 统计
    print(f"高斯数量: {len(gaussians)}")
    print(f"平均尺度: {gaussians.get_scales().mean():.6f}")
    print(f"平均α: {gaussians.alpha.mean():.6f}")
    
    # 3. 可视化
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(rendered.permute(1,2,0).cpu().numpy().clip(0,1))
    plt.title("Initial Render")
    
    plt.subplot(132)
    mu_2d, _, _ = project_gaussian(gaussians.mu, gaussians.Sigma,
                                    camera['R'], camera['T'], camera['K'])
    plt.scatter(mu_2d[:,0].cpu(), mu_2d[:,1].cpu(), s=1, alpha=0.5)
    plt.xlim(0, W); plt.ylim(H, 0)
    plt.title(f"Projected centers ({len(gaussians)})")
    
    plt.subplot(133)
    scales = gaussians.get_scales().max(dim=1)[0].cpu()
    plt.hist(scales.numpy(), bins=50)
    plt.title("Scale distribution")
    plt.tight_layout()
    plt.show()
```

---

## 六、思考题

1. **尺度估计**: 如果SfM重投影误差都很小（<1像素），说明什么？应该怎么调scale_factor？
2. **坐标系**: 如果COLMAP用OpenGL坐标系（右手Y-up），而渲染用OpenCV（右手Z-up），写出完整的变换矩阵
3. **初始化策略**: 为什么不直接用SfM点云作为"点"开始训练，而要加高斯？实验点精灵渲染，观察效果
4. **数据增强**: 在训练时，可以对图像做哪些增强？对相机参数有什么影响？

---

## 七、下一章预告

**第7章**：完整训练流程 - 将所有组件整合成闭环，详解训练循环、学习率调度、收敛判断、常见问题诊断。

---

**关键记忆点**:
- ✅ SfM工具: COLMAP（二进制 + pycolmap API）
- ✅ 尺度估计: scale = median(重投影误差) × factor
- ✅ 初始化: μ=xyz, c=RGB/255, Σ=scale²·I, α=0.5
- ✅ 坐标系: 注意COLMAP与渲染坐标系的差异（Y/Z翻转）
- 🎯 **初始化质量决定训练稳定性**