# 第6章：从照片到高斯——数据准备与初始化详解

**学习路径**：`problem → invention → verification`

**本章核心问题**：训练开始前，我们需要一个初始高斯集合。但3DGS不是从零训练（像神经网络那样随机初始化），而是**从SfM点云"转化"**而来。怎么转化？为什么这样转化？如果转化错了会怎样？

---

## 一、问题的本质：初始化为什么重要？

### 1.1 初始化 vs 训练

**对比神经网络**：
- CNN：随机初始化权重（Xavier, He），然后训练
- 3DGS：有**先验知识**（SfM点云），初始化不是随机的

**为什么3DGS需要SfM？**
- 因为我们要学的是**3D几何**，不是抽象的神经网络权重
- 点云提供了"场景中哪里有东西"的先验
- 没有SfM，高斯会散落在错误位置，训练可能不收敛

**这带来一个关键问题**：SfM点云是**点**（零维度），而我们需要**高斯**（有体积）。怎么给点加体积？

---

### 1.2 三个核心需求

从第一性原理，初始化必须满足：

**需求1：几何正确**
- 高斯位置应该在真实表面附近（不能偏离太远）
- 高斯朝向应该大致正确（虽然初始用各向同性，但位置要对）

**需求2：尺度合理**
- 尺度太大 → 渲染模糊，需要densify分裂
- 尺度太小 → 渲染几乎不可见（像噪点）
- 必须是"刚好可见"的尺度

**需求3：相机参数对齐**
- COLMAP的坐标系和渲染的坐标系可能不同
- 内参矩阵K的格式要正确
- 否则渲染图像会偏移/颠倒

**这三个需求是硬约束，初始化失败会导致训练失败。**

---

## 二、Axioms：从公理推导初始化策略

### 公理1：初始化应该"最小惊讶"（Least Surprise）

**含义**：初始状态应该让训练从"接近正确解"开始，而不是从"完全错误"开始。

**推导**：
- 如果初始高斯在错误位置，梯度会把它们拉到正确位置
- 但如果初始位置离正确位置太远，梯度方向可能混乱（多个视角冲突）
- **结论**：初始位置应该用SfM点云（已经被多视角几何验证过）

---

### 公理2：体积应该从"重建误差"估计

**问题**：给每个点加多大体积合适？

**观察**：SfM点云不是完美精确的，每个点有**重投影误差**（projected pixel vs observed pixel）。

**洞察**：
- 重投影误差大 → 该点位置不确定 → 应该给大体积（大Σ）
- 重投影误差小 → 该点位置确定 → 可以给小体积（小Σ）

**结论**：体积（尺度）应该从重投影误差估计，而不是固定值。

---

### 公理3：各向同性是最安全的起点

**问题**：初始协方差应该各向异性（沿法线拉长）还是各向同性（球）？

**分析**：
- 各向异性需要知道表面法线，但SfM点云没有显式法线
- 可以从邻近点拟合，但引入额外误差
- 各向同性虽然"浪费"参数，但：
  - 简单，无额外假设
  - 训练中会自动学出各向异性（通过梯度）
  - 初始体积已经由重投影误差决定，不会太离谱

**结论**：初始用各向同性 Σ = scale²·I，让训练自己学形状。

---

### 公理4：不透明度应该"半透明"

**问题**：初始α应该设多少？

**分析**：
- α=1（完全不透明）：会遮挡后面高斯，破坏混合
- α=0（完全透明）：无贡献
- **中间值**：0.5 是自然起点，允许后续调整

**结论**：α = 0.5 是安全起点。

---

## 三、Contradictions：初始化中的权衡

### 矛盾1：SfM点云稀疏 vs 渲染需要连续

**问题**：
- SfM点云只包含特征点（比如角点），大部分表面没有点
- 但渲染需要连续表面（不能只有点）

**解决方案**：
- 点云稀疏是事实，无法改变
- 但3DGS的**高斯有体积**，一个高斯可以覆盖周围区域
- 训练中的 densify 会自动填补空洞

**风险**：
- 如果点云太稀疏（比如只有1000点），初始渲染会有大空洞
- densify 需要时间（几千步）才能填补
- **怎么办？** 调整SfM参数增加点云密度，或接受前期空洞

---

### 矛盾2：重投影误差的尺度估计 vs 实际渲染尺度

**问题**：
- 重投影误差是**像素误差**（2D）
- 但我们需要**3D尺度**（世界坐标单位）
- 两者如何转换？

**推导**：
```
3D点 P 在相机坐标系中为 X_cam = (X, Y, Z)
投影到2D: x = fx·X/Z + cx

如果 P 有微小位移 ΔX（沿X方向），投影变化:
Δx ≈ fx·(ΔX/Z)  （忽略ΔZ项，假设Z不变）

反过来，如果重投影误差是 e（像素），对应的3D误差:
ΔX ≈ e·Z/fx

但Z是多少？对于该点，Z是它到相机的距离。
```

**简化方案**：
- 假设平均深度 Z_avg ≈ 场景的"典型深度"
- 取所有点的中位数深度
- 然后 ΔX ≈ e·Z_avg/fx_avg

**更简单的经验公式**：
```
scale_3d = e_pixel * factor
factor = 0.5 ~ 1.0（经验调优）
```

**为什么用中位数？**
- 对抗离群点（某些点误差大）
- 稳健估计

---

### 矛盾3：坐标系差异

**COLMAP坐标系**（通常）：
- 右手系
- Y轴向上（+Y up）
- Z轴向前（+Z forward，相机光轴）

**3DGS/NeRF坐标系**（常见）：
- 右手系
- Z轴向上（+Z up）
- Y轴向下（+Y down，图像坐标y向下）

**转换矩阵**：
```
transform = [1,  0,  0]
            [0, -1,  0]  # Y翻转
            [0,  0, -1]  # Z翻转

应用：
  R_new = R_old @ transform
  T_new = transform @ T_old
```

**验证方法**：渲染一帧，与GT图像对比，如果左右/上下颠倒，调整符号。

---

## 四、Solution Path：从SfM输出到高斯集合

### 4.1 完整初始化流程

```
SfM输出 (points3D.bin, cameras.bin, images.bin)
    ↓
解析 + 坐标系转换
    ↓
对每个3D点：
  μ = xyz（直接复制）
  c = RGB/255（归一化）
  scale = median(重投影误差) * factor
  Σ = scale²·I（各向同性）
  α = 0.5（半透明）
    ↓
高斯集合 {μ_i, Σ_i, α_i, c_i}
    ↓
训练开始
```

---

### 4.2 详细算法

**步骤1：加载COLMAP输出**

```python
import pycolmap

# 加载
recon = pycolmap.Reconstruction("sparse/0/")

# 相机
cameras = recon.cameras  # dict: camera_id → Camera

# 图像（外参）
images = recon.images  # dict: image_id → Image
# Image 有: qvec (四元数), tvec (平移)

# 点云
points3D = recon.points3D  # dict: point3d_id → Point3D
# Point3D 有: xyz, rgb, track (观测该点的所有图像)
```

**步骤2：坐标系转换**

```python
def transform_colmap_to_nerf(R_colmap, T_colmap):
    """
    COLMAP (Y-up, Z-forward) → NeRF/3DGS (Z-up, Y-down)
    """
    transform = np.diag([1.0, -1.0, -1.0])  # Y和Z翻转
    R_nerf = R_colmap @ transform
    T_nerf = transform @ T_colmap
    return R_nerf, T_nerf

# 应用到所有图像
for img in images.values():
    R = qvec2rotmat(img.qvec)  # 四元数→旋转矩阵
    T = img.tvec
    R, T = transform_colmap_to_nerf(R, T)
    img.R = R  # 保存
    img.T = T
```

**步骤3：尺度估计**

```python
def estimate_scale_for_point(point3D, cameras, images, scale_factor=0.5):
    """
    从重投影误差估计该点的高斯尺度（3D世界坐标单位）
    """
    reproj_errors = []
    
    for track in point3D.track:  # track 包含: image_id, point2D (像素坐标)
        img = images[track.image_id]
        cam = cameras[img.camera_id]
        
        # 1. 构建相机矩阵
        K = build_K(cam)  # 见4.2节
        
        # 2. 投影
        X_cam = img.R @ point3D.xyz + img.T  # (3,)
        proj_hom = K @ X_cam
        proj = proj_hom[:2] / proj_hom[2]
        
        # 3. 误差
        err = np.linalg.norm(proj - track.point2D)
        reproj_errors.append(err)
    
    # 4. 稳健估计（中位数）
    e_median = np.median(reproj_errors) if reproj_errors else 1.0
    
    # 5. 转换为3D尺度
    # 简单经验公式：3D尺度 ≈ 2D像素误差 × 典型深度 / 焦距
    # 但这里简化：直接用 e_median * factor
    scale = e_median * scale_factor
    
    return scale
```

**步骤4：初始化高斯**

```python
def init_gaussians_from_sfm(recon, scale_factor=0.5, alpha_init=0.5):
    """
    完整初始化
    """
    cameras = recon.cameras
    images = recon.images
    points3D = recon.points3D
    
    # 转换坐标系
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        T = img.tvec
        img.R, img.T = transform_colmap_to_nerf(R, T)
    
    gaussians = []
    for pid, p in points3D.items():
        # 位置
        mu = torch.tensor(p.xyz, dtype=torch.float32)
        
        # 颜色
        color = torch.tensor(p.rgb / 255.0, dtype=torch.float32)
        
        # 尺度
        scale = estimate_scale_for_point(p, cameras, images, scale_factor)
        Sigma = torch.eye(3) * (scale**2)
        
        # 不透明度
        alpha = torch.tensor([alpha_init])
        
        gaussians.append(Gaussian(mu, Sigma, alpha, color))
    
    return GaussiansList(gaussians)
```

---

### 4.3 为什么 scale_factor = 0.5？

**经验值来源**：
- 如果重投影误差 e = 1像素，scale = 0.5
- 对应3D尺度约 0.5 世界单位（场景尺度约10-100单位，所以0.5是小尺度）
- 这个尺度在渲染时刚好可见（投影到2D约0.5-1像素）

**如果 scale_factor 太大**：
- 初始高斯太大 → 渲染模糊 → 需要densify分裂
- 但分裂会增加计算量

**如果 scale_factor 太小**：
- 初始高斯太小 → 渲染几乎不可见（像噪点）
- 梯度信号弱 → 训练慢

**调参建议**：
- 先试 0.5
- 如果渲染全黑 → scale_factor *= 5-10
- 如果渲染全白（模糊）→ scale_factor /= 2

---

## 五、验证与调试：初始化质量检查

### 5.1 必须做的检查

**检查1：渲染初始图像**
```python
# 选一个视图
camera = get_camera_from_image(images[0])
rendered = render(gaussians, camera, H, W)

# 显示
plt.imshow(rendered.permute(1,2,0).cpu().numpy().clip(0,1))
plt.title("Initial Render")
plt.show()
```

**期望**：
- 不是全黑/全白
- 有大致轮廓（即使模糊）
- 颜色基本正确（来自SfM点云RGB）

**如果全黑**：尺度太小 → scale_factor ×= 10
**如果全白**：尺度太大或α太大 → scale_factor ÷= 2, α调至0.3

---

**检查2：投影中心分布**
```python
mu_2d = project_mu(gaussians.mu, camera)  # (N,2)
plt.scatter(mu_2d[:,0], mu_2d[:,1], s=1, alpha=0.5)
plt.xlim(0, W); plt.ylim(H, 0)
plt.title(f"Projected centers: {len(gaussians)}")
plt.show()
```

**期望**：
- 点分布在图像区域内（不是全在角落）
- 密度大致符合GT（粗糙匹配）

**如果偏移**：坐标系转换错误 → 检查R/T符号

---

**检查3：尺度分布**
```python
scales = gaussians.get_scales().max(dim=1)[0]  # (N,)
plt.hist(scales.numpy(), bins=50)
plt.title("Scale distribution")
plt.show()
```

**期望**：
- 大部分尺度在合理范围（比如0.1-10世界单位）
- 没有极端值（>100或<1e-6）

**如果有极端值**：
- 检查重投影误差计算
- clamp尺度到合理范围

---

### 5.2 陷阱诊断表（完整版）

```
+--------+----------+-------------------+----------+
| 症状   | 可能原因 | 检查方法          | 解决方案  |
+--------+----------+-------------------+----------+
| 渲染全黑 | Σ太小    | scales.mean()     | scale_factor×5-10 |
| 渲染全白 | Σ太大或α太大| scales.mean(), alpha.mean() | scale_factor÷2, α=0.3 |
| 图像偏移 | 坐标系错 | 渲染vs GT对比     | 调整R/T符号 |
| 左右翻转 | X轴翻转   | 镜像检查          | R[0,:] *= -1 |
| 上下颠倒 | Y轴翻转   | 上下对比          | R[1,:] *= -1 |
| 空洞多   | 初始点云稀疏| len(gaussians)   | SfM调参数增加点 |
| 条纹伪影 | Σ奇异或接近奇异| det(Sigma)最小值 | Σ += I·1e-8 |
| 颜色错误 | RGB未归一化| color.max()      | 确保 RGB/255 |
+--------+----------+-------------------+----------+
```

---

## 六、思考题（第一性原理式）

**1. 尺度估计的"闭环验证"**
- 如果初始高斯尺度设为 scale = median(重投影误差)
- 渲染后计算重投影误差（将高斯中心投影到像素）
- 这个误差应该接近 scale 吗？为什么？
- 尝试推导：从3D尺度到2D投影误差的映射关系

**2. 为什么不用点精灵（Point Sprites）？**
- 初始化直接用SfM点，每个点画一个圆形（固定半径）
- 这样简单，为什么3DGS要用高斯？
- 实验：用点精灵渲染初始图像，观察与高斯渲染的差异

**3. 坐标系转换的"唯一性"**
- COLMAP坐标系是固定的吗？还是可配置？
- 如果SfM时用了不同相机模型（比如OPENCV），坐标系会变吗？
- 如何自动检测坐标系（而不是凭经验猜）？

**4. 初始不透明度的"马太效应"**
- 如果α初始设为0.9（接近不透明），会怎样？
- 从alpha blending公式，高α高斯会"吃掉"透射率
- 推导：如果所有高斯α=0.9，混合后图像会怎样？
- 为什么0.5是安全值？

**5. 如果SfM失败（没有点云）怎么办？**
- 随机初始化高斯（位置随机，尺度固定）
- 但这样训练可能不收敛（高斯飞散）
- 你能想到什么约束/正则化让随机初始化也能训练吗？

---

## 七、本章核心记忆点

✅ **数据流程**：多视角照片 → COLMAP SfM → 稀疏点云 + 相机位姿 → 初始化高斯 → 训练

✅ **初始化参数**：
- μ = SfM点云 xyz
- c = SfM点云 RGB / 255
- Σ = scale²·I（各向同性）
- α = 0.5（半透明）

✅ **尺度估计**：scale = median(重投影误差) × scale_factor (0.5)

✅ **坐标系转换**：COLMAP (Y-up, Z-forward) → NeRF (Z-up, Y-down) 用 transform = diag[1, -1, -1]

✅ **调试检查**：渲染初始图像 → 投影中心分布 → 尺度分布

✅ **关键洞察**：初始化不是随机的，而是利用SfM的先验；体积从重投影误差估计；各向同性是安全起点

---

**下一章**：我们将把所有组件（SfM、初始化、渲染管线、损失、密度控制）整合成**完整训练闭环**，详解训练循环、学习率调度、收敛判断、以及如何诊断训练中的各种问题。
