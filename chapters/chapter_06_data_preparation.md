# 第 6 章：数据准备与初始化——从 SfM 点云到高斯集合

**学习路径**：`axioms → contradictions → solution path → compression → verification`

---

## 问题起点：为什么需要初始化？

想象一下，你有一堆照片要重建 3D 场景。训练开始前，你必须回答一个根本性问题：

> **高斯应该放在哪里？**

### 对比神经网络训练

- **CNN**：权重随机初始化（Xavier、He），然后让梯度自己找路
- **3DGS**：不能随机初始化位置！如果高斯散落在错误地方，训练几乎不可能收敛

**为什么？**

因为我们要学的是**物理世界的几何结构**。随机位置的高斯就像把神经元放在随机坐标上——它们不知道场景中哪里有物体。

### 核心矛盾

```
SfM 点云 = 零维度的"点"（有位置、颜色，但无体积）
高斯      = 三维的椭球体（需要位置、尺度、朝向、不透明度）
         
问题：怎么把"点"变成"高斯"？
```

这不是简单的参数填充。每个选择——初始位置、初始尺度、初始不透明度——都会影响训练能否成功收敛。

---

## Axioms：不可约的基础事实

### 公理 1：SfM 提供了"场景先验"（Scene Priors）

**观察**：COLMAP 等 SfM 工具已经用多视角几何约束计算出了：
- 3D 点位置（被多个视图验证过）
- 相机位姿（外参 + 内参）

**这意味着什么？**

这些点不是随机猜测。它们是**通过三角测量得到的真实世界坐标**。如果 SfM 成功，点云已经大致勾勒出场景的几何骨架。

**推导**：初始化应该利用这个先验，而不是从零开始。

---

### 公理 2：点云的"不确定性"决定了高斯的体积

SfM 不是完美的。每个 3D 点的重投影误差（reprojection error）反映了它的位置有多不确定：

```
重投影误差 = || 预测像素位置 - 实际观测像素位置 ||₂
          ↓
大误差 → 该点位置不可靠 → 应该用更大的高斯体积覆盖不确定性
小误差 → 该点位置可信    → 可以用更小的高斯
```

**关键洞察**：初始尺度不应该是一个固定值，而应该从每个点的重建质量动态估计。

---

### 公理 3：各向同性是最安全的起点

你有一个选择：

- **选项 A**：各向异性高斯（沿表面法线拉长）
- **选项 B**：各向同性高斯（球形）

A 听起来更"智能"，但需要知道表面法线。SfM 点云没有显式法线——你需要从邻域拟合，这引入额外误差和超参数。

B 虽然简单，但有优势：
- 无额外假设
- 训练中会自动学出各向异性（梯度会拉伸高斯）
- 初始体积已由重投影误差决定，不会离谱

**结论**：用 Σ = scale²·I 做起点，让训练自己学形状。

---

### 公理 4：不透明度应该"半透明"

Alpha blending 的数学告诉我们：

```
C_out = α₁·C₁ + (1-α₁)·α₂·C₂ + ... 
```

如果 α=1（完全不透明）：后面所有高斯被遮挡，无法混合。
如果 α=0（完全透明）：无贡献。

**中间值 0.5** 是安全起点——允许足够透射率让后续高斯参与混合。

---

## Contradictions：初始化中的矛盾与解决路径

### 矛盾 1：点云稀疏 vs 渲染需要连续表面

**问题**：COLMAP 点云通常只有几万到几十万点，而且集中在"特征丰富"区域（角点、边缘）。大部分平坦表面是空洞。

但渲染需要连续的图像——不能只看到离散的点。

**解决路径**：
1. **接受事实**：初始渲染会有空洞（这是正常的）
2. **依赖 densify 机制**：训练中的密度控制会自动在高梯度区域分裂高斯，填补空洞
3. **调 SfM 参数**：如果点云太稀疏，可以调整 COLMAP 的 matcher/filterer 增加点数

---

### 矛盾 2：像素误差（2D）vs 世界尺度（3D）

重投影误差是像素单位（比如"1.5 像素"），但高斯尺度需要世界坐标单位。怎么转换？

**几何推导**：
```
相机坐标系中，点 P = (X, Y, Z)
投影到图像: x = fx·X/Z + cx

如果 P 沿 X 方向位移 ΔX（忽略ΔZ），像素变化:
Δx ≈ fx · (ΔX / Z)

反过来，从像素误差 e 推断 3D 位移:
ΔX ≈ e · Z / fx
```

**问题**：Z 是多少？每个点到相机的距离不同。

**工程妥协**（中位数启发式）：
- 取所有点重投影误差的中位数 `e_median`
- 用经验公式 `scale = e_median * factor`，factor ≈ 0.5~1.0
- **为什么有效？** 因为训练中的尺度是可学习的参数——初始值只要"差不多对"即可

---

### 矛盾 3：坐标系地狱（Coordinate Hell）

**COLMAP 约定**：
- Y 轴向上（+Y up）
- Z 轴向相机光轴方向（+Z forward）

**NeRF/3DGS 常见约定**：
- Z 轴向上（+Z up）
- Y 轴向下（图像坐标 y 向下，+Y down）

如果直接套用 COLMAP 的位姿，渲染出来的图像会上下颠倒、左右翻转。

**转换矩阵**：
```python
transform = np.diag([1.0, -1.0, -1.0])  # Y和Z翻转

R_nerf = R_colmap @ transform
T_nerf = transform @ T_colmap
```

---

## Invention：完整初始化算法推导

现在我们从 axioms 出发，构造一个完整的初始化流程。

### 输入与输出

**输入**（COLMAP 稀疏重建）：
- `points3D`：{id → {xyz, rgb, track}}
- `cameras`：{id → {fx, fy, cx, cy}}
- `images`：{id → {qvec, tvec, camera_id}}

**输出**（高斯集合）：
```python
class Gaussian:
    mu:      Tensor[3]   # 位置
    Sigma:   Tensor[3,3] # 协方差
    alpha:   Tensor[1]   # 不透明度
    color:   Tensor[3]   # RGB 颜色 [0,1]

gaussians = List[Gaussian]
```

### 步骤 1：坐标系转换（前置处理）

```python
def transform_colmap_to_nerf(R: Tensor[3,3], T: Tensor[3]) -> Tuple[Tensor[3,3], Tensor[3]]:
    """COLMAP (Y-up, Z-forward) → NeRF/3DGS (Z-up, Y-down)"""
    transform = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    R_nerf = R @ transform
    T_nerf = transform @ T
    return R_nerf, T_nerf

# 对所有图像应用转换
for img in images.values():
    R = quaternion_to_rotation_matrix(img.qvec)  # [w,x,y,z] → R[3,3]
    T = torch.tensor(img.tvec)
    img.R, img.T = transform_colmap_to_nerf(R, T)
```

---

### 步骤 2：从每个点的观测历史估计尺度

**核心思想**：如果一个点被多个视图观测，且投影位置都很准 → 该点可信度高，可以用小高斯。反之用大高斯。

```python
def estimate_scale_from_reprojection(
    point3D: Point3D,
    cameras: Dict[Camera],
    images: Dict[Image]
) -> float:
    """从重投影误差估计 3D 尺度"""
    
    reproj_errors = []
    
    for track in point3D.track:  # track = {image_id, x, y} (像素坐标)
        img = images[track.image_id]
        cam = cameras[img.camera_id]
        
        # 1. 构建内参矩阵 K
        if cam.model == "PINHOLE":
            K = torch.tensor([
                [cam.fx, 0,    cam.cx],
                [0,    cam.fy, cam.cy],
                [0,    0,      1]
            ])
        
        # 2. 投影：世界坐标 → 相机坐标 → 像素
        X_cam = img.R @ point3D.xyz + img.T   # (3,)
        proj_hom = K @ X_cam                   # (3,) 齐次
        proj = proj_hom[:2] / proj_hom[2]      # (2,) 归一化
        
        # 3. 计算误差
        err = torch.norm(proj - track.xy)
        reproj_errors.append(err.item())
    
    if not reproj_errors:
        return 1.0  # 默认值（无观测）
    
    # 用中位数抗离群点
    e_median = np.median(reproj_errors)
    
    # 经验公式：3D 尺度 ≈ 像素误差 × factor
    # factor=0.5 是常用起点，对应"渲染时约 0.5-1 像素可见度"
    scale_factor = 0.5
    return float(e_median * scale_factor)
```

**为什么用中位数？**  
某些点的 track 可能有坏匹配（离群值），用均值会被拉偏。中位数是稳健统计量的标准选择。

---

### 步骤 3：构造高斯集合

现在把所有参数拼起来：

```python
def initialize_gaussians_from_sfm(
    reconstruction: Reconstruction,
    scale_factor: float = 0.5,
    alpha_init: float = 0.5
) -> List[Gaussian]:
    """从 COLMAP SfM 初始化高斯集合"""
    
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = reconstruction.points3D
    
    # Step A: 转换所有相机位姿到 NeRF 坐标系
    for img in images.values():
        R = quaternion_to_rotation_matrix(img.qvec)
        T = torch.tensor(img.tvec)
        img.R, img.T = transform_colmap_to_nerf(R, T)
    
    gaussians = []
    
    # Step B: 为每个 3D 点创建高斯
    for pid, p in points3D.items():
        # 位置：直接复制 SfM xyz（公理1）
        mu = torch.tensor(p.xyz, dtype=torch.float32)
        
        # 颜色：归一化到 [0,1]（SfM 输出是 0-255）
        color = torch.tensor(p.rgb, dtype=torch.float32) / 255.0
        
        # 尺度：从重投影误差估计（公理2）
        scale = estimate_scale_from_reprojection(p, cameras, images) * scale_factor
        
        # 协方差：各向同性（公理3）
        Sigma = torch.eye(3) * (scale ** 2)
        
        # 不透明度：半透明起点（公理4）
        alpha = torch.tensor([alpha_init])
        
        gaussians.append(Gaussian(mu=mu, Sigma=Sigma, alpha=alpha, color=color))
    
    return gaussians
```

---

## Compression Mechanism：超参数简化策略

初始化有很多可调参数。但根据 axioms，大部分可以固定为"安全默认值"：

| 参数 | 推荐值 | 调参启发式 |
|------|--------|-----------|
| `scale_factor` | 0.5 | 渲染全黑 → ×10；全白模糊 → ÷2 |
| `alpha_init`   | 0.5 | >0.8:遮挡严重; <0.1:贡献不足 |

**关键原则**：初始化参数不是"魔法值"，而是基于物理直觉的工程妥协。只要初始状态"大致合理"，梯度优化会找到正确解。

---

## Verification：如何验证初始化质量？

### 检查 1：渲染第一帧（Initial Render）

```python
# 选一个视图（比如第一个相机）
camera = get_camera_from_image(list(images.values())[0])
rendered = render(gaussians, camera, H=height, W=width)

# 显示并对比 GT
plt.figure(figsize=(15, 5))
plt.subplot(131); plt.imshow(rendered.permute(1,2,0).cpu().numpy().clip(0,1)); plt.title("Initial Render")
plt.subplot(132); plt.imshow(gt_image); plt.title("Ground Truth")
plt.subplot(133); plt.imshow(np.abs(rendered.cpu().numpy() - gt_image).mean(axis=-1), cmap='hot'); plt.title("Error Map")
```

**期望结果**：
- ✅ 不是全黑/全白（至少有些东西）
- ✅ 有大致轮廓（即使模糊）
- ✅ 颜色基本正确（来自 SfM RGB）

---

### 检查 2：投影中心分布

把所有高斯中心投影到该视图的图像平面，看是否合理分布：

```python
# 将 mu 投影到像素坐标
mu_2d = project_points(gaussians, camera)  # (N,2)

plt.scatter(mu_2d[:,0], mu_2d[:,1], s=0.5, alpha=0.3)
plt.xlim(0, width); plt.ylim(height, 0)  # y轴翻转（图像坐标系）
plt.title(f"Projected centers: {len(gaussians)} gaussians")
plt.gca().set_aspect('equal')
```

**期望**：点分布大致覆盖场景区域，没有全挤在角落。如果偏移严重 → 检查 R/T 符号。

---

### 检查 3：尺度直方图

```python
scales = [g.get_scale() for g in gaussians]  # scale = sqrt(max eigenvalue of Sigma)

plt.hist(scales, bins=100)
plt.xlabel("Scale (world units)")
plt.ylabel("Count")
plt.title("Gaussian Scale Distribution")
```

**期望**：大部分尺度在合理范围（比如 0.1~10），没有极端值（>100 或 <0.001）。

---

## Example：典型初始化参数对比实验

假设我们用 COLMAP 重建了一个室内场景，得到 5 万点。尝试不同 `scale_factor`：

| scale_factor | 初始渲染效果 | 训练收敛情况 |
|-------------|------------|------------|
| 0.1 | 几乎全黑（高斯太小） | 梯度信号弱 → 慢收敛或不收敛 |
| **0.5** | **可见轮廓，略模糊** | ✅ 正常收敛 |
| 2.0 | 非常模糊，细节丢失 | 需要更多 densify 操作才能恢复细节 |
| 10.0 | 几乎全白（过度重叠） | alpha 衰减很慢，训练不稳定 |

---

## 调试检查表（完整陷阱清单）

| 症状 | 诊断方法 | 可能原因 | 解决方案 |
|------|---------|---------|---------|
| **渲染全黑** | `scales.mean()` | 尺度太小 | `scale_factor *= 10` |
| **渲染全白/模糊** | `scales.mean()`, `alpha.mean()` | 尺度太大或 α 太高 | `scale_factor /= 2`, `alpha_init=0.3` |
| **图像偏移** | Render vs GT 对比 | R/T 符号错 | 尝试翻转某轴 |
| **左右镜像** | 检查对称物体方向 | X 轴未转换 | `R[:,0] *= -1` |
| **上下颠倒** | 天空/地面位置 | Y 或 Z 翻转缺失 | 重新应用 transform 矩阵 |
| **大量空洞** | `len(gaussians)` vs GT 覆盖度 | 初始点云太稀疏 | 调整 COLMAP matcher/filterer 参数 |
| **条纹伪影** | `det(Sigma).min()` | Σ接近奇异 | `Sigma += I·1e-8`（正则化） |
| **颜色错误** | `color.max()`, `color.min()` | RGB 未归一化 | 确保 `/255.0` |

---

## 第一性原理思考题 🔥

**1. 尺度估计的闭环验证**  
如果初始高斯尺度设为 `scale = median(重投影误差)`，然后用这些高斯渲染，再把高斯中心投影回像素计算新的重投影误差——这个新误差应该接近原始 scale 吗？为什么？尝试推导从 3D 尺度到 2D 投影误差的映射公式。

**2. 为什么不直接用点精灵（Point Sprites）？**  
初始化时每个 SfM 点画一个固定半径的圆形（point sprites），比高斯简单得多。但为什么 3DGS 选择高斯？尝试用 point sprites 渲染初始图像，观察与高斯渲染的差异——你会发现在不同深度/视角下的缩放行为完全不同。

**3. 坐标系的"自动检测"**  
COLMAP 坐标系是固定的吗？如果用不同的相机模型（如 OPENCV vs PINHOLE），或者在 COLMAP 中用了 `--Mapper.ba_refine_extrinsics`，坐标系会变化吗？能否设计一个自动检测算法：用少量渲染样本对比 GT，推断出正确的坐标转换矩阵？

**4. Alpha 初始化的"马太效应"**  
如果所有高斯 α=0.9（接近不透明），从 alpha blending 公式推导：前几个排序靠前的高斯会"吃掉"大部分透射率。为什么 0.5 是安全值？尝试证明：当 N 个高斯等概率混合时，α=0.5 使得每个高斯的期望贡献最均匀。

**5. SfM 失败的备选方案**  
如果 COLMAP 完全失败（比如单目视频、动态场景），没有点云怎么办？随机初始化位置的高斯集合几乎不可能收敛——梯度会互相冲突。你能设计什么约束或正则化项，让"无先验"的初始化也能训练成功吗？

---

## 本章核心记忆点 ✅

**数据流程**：  
多视角照片 → COLMAP SfM（稀疏点云 + 相机位姿）→ 坐标系转换 → 高斯初始化 → 开始训练

**初始化参数推导**：
- `μ = SfM_xyz`（公理1：利用场景先验）
- `c = SfM_RGB / 255.0`（归一化）
- `Σ = scale²·I`（公理3：各向同性起点）
- `α = 0.5`（公理4：半透明允许混合）

**尺度估计核心公式**：  
`scale = median(重投影误差) × scale_factor (≈0.5)`  
——从重投影误差不确定性推导体积

**坐标系转换**：  
`transform = diag[1, -1, -1]` 将 COLMAP (Y-up) → NeRF (Z-up)

**调试三件套**：
1. 渲染初始图像（检查是否全黑/全白）
2. 投影中心分布图（检查坐标系对齐）
3. 尺度直方图（检查统计合理性）

---

## 下一章预告

第 7 章《训练循环》将把所有组件整合成完整闭环：初始化高斯 + 可微渲染管线 + 损失函数 + 密度控制 + 学习率调度。我们将推导一个从零开始的训练器，并深入分析训练中的动态演化过程——为什么需要 densify？何时应该停止？如何诊断不收敛的问题？
