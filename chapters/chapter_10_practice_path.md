# 第10章：实践路径 - 从零到可运行的3DGS实现

**学习路径**：`example`（完整实践指南）

---

## 引言：理论到代码的最后一公里

读完前9章，你已经理解了3DGS的**为什么**和**是什么**。本章解决**怎么做**：

- 环境配置
- 代码架构设计
- 分阶段实现策略
- 调试与验证方法
- 常见陷阱与解决

**目标**：在1-2周内，用PyTorch实现一个能跑通小数据集的简化版3DGS（无需CUDA优化，CPU/GPU均可）。

---

## 阶段0：环境准备（1小时）

### 0.1 依赖安装

```bash
# Python 3.10+
conda create -n 3dgs python=3.10 -y
conda activate 3dgs

# PyTorch（根据CUDA版本选择）
# 如果有NVIDIA GPU：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# 如果只有CPU：
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 其他依赖
pip install tqdm opencv-python matplotlib numpy imageio scikit-image

# 可选：SSIM计算
pip install lpips  # 用于LPIPS指标
```

### 0.2 数据集准备

**快速开始**：使用COLMAP自带的`nerf_synthetic`数据集（小型合成场景）

```bash
# 下载示例数据（或用自己的COLMAP输出）
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/dataset/nerf_synthetic.zip
unzip nerf_synthetic.zip -d data/
```

**数据集结构**：
```
data/nerf_synthetic/chair/
├── train/
│   ├── 000.png
│   ├── 001.png
│   └── ...
├── test/
│   └── ...
└── transforms.json  # 或直接用COLMAP的cameras/images/points3D.txt
```

---

## 阶段1：数据加载模块（2-3小时）

**目标**：实现 `Dataset` 类，能加载COLMAP输出或 transforms.json，返回图像 + 相机参数

### 1.1 COLMAP解析器

```python
# utils/colmap_loader.py
import numpy as np
from pathlib import Path

def load_cameras(txt_path):
    """解析 cameras.txt，返回字典{camera_id: Camera}"""
    cameras = {}
    with open(txt_path) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array(list(map(float, parts[4:])))
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def load_images(txt_path):
    """解析 images.txt，返回列表[Image]"""
    images = []
    with open(txt_path) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = parts[9]
            images.append({
                'id': img_id,
                'qvec': np.array([qw, qx, qy, qz]),
                'tvec': np.array([tx, ty, tz]),
                'camera_id': cam_id,
                'name': name
            })
    return images

def qvec2rotmat(qvec):
    """四元数 → 旋转矩阵"""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R
```

### 1.2 Dataset类

```python
# data/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SfMDataset(Dataset):
    def __init__(self, colmap_path, split='train'):
        self.base_path = Path(colmap_path)
        self.cameras = load_cameras(self.base_path / 'cameras.txt')
        self.images = load_images(self.base_path / 'images.txt')
        self.points3d = load_points3d(self.base_path / 'points3D.txt')  # 自己实现

        # 过滤训练/测试
        self.images = [img for img in self.images if img['name'].startswith(split)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        cam_info = self.cameras[img_info['camera_id']]

        # 1. 加载图像
        img_path = self.base_path / 'images' / img_info['name']
        image = np.array(Image.open(img_path)) / 255.0  # H×W×3, float32
        image = torch.from_numpy(image).float().permute(2,0,1)  # C×H×W

        # 2. 构建相机参数
        R = qvec2rotmat(img_info['qvec'])
        T = img_info['tvec']
        K = self._build_K(cam_info)

        return image, {
            'R': torch.from_numpy(R).float(),
            'T': torch.from_numpy(T).float(),
            'K': torch.from_numpy(K).float(),
            'width': cam_info['width'],
            'height': cam_info['height']
        }

    def _build_K(self, cam_info):
        model = cam_info['model']
        params = cam_info['params']
        if model == 'PINHOLE':
            fx, fy, cx, cy = params
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        else:
            raise NotImplementedError
        return K
```

---

## 阶段2：高斯表示与初始化（3-4小时）

**目标**：从SfM点云初始化高斯集合

### 2.1 Gaussian数据结构

```python
# gaussian/gaussian.py
import torch

class GaussianModel:
    def __init__(self, points3d, cameras, images):
        """
        points3d: list of dict{xyz, rgb, error, track}
        cameras: 从COLMAP解析的相机字典
        images:  图像列表（用于计算重投影误差）
        """
        N = len(points3d)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 参数张量
        self.mu = torch.zeros((N, 3), device=device)       # 位置
        self.Sigma = torch.zeros((N, 3, 3), device=device) # 协方差
        self.alpha = torch.full((N, 1), 0.5, device=device) # 不透明度
        self.color = torch.zeros((N, 3), device=device)    # 颜色

        # 初始化每个点
        for i, p in enumerate(points3d):
            self.mu[i] = torch.tensor(p['xyz'], device=device)
            self.color[i] = torch.tensor(p['rgb'] / 255.0, device=device)

        # 估计尺度（从重投影误差）
        self._estimate_scales(points3d, cameras, images)

        # 初始化协方差（各向同性）
        scales = self.get_scales()  # (N, 3)
        for i in range(N):
            s = scales[i].clamp(min=1e-6)
            self.Sigma[i] = torch.diag(s**2)

    def _estimate_scales(self, points3d, cameras, images):
        """计算每个点的重投影误差中位数，作为尺度初值"""
        scales = []
        for i, p in enumerate(points3d):
            errors = []
            for img_id, px, py in p['track']:
                img = images[img_id]
                cam = cameras[img['camera_id']]
                # 投影
                R, T = img['R'], img['T']
                K = cam['K']
                proj = K @ (R @ self.mu[i] + T)
                proj = proj[:2] / proj[2]
                err = torch.norm(proj - torch.tensor([px, py], device=self.mu.device))
                errors.append(err.item())
            scale = np.median(errors) if errors else 0.01
            scales.append(scale * 0.5)  # 保守系数

        self.scale_init = torch.tensor(scales, device=self.mu.device)
```

---

## 阶段3：投影与渲染（4-5小时）

**目标**：实现第4章的可微分渲染管线（CPU版本先跑通）

### 3.1 投影函数

```python
# rendering/projection.py
import torch

def project_gaussian(mu, Sigma, R, T, K):
    """
    将3D高斯投影到2D
    输入：
      mu: (N, 3) 世界坐标
      Sigma: (N, 3, 3) 协方差
      R, T: (3,3), (3,) 相机外参
      K: (3,3) 内参
    输出：
      mu_2d: (N, 2) 像素坐标
      Sigma_2d: (N, 2, 2) 2D协方差
      depth: (N,) 深度（用于排序）
    """
    # 1. 转到相机坐标系
    mu_cam = (R @ mu.T).T + T[None, :]  # (N, 3)
    depth = mu_cam[:, 2].clone()

    # 2. 投影中心
    mu_hom = (K @ mu_cam.T).T  # (N, 3)
    mu_2d = mu_hom[:, :2] / mu_hom[:, 2:3]  # 除以z

    # 3. 投影协方差（简化：用雅可比近似）
    # W = R^T? 注意：这里需要仔细推导，先简化用正交投影
    Sigma_cam = R @ Sigma @ R.T  # (N, 3, 3)

    # 简化版：假设正交投影，J ≈ K * [1/z, 0, 0; 0, 1/z, 0]
    # 实际上应该用透视投影的雅可比
    z = mu_cam[:, 2].clamp(min=1e-6)
    J = torch.zeros((mu.shape[0], 2, 3), device=mu.device)
    J[:, 0, 0] = K[0, 0] / z
    J[:, 0, 2] = -K[0, 0] * mu_cam[:, 0] / (z**2)
    J[:, 1, 1] = K[1, 1] / z
    J[:, 1, 2] = -K[1, 1] * mu_cam[:, 1] / (z**2)

    Sigma_2d = J @ Sigma_cam @ J.transpose(1, 2)  # (N, 2, 2)

    return mu_2d, Sigma_2d, depth
```

### 3.2 2D高斯评估与Alpha Blending

```python
# rendering/render.py
def evaluate_gaussian_2d(x, mu_2d, Sigma_2d):
    """
    计算像素x处的高斯值
    x: (H, W, 2) 像素网格或 (2,) 单点
    mu_2d: (N, 2)
    Sigma_2d: (N, 2, 2)
    返回： (N,) 每个高斯在该点的值
    """
    N = mu_2d.shape[0]
    if x.dim() == 1:
        x = x.unsqueeze(0).expand(N, -1)  # (N, 2)

    diff = x - mu_2d  # (N, 2)

    # 计算逆协方差（简化：假设各向同性，直接除方差）
    # 实际需要 Cholesky 或直接用协方差求逆
    # 这里用简化版：假设 Sigma_2d = [[s^2, 0], [0, s^2]]
    # 则 exp(-0.5 * diff^T Σ⁻¹ diff) = exp(-0.5 * |diff|^2 / s^2)
    var = Sigma_2d.diagonal(dim1=1, dim2=2)  # (N, 2)
    inv_var = 1.0 / (var + 1e-8)
    exponent = -0.5 * (diff**2 * inv_var).sum(dim=1)
    return torch.exp(exponent)

def render_gaussians(gaussians, camera, H, W):
    """
    简化版渲染（CPU/GPU均可）
    gaussians: GaussianModel实例
    camera: dict{R, T, K}
    H, W: 图像分辨率
    返回：image (3, H, W)
    """
    # 1. 投影
    mu_2d, Sigma_2d, depth = project_gaussian(
        gaussians.mu, gaussians.Sigma,
        camera['R'], camera['T'], camera['K']
    )

    # 2. 按深度排序
    indices = torch.argsort(depth, descending=True)  # 远的先画

    mu_2d = mu_2d[indices]
    Sigma_2d = Sigma_2d[indices]
    alpha = gaussians.alpha[indices]
    color = gaussians.color[indices]

    # 3. 创建图像缓冲区
    image = torch.zeros((3, H, W), device=gaussians.mu.device)
    accum_alpha = torch.zeros((1, H, W), device=gaussians.mu.device)

    # 4. 遍历高斯（简化：每个像素都算所有高斯，实际应tile优化）
    # 注意：这里非常慢，仅用于验证正确性
    y, x = torch.meshgrid(
        torch.arange(H, device=gaussians.mu.device),
        torch.arange(W, device=gaussians.mu.device),
        indexing='ij'
    )
    pixels = torch.stack([x.float(), y.float()], dim=-1)  # (H, W, 2)

    for i in range(len(gaussians)):
        mu_i = mu_2d[i]  # (2,)
        Sigma_i = Sigma_2d[i]  # (2,2)

        # 计算该高斯对所有像素的贡献（简化：只算在3σ范围内的）
        diff = pixels - mu_i[None, None, :]  # (H, W, 2)
        exponent = -0.5 * (diff @ Sigma_i.inverse() * diff).sum(dim=2)
        gauss_val = alpha[i] * torch.exp(exponent)  # (H, W)

        # Alpha blending
        contrib = gauss_val[None, :, :] * color[i][:, None, None] * (1 - accum_alpha)
        image += contrib
        accum_alpha += gauss_val[None, :, :]

        if accum_alpha.max() > 0.99:
            break  # 早停

    return image
```

**⚠️ 注意**：上面的双重循环是O(N*H*W)，仅用于调试（N=100, H=W=100时还能接受）。正式训练必须换tile-based实现（见阶段4）。

---

## 阶段4：训练循环与损失（3-4小时）

**目标**：整合渲染、损失、优化器、密度控制

### 4.1 损失函数

```python
# training/loss.py
import torch
import torch.nn.functional as F

def ms_ssim_loss(pred, target):
    """简化版：用单尺度SSIM"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = F.avg_pool2d(pred, 3, 1, 1)
    mu_target = F.avg_pool2d(target, 3, 1, 1)

    sigma_pred = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_pred ** 2
    sigma_target = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_target ** 2
    sigma_cross = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred * mu_target

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    return 1 - ssim.mean()

def total_loss(rendered, gt, gaussians, λ_ssim=0.8, λ_scale=0.01):
    L1 = F.l1_loss(rendered, gt)
    L_ssim = ms_ssim_loss(rendered, gt)
    L_img = (1 - λ_ssim) * L1 + λ_ssim * L_ssim

    # 尺度正则
    scales = gaussians.get_scales()  # (N, 3)
    L_scale = torch.clamp(scales - 1.0, min=0).mean()

    return L_img + λ_scale * L_scale, L1, L_ssim
```

### 4.2 训练循环（简化版，无密度控制）

```python
# train.py（简化先跑通）
from torch.utils.data import DataLoader
from data.dataset import SfMDataset
from gaussian.gaussian import GaussianModel
from rendering.render import render_gaussians  # 慢速版，仅调试
from training.loss import total_loss

dataset = SfMDataset('data/nerf_synthetic/chair', split='train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 初始化高斯（用第一帧的SfM点云）
# 实际应该用所有帧的SfM点云
first_img, first_camera = dataset[0]
gaussians = GaussianModel(
    dataset.points3d,  # 需要从points3D.txt加载
    dataset.cameras,
    dataset.images
)

optimizer = torch.optim.Adam([
    {'params': gaussians.mu, 'lr': 1e-3},
    {'params': gaussians.Sigma, 'lr': 1e-3},
    {'params': gaussians.alpha, 'lr': 5e-2},
    {'params': gaussians.color, 'lr': 5e-3}
])

for epoch in range(30):
    for img, camera in dataloader:
        # 渲染
        rendered = render_gaussians(gaussians, camera, H=camera['height'], W=camera['width'])

        # 损失
        loss, L1, L_ssim = total_loss(rendered, img.squeeze(0), gaussians)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, L1={L1:.4f}, SSIM={1-L_ssim:.4f}")

    # 每epoch保存一次渲染结果
    save_image(rendered, f"output/epoch_{epoch:03d}.png")
```

---

## 阶段5：密度控制实现（5-6小时）

**目标**：实现densify和prune（第5、7章）

### 5.1 梯度缓存

```python
# training/density_control.py
def cache_gradients(gaussians):
    """缓存梯度幅度，用于densify判断"""
    grads_mu = gaussians.mu.grad.detach().norm(dim=1)
    grads_Sigma = gaussians.Sigma.grad.detach().view(gaussians.N, -1).norm(dim=1)
    return grads_mu, grads_Sigma
```

### 5.2 Densify & Prune

```python
def densify_and_prune(gaussians, optimizer, radii, camera,
                      grad_threshold=0.0002, scale_threshold=0.01,
                      prune_alpha=0.001, max_gaussians=2e6):
    """
    radii: (N,) 每个高斯在屏幕上的投影半径（像素），用于尺度判断
    """
    with torch.no_grad():
        # 1. Densify
        grads_mu, grads_Sigma = cache_gradients(gaussians)

        # 条件：投影尺度大 + 梯度大
        large_scale = radii > scale_threshold
        large_grad = (grads_mu > grad_threshold) | (grads_Sigma > grad_threshold)
        to_densify = large_scale & large_grad

        # 克隆或分裂
        if to_densify.any():
            # 简化：只克隆（不分裂）
            new_mu = gaussians.mu[to_densify].clone()
            new_Sigma = gaussians.Sigma[to_densify].clone()
            new_alpha = gaussians.alpha[to_densify].clone()
            new_color = gaussians.color[to_densify].clone()

            gaussians.extend(new_mu, new_Sigma, new_alpha, new_color)
            optimizer.add_param_group({'params': [new_mu, new_Sigma, new_alpha, new_color]})

        # 2. Prune
        scales = gaussians.get_scales().max(dim=1)[0]
        to_prune = (gaussians.alpha.squeeze() < prune_alpha) | (scales < 1e-6)

        if to_prune.any():
            gaussians.mask = ~to_prune  # 标记保留
            optimizer.prune(gaussians.keep_mask)  # 需要自己实现optimizer.prune

        # 3. 数量限制
        if len(gaussians) > max_gaussians:
            # 随机删除多余高斯
            keep = torch.randperm(len(gaussians))[:int(max_gaussians)]
            gaussians.mask = keep
            optimizer.prune(keep)
```

---

## 阶段6：性能优化（可选，2-3小时）

**如果阶段1-5能跑通小数据集（10-20张图），继续优化到实时渲染**

### 6.1 Tile-based渲染（核心）

参考第4章，实现tile预筛选：

```python
def render_gaussians_tiled(gaussians, camera, H, W, tile_size=16):
    mu_2d, Sigma_2d, depth = project_gaussian(...)
    order = torch.argsort(depth, descending=True)

    # 分tile
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size

    image = torch.zeros((3, H, W), device=gaussians.mu.device)
    accum_alpha = torch.zeros((1, H, W), device=gaussians.mu.device)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            # 该tile的像素范围
            x0, x1 = tx*tile_size, min((tx+1)*tile_size, W)
            y0, y1 = ty*tile_size, min((ty+1)*tile_size, H)

            # 预筛选：哪些高斯影响该tile？
            # 计算每个高斯的包围盒
            bbox_min = mu_2d - 3 * Sigma_2d.diagonal(dim1=1, dim2=2).sqrt()  # 简化
            bbox_max = mu_2d + 3 * Sigma_2d.diagonal(dim1=1, dim2=2).sqrt()

            in_tile = (bbox_min[:,0] < x1) & (bbox_max[:,0] > x0) & \
                      (bbox_min[:,1] < y1) & (bbox_max[:,1] > y0)

            tile_gaussians = order[in_tile[order]]

            # 对该tile的像素循环
            for y in range(y0, y1):
                for x in range(x0, x1):
                    pixel = torch.tensor([x, y], device=gaussians.mu.device)
                    for i in tile_gaussians:
                        # 计算高斯值（同上）
                        ...
```

### 6.2 性能检查清单

- [ ] 用 `torch.no_grad()` 包装推理部分
- [ ] 用 `torch.cuda.amp` 混合精度训练
- [ ] 避免循环中的 `.item()` 或 `.cpu()` 调用
- [ ] 用 `torch.compile`（PyTorch 2.0+）
- [ ] 高斯数据用 `contiguous()` 存储

---

## 阶段7：调试与验证（贯穿全程）

### 7.1 可视化检查点

```python
def debug_visualize(gaussians, camera, H, W, step):
    # 1. 渲染
    rendered = render_gaussians_tiled(gaussians, camera, H, W)

    # 2. 保存图像
    save_image(rendered, f"debug/step_{step:06d}.png")

    # 3. 可视化高斯位置（投影到图像）
    mu_2d, _, _ = project_gaussian(...)
    plt.scatter(mu_2d[:,0].cpu(), mu_2d[:,1].cpu(), s=1, alpha=0.5)
    plt.xlim(0, W); plt.ylim(H, 0)
    plt.savefig(f"debug/gaussians_{step:06d}.png")
    plt.close()
```

### 7.2 常见问题诊断

| 症状 | 可能原因 | 检查点 |
|------|----------|--------|
| 渲染全黑 | 高斯尺度太小 | 查看 `gaussians.Sigma.diagonal()` |
| 渲染全白 | α 太大或Σ太大 | 检查 `alpha.mean()`, `Sigma.mean()` |
| 噪点很多 | 高斯数量不足 | `len(gaussians)` 是否增长？ |
| 条纹伪影 | Σ 奇异（扁高斯） | `Sigma.det()` 是否接近0 |
| 训练不收敛 | 学习率太高 | loss是否 NaN 或爆炸 |
| 慢（O(NHW)） | 未用tile优化 | 确认用了 `render_gaussians_tiled` |

---

## 阶段8：完整实现路线图（总结）

```
Week 1:
  Day 1-2: 环境 + 数据加载（阶段0-1）
  Day 3-4: 投影 + 慢速渲染（阶段2-3），验证能出图
  Day 5-6: 训练循环 + 损失（阶段4），跑通100步

Week 2:
  Day 1-2: 密度控制（阶段5），高斯数量开始增长
  Day 3-4: Tile优化（阶段6），速度提升到可接受（<1s/帧）
  Day 5: 完整训练（30k步），评估PSNR

Week 3（可选）:
  - 多视角支持（每个epoch随机采相机）
  - 更快的CUDA kernel（如果GPU够强）
  - 部署到移动端（ONNX转换）
```

---

## 实践检查清单

在开始前，确认：

- [ ] 理解第3章的投影公式（自己推一遍）
- [ ] 理解第4章的Alpha Blending（写个2高斯例子）
- [ ] 理解第5章的损失函数（能手写L1+SSIM）
- [ ] 理解第7章的密度控制逻辑（能口述densify条件）
- [ ] 安装了PyTorch + 有GPU（可选但推荐）

---

## 最后提醒

- **先跑通，再优化**：阶段1-5的慢速版能出图，比优化但跑不通更重要
- **小数据开始**：用`nerf_synthetic/chair`（100张图，1000个SfM点）
- **可视化驱动**：每100步保存渲染图，肉眼观察进步
- **梯度监控**：打印 `mu.grad.norm()`，确保梯度不为0
- **社区资源**：参考官方实现（https://github.com/graphdeco-inria/gaussian-splatting）但**先自己写**

---

**现在，去烧起来吧！🔥** 遇到卡点，回到对应章节重推。
