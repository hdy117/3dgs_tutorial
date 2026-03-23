# 第10章：实践路径 - 从零到可运行的3DGS实现

**学习路径**：`example`（完整实践指南）

**核心目标**：在1-2周内，用PyTorch实现一个能跑通小数据集的简化版3DGS

---

## 一、实践路线图总览

```
Week 1: 基础框架
  Day 1-2: 环境配置 + 数据加载
  Day 3-4: 投影渲染（慢速版，先跑通）
  Day 5-6: 训练循环 + 损失函数（无密度控制）
  Day 7: 验证能出图（哪怕质量差）

Week 2: 完整流程
  Day 1-2: 密度控制（densify/prune）
  Day 3-4: Tile优化（速度提升到可接受）
  Day 5: 完整训练（30k步） + 评估
  Day 6-7: 调试优化 + 尝试不同数据集

Week 3（可选）:
  - 多视角支持完善
  - CUDA kernel优化
  - 部署到移动端
```

---

## 二、阶段0：环境准备（2-4小时）

### 2.1 系统要求

- **OS**：Linux / macOS / Windows（WSL2）
- **Python**：3.10+
- **GPU**：NVIDIA（推荐RTX 3060+），无GPU可用CPU（慢）
- **CUDA**：11.8+（如有GPU）

---

### 2.2 依赖安装

```bash
# 1. 创建虚拟环境
conda create -n 3dgs python=3.10 -y
conda activate 3dgs

# 2. 安装PyTorch（根据CUDA版本选择）
# 有GPU（CUDA 11.8）：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# 无GPU（CPU）：
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 3. 安装其他依赖
pip install tqdm opencv-python matplotlib numpy imageio scikit-image

# 4. 可选：评估指标
pip install lpips  # LPIPS感知损失
pip install torchmetrics  # SSIM等
```

---

### 2.3 数据集准备

**快速开始**：使用nerf_synthetic数据集（小型合成场景，100张图）

```bash
# 下载示例数据
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/dataset/nerf_synthetic.zip
unzip nerf_synthetic.zip -d data/

# 结构
data/nerf_synthetic/chair/
├── train/      # 训练图像（100张）
├── test/       # 测试图像（100张）
└── transforms.json  # NeRF格式的相机参数
```

**用自己的数据**：
- 用COLMAP处理你的照片集
- 输出格式：`cameras.bin`, `images.bin`, `points3D.bin`
- 见第6章详细说明

---

## 三、阶段1：数据加载模块（6-8小时）

**目标**：实现 `Dataset` 类，能加载COLMAP输出或 transforms.json

### 3.1 COLMAP解析器

```python
# utils/colmap_loader.py
import numpy as np
from pathlib import Path
import pycolmap  # pip install pycolmap

def load_sfm_reconstruction(sparse_path):
    """
    加载COLMAP reconstruction
    返回：recon对象（包含cameras, images, points3D）
    """
    recon = pycolmap.Reconstruction(str(sparse_path))
    return recon

def build_K(camera):
    """
    从COLMAP相机对象构建内参矩阵K
    """
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
    elif camera.model == "SIMPLE_PINHOLE":
        fx, cx, cy = camera.params
        K = np.array([[fx, 0, cx],
                      [0, fx, cy],
                      [0,  0,  1]])
    else:
        raise NotImplementedError(f"Camera model {camera.model} not supported")
    return K

def qvec2rotmat(qvec):
    """四元数(w,x,y,z) → 旋转矩阵"""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R
```

---

### 3.2 Dataset类

```python
# data/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import torch

class SfMDataset(Dataset):
    def __init__(self, sparse_path, image_path, split='train', image_scale=1.0):
        """
        Args:
            sparse_path: COLMAP sparse/0 目录
            image_path: 图像文件夹
            split: 'train' 或 'test'
            image_scale: 图像缩放比例（1.0=原尺寸，0.5=半尺寸）
        """
        self.recon = load_sfm_reconstruction(sparse_path)
        self.image_path = Path(image_path)
        self.image_scale = image_scale

        # 过滤训练/测试图像（简单：按文件名或随机）
        self.images = list(self.recon.images.values())
        if split == 'train':
            # 简单策略：取前80%
            self.images = self.images[:int(0.8*len(self.images))]
        else:
            self.images = self.images[int(0.8*len(self.images)):]

        # 提取点云（用于高斯初始化）
        self.points3d = list(self.recon.points3D.values())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        camera = self.recon.cameras[img.camera_id]

        # 1. 加载图像
        img_file = self.image_path / img.name
        image = Image.open(img_file).convert("RGB")

        # 缩放（加速训练）
        if self.image_scale != 1.0:
            new_size = (int(image.width * self.image_scale),
                        int(image.height * self.image_scale))
            image = image.resize(new_size, Image.LANCZOS)

        image = torch.from_numpy(np.array(image) / 255.0).float()
        image = image.permute(2, 0, 1)  # HWC → CHW

        # 2. 相机参数
        K = build_K(camera)
        R = qvec2rotmat(img.qvec)
        T = img.tvec

        # 3. 图像缩放调整K
        if self.image_scale != 1.0:
            K[0] *= self.image_scale
            K[1] *= self.image_scale

        # 4. 坐标系转换（如需要，见第6章）
        # R = R @ np.diag([1, -1, -1])

        return image, {
            'R': torch.from_numpy(R).float(),
            'T': torch.from_numpy(T).float(),
            'K': torch.from_numpy(K).float(),
            'width': int(camera.width * self.image_scale),
            'height': int(camera.height * self.image_scale),
            'name': img.name
        }
```

---

## 四、阶段2：高斯初始化（4-6小时）

**目标**：从SfM点云初始化高斯参数

### 4.1 GaussianModel类

```python
# gaussian/gaussian.py
import torch

class GaussianModel:
    def __init__(self, points3d, cameras, images, device='cuda'):
        """
        points3d: list of Point3D（从COLMAP加载）
        cameras: dict{camera_id: Camera}
        images: list of Image
        """
        self.N = len(points3d)
        self.device = torch.device(device)

        # 初始化参数张量
        self.mu = torch.zeros((self.N, 3), device=self.device)
        self.Sigma = torch.zeros((self.N, 3, 3), device=self.device)
        self.alpha = torch.full((self.N, 1), 0.5, device=self.device)
        self.color = torch.zeros((self.N, 3), device=self.device)

        # 逐点初始化
        for i, p in enumerate(points3d):
            self.mu[i] = torch.tensor(p.xyz, device=self.device, dtype=torch.float32)
            self.color[i] = torch.tensor(p.rgb / 255.0, device=self.device, dtype=torch.float32)

        # 估计尺度（从重投影误差）
        self._estimate_scales(points3d, cameras, images)

        # 初始化协方差（各向同性）
        scales = self.get_scales()  # (N, 3)
        for i in range(self.N):
            s = scales[i].clamp(min=1e-6)
            self.Sigma[i] = torch.diag(s**2)

    def _estimate_scales(self, points3d, cameras, images, scale_factor=0.5):
        """从重投影误差估计尺度"""
        scales = []
        for i, p in enumerate(points3d):
            errors = []
            for track in p.track:
                img = images[track.image_id]
                cam = cameras[img.camera_id]
                # 投影
                R = torch.from_numpy(qvec2rotmat(img.qvec)).float()
                T = torch.from_numpy(img.tvec).float()
                K = torch.from_numpy(build_K(cam)).float()
                mu = self.mu[i].cpu().numpy()
                X_cam = R @ mu + T
                proj = K @ X_cam
                proj = proj[:2] / proj[2]
                err = np.linalg.norm(proj - track.point2D)
                errors.append(err)
            scale = np.median(errors) if errors else 0.01
            scales.append(scale * scale_factor)

        self.scale_init = torch.tensor(scales, device=self.device)

    def get_scales(self):
        """从Σ提取各轴尺度（特征值开根）"""
        eigvals = torch.linalg.eigvalsh(self.Sigma)  # (N, 3)
        return torch.sqrt(eigvals)  # (N, 3)

    def to(self, device):
        self.mu = self.mu.to(device)
        self.Sigma = self.Sigma.to(device)
        self.alpha = self.alpha.to(device)
        self.color = self.color.to(device)
        self.device = device
        return self

    def half(self):
        self.mu = self.mu.half()
        self.Sigma = self.Sigma.half()
        self.alpha = self.alpha.half()
        self.color = self.color.half()
        return self

    def __len__(self):
        return self.N
```

---

### 4.2 初始化流程

```python
# 加载数据
dataset = SfMDataset(
    sparse_path="data/nerf_synthetic/chair/sparse/0",
    image_path="data/nerf_synthetic/chair/train",
    split='train',
    image_scale=0.5  # 用半尺寸加速
)

# 初始化高斯
gaussians = GaussianModel(
    points3d=dataset.points3d,
    cameras=dataset.recon.cameras,
    images=dataset.images,
    device='cuda'
)

print(f"Initialized {len(gaussians)} gaussians")
# 预期：5k-20k
```

---

## 五、阶段3：投影与渲染（8-10小时）

**目标**：实现第4章的渲染管线（慢速版先跑通）

### 5.1 投影函数

```python
# rendering/projection.py
import torch

def project_gaussian(mu, Sigma, R, T, K):
    """
    将3D高斯投影到2D
    输入：
      mu: (N, 3) 世界坐标
      Sigma: (N, 3, 3) 协方差
      R: (3,3) 相机外参旋转
      T: (3,) 平移
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

    # 3. 投影协方差（简化版：用雅可比）
    Sigma_cam = R @ Sigma @ R.T  # (N, 3, 3)

    # 雅可比（透视投影）
    z = mu_cam[:, 2].clamp(min=1e-6)  # (N,)
    J = torch.zeros((mu.shape[0], 2, 3), device=mu.device)
    J[:, 0, 0] = K[0, 0] / z
    J[:, 0, 2] = -K[0, 0] * mu_cam[:, 0] / (z**2)
    J[:, 1, 1] = K[1, 1] / z
    J[:, 1, 2] = -K[1, 1] * mu_cam[:, 1] / (z**2)

    Sigma_2d = J @ Sigma_cam @ J.transpose(1, 2)  # (N, 2, 2)

    # 数值稳定性：确保正定
    Sigma_2d = Sigma_2d + torch.eye(2, device=mu.device)[None, :, :] * 1e-8

    return mu_2d, Sigma_2d, depth
```

---

### 5.2 慢速渲染（用于调试）

**重要**：先实现O(N·H·W)的慢速版，验证正确性，再优化

```python
# rendering/render_slow.py
def render_gaussians_slow(gaussians, camera, H, W):
    """
    慢速渲染（O(N·H·W)），仅用于调试验证
    """
    mu_2d, Sigma_2d, depth = project_gaussian(
        gaussians.mu, gaussians.Sigma,
        camera['R'], camera['T'], camera['K']
    )

    # 按深度排序
    indices = torch.argsort(depth, descending=True)

    mu_2d = mu_2d[indices]
    Sigma_2d = Sigma_2d[indices]
    alpha = gaussians.alpha[indices]
    color = gaussians.color[indices]

    # 创建图像缓冲区
    image = torch.zeros((3, H, W), device=gaussians.mu.device)
    accum_alpha = torch.zeros((1, H, W), device=gaussians.mu.device)

    # 生成像素网格
    y, x = torch.meshgrid(
        torch.arange(H, device=gaussians.mu.device),
        torch.arange(W, device=gaussians.mu.device),
        indexing='ij'
    )
    pixels = torch.stack([x.float(), y.float()], dim=-1)  # (H, W, 2)

    # 遍历所有高斯（极慢！）
    for i in range(len(gaussians)):
        mu_i = mu_2d[i]  # (2,)
        Sigma_i = Sigma_2d[i]  # (2,2)

        # 计算该高斯对所有像素的贡献
        diff = pixels - mu_i[None, None, :]  # (H, W, 2)
        inv_Sigma = torch.linalg.inv(Sigma_i)  # (2,2)
        exponent = -0.5 * (diff @ inv_Sigma * diff).sum(dim=2)
        gauss_val = alpha[i] * torch.exp(exponent)  # (H, W)

        # Alpha blending
        contrib = gauss_val[None, :, :] * color[i][:, None, None] * (1 - accum_alpha)
        image += contrib
        accum_alpha += gauss_val[None, :, :]

        if accum_alpha.max() > 0.99:
            break

    return image
```

**调试方法**：
```python
# 测试渲染一帧
sample_img, sample_camera = dataset[0]
rendered = render_gaussians_slow(gaussians, sample_camera, H, W)

# 可视化对比
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2)
axes[0].imshow(sample_img.permute(1,2,0).cpu().numpy())
axes[0].set_title("GT")
axes[1].imshow(rendered.permute(1,2,0).cpu().numpy().clip(0,1))
axes[1].set_title("Rendered")
plt.show()
```

**预期**：
- 初始渲染可能全黑或噪点多（正常）
- 如果全黑：检查高斯尺度是否太小（`gaussians.Sigma.diagonal().mean()`）
- 如果全白：检查α是否太大

---

## 六、阶段4：训练循环（无密度控制）（6-8小时）

**目标**：跑通1000步训练，损失下降

### 6.1 损失函数

```python
# training/loss.py
import torch
import torch.nn.functional as F

def ms_ssim_loss(pred, target):
    """简化版单尺度SSIM"""
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

def compute_loss(rendered, gt, gaussians, λ_ssim=0.8, λ_scale=0.01):
    L1 = F.l1_loss(rendered, gt)
    L_ssim = ms_ssim_loss(rendered, gt)
    L_img = (1 - λ_ssim) * L1 + λ_ssim * L_ssim

    # 尺度正则
    scales = gaussians.get_scales()  # (N, 3)
    L_scale = torch.clamp(scales - 1.0, min=0).mean()

    L_total = L_img + λ_scale * L_scale
    return L_total, L1, L_ssim
```

---

### 6.2 训练循环（简化版）

```python
# train_simple.py
from torch.utils.data import DataLoader
from tqdm import tqdm

# 超参数
num_epochs = 30
steps_per_epoch = 100
batch_size = 1

# 优化器（分组LR）
optimizer = torch.optim.Adam([
    {'params': gaussians.mu, 'lr': 1.6e-4},
    {'params': gaussians.Sigma, 'lr': 1e-3},
    {'params': gaussians.alpha, 'lr': 5e-2},
    {'params': gaussians.color, 'lr': 5e-3}
])

# 训练
for epoch in range(num_epochs):
    pbar = tqdm(range(steps_per_epoch))
    for step in pbar:
        # 1. 采样
        idx = np.random.randint(len(dataset))
        gt_image, camera = dataset[idx]
        gt_image = gt_image.cuda()
        camera = {k: v.cuda() for k, v in camera.items()}

        # 2. 渲染（慢速版）
        rendered = render_gaussians_slow(gaussians, camera, camera['height'], camera['width'])

        # 3. 损失
        loss, L1, L_ssim = compute_loss(rendered, gt_image, gaussians)

        # 4. 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": loss.item(), "PSNR": 10*np.log10(1/L1.item())})

    # 每epoch保存
    save_image(rendered, f"output/epoch_{epoch:03d}.png")
```

**预期**：
- 前100步：损失快速下降
- 1000步后：PSNR达到15-20（小数据集）
- 如果损失不降：检查梯度（`gaussians.mu.grad.norm()`）

---

## 七、阶段5：密度控制（8-10小时）

**目标**：实现densify和prune，让高斯数量动态增长

### 7.1 梯度缓存

```python
def cache_gradients(gaussians):
    """缓存梯度幅度"""
    grads_mu = gaussians.mu.grad.detach().norm(dim=1)  # (N,)
    grads_Sigma = gaussians.Sigma.grad.detach()
    grads_Sigma = grads_Sigma.view(gaussians.N, -1).norm(dim=1)  # (N,)
    return grads_mu, grads_Sigma
```

---

### 7.2 Densify & Prune

```python
def densify_and_prune(gaussians, optimizer, radii,
                      grads_mu, grads_Sigma,
                      grad_threshold=0.0002,
                      scale_threshold=0.01,
                      prune_alpha=0.001,
                      max_gaussians=2e6):
    with torch.no_grad():
        # Densify条件
        large_scale = radii > scale_threshold
        large_grad = (grads_mu > grad_threshold) | (grads_Sigma > grad_threshold)
        to_densify = large_scale & large_grad

        if to_densify.any():
            # 克隆（简化：不分裂）
            new_mu = gaussians.mu[to_densify].clone()
            new_Sigma = gaussians.Sigma[to_densify].clone()
            new_alpha = gaussians.alpha[to_densify].clone()
            new_color = gaussians.color[to_densify].clone()

            gaussians.mu = torch.cat([gaussians.mu, new_mu])
            gaussians.Sigma = torch.cat([gaussians.Sigma, new_Sigma])
            gaussians.alpha = torch.cat([gaussians.alpha, new_alpha])
            gaussians.color = torch.cat([gaussians.color, new_color])
            gaussians.N = len(gaussians)

            # 优化器添加新参数
            optimizer.add_param_group({'params': new_mu, 'lr': 1.6e-4})
            optimizer.add_param_group({'params': new_Sigma, 'lr': 1e-3})
            optimizer.add_param_group({'params': new_alpha, 'lr': 5e-2})
            optimizer.add_param_group({'params': new_color, 'lr': 5e-3})

        # Prune
        scales = gaussians.get_scales().max(dim=1)[0]
        to_prune = (gaussians.alpha.squeeze() < prune_alpha) | (scales < 1e-6)

        if to_prune.any():
            keep_mask = ~to_prune
            gaussians.mu = gaussians.mu[keep_mask]
            gaussians.Sigma = gaussians.Sigma[keep_mask]
            gaussians.alpha = gaussians.alpha[keep_mask]
            gaussians.color = gaussians.color[keep_mask]
            gaussians.N = len(gaussians)

            # 优化器prune（简化：重建param_groups）
            optimizer.param_groups = [g for g in optimizer.param_groups if len(g['params']) > 0]

        # 数量上限
        if len(gaussians) > max_gaussians:
            keep = torch.randperm(len(gaussians))[:int(max_gaussians)]
            gaussians.mu = gaussians.mu[keep]
            gaussians.Sigma = gaussians.Sigma[keep]
            gaussians.alpha = gaussians.alpha[keep]
            gaussians.color = gaussians.color[keep]
            gaussians.N = len(gaussians)
```

---

### 7.3 整合到训练循环

```python
# 密度控制参数
densify_interval = 1000
densify_from = 500
prune_from = 1500

for step in range(total_steps):
    # ... 渲染、损失、反向 ...

    # 缓存梯度（densify用）
    if step < densify_from or step % densify_interval == 0:
        grads_mu, grads_Sigma = cache_gradients(gaussians)

    # 密度控制
    if densify_from <= step < prune_from and step % densify_interval == 0:
        densify_and_prune(gaussians, optimizer, radii,
                          grads_mu, grads_Sigma)
    if step >= prune_from and step % densify_interval == 0:
        densify_and_prune(gaussians, optimizer, radii,
                          grads_mu, grads_Sigma, prune_alpha=0.001)

    # 学习率调度
    if step in [7500, 15000]:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
```

**预期**：
- 高斯数量从10k增长到100k-500k
- PSNR继续提升

---

## 八、阶段6：Tile优化（8-12小时）

**目标**：将渲染速度从秒级降到毫秒级

### 8.1 优化思路

1. **预筛选**：只处理影响tile的高斯
2. **tile并行**：每个tile独立计算
3. **shared memory**：缓存tile内高斯

### 8.2 Tile渲染实现

```python
# rendering/render_tiled.py
def compute_bbox(mu_2d, Sigma_2d, scale=3.0):
    """计算高斯的屏幕包围盒"""
    eigvals = torch.linalg.eigvalsh(Sigma_2d)  # (N, 2)
    radii = scale * torch.sqrt(eigvals.max(dim=1)[0])  # (N,)
    bbox_min = (mu_2d - radii[:, None]).long().clamp(min=0)
    bbox_max = (mu_2d + radii[:, None]).long().clamp(min=0)
    return bbox_min, bbox_max  # (N,2)

def assign_gaussians_to_tiles(bbox_min, bbox_max, tile_size=16, W=800, H=600):
    """将高斯分配给tile"""
    n_tiles_x = (W + tile_size - 1) // tile_size
    n_tiles_y = (H + tile_size - 1) // tile_size
    tile_mapping = [[] for _ in range(n_tiles_x * n_tiles_y)]

    for g_idx in range(len(bbox_min)):
        x0, y0 = bbox_min[g_idx]
        x1, y1 = bbox_max[g_idx]
        # 该高斯影响的所有tile
        tile_x0 = x0 // tile_size
        tile_x1 = x1 // tile_size
        tile_y0 = y0 // tile_size
        tile_y1 = y1 // tile_size
        for ty in range(tile_y0, tile_y1+1):
            for tx in range(tile_x0, tile_x1+1):
                tile_id = ty * n_tiles_x + tx
                if 0 <= tile_id < len(tile_mapping):
                    tile_mapping[tile_id].append(g_idx)

    return tile_mapping

def render_tiled(gaussians, sorted_indices, mu_2d, Sigma_2d, tile_mapping, H, W):
    """Tile-based渲染（简化CPU版）"""
    image = torch.zeros((3, H, W), device=gaussians.mu.device)
    accum_alpha = torch.zeros((1, H, W), device=gaussians.mu.device)

    tile_size = 16
    n_tiles_x = (W + tile_size - 1) // tile_size

    for tile_id, g_indices in enumerate(tile_mapping):
        if not g_indices:
            continue

        ty = tile_id // n_tiles_x
        tx = tile_id % n_tiles_x
        x0, x1 = tx*tile_size, min((tx+1)*tile_size, W)
        y0, y1 = ty*tile_size, min((ty+1)*tile_size, H)

        # 遍历该tile的像素
        for y in range(y0, y1):
            for x in range(x0, x1):
                pixel = torch.tensor([x, y], device=gaussians.mu.device)
                for g_idx in g_indices:
                    g_idx_sorted = sorted_indices[g_idx]
                    mu_i = mu_2d[g_idx_sorted]
                    Sigma_i = Sigma_2d[g_idx_sorted]
                    alpha_i = gaussians.alpha[g_idx_sorted]
                    color_i = gaussians.color[g_idx_sorted]

                    # 评估2D高斯
                    diff = pixel - mu_i
                    inv_Sigma = torch.linalg.inv(Sigma_i)
                    exponent = -0.5 * (diff @ inv_Sigma @ diff)
                    g_val = alpha_i * torch.exp(exponent)

                    # Alpha blending
                    contrib = g_val * color_i * (1 - accum_alpha[:, y, x])
                    image[:, y, x] += contrib.squeeze()
                    accum_alpha[:, y, x] += g_val

                    if accum_alpha[0, y, x] >= 0.99:
                        break

    return image
```

**性能**：
- 慢速版：O(N·H·W) → 100高斯 + 100×100图像 ≈ 1秒
- Tile版：O(N·窗口) → 同样配置 ≈ 10ms（提升100倍）

---

## 九、阶段7：完整训练与评估（持续）

### 9.1 完整训练脚本

```python
# train_full.py
from torch.utils.data import DataLoader
import numpy as np

# 超参数
total_steps = 30000
batch_size = 1
log_interval = 100
save_interval = 5000

# 数据加载器（打乱）
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_iter = iter(dataloader)

for step in range(total_steps):
    try:
        gt_image, camera = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        gt_image, camera = next(data_iter)

    gt_image = gt_image[0].cuda()  # batch=1
    camera = {k: v[0].cuda() for k, v in camera.items()}

    # 渲染（tile版）
    rendered, radii = render_gaussians_tiled(gaussians, camera, camera['height'], camera['width'])

    # 损失
    loss, L1, L_ssim = compute_loss(rendered, gt_image, gaussians)

    # 反向
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 缓存梯度
    if step % densify_interval == 0:
        grads_mu = gaussians.mu.grad.detach().norm(dim=1)
        grads_Sigma = gaussians.Sigma.grad.detach().view(gaussians.N, -1).norm(dim=1)

    # 密度控制
    if step % densify_interval == 0:
        if step < prune_from:
            densify_and_prune(gaussians, optimizer, radii, grads_mu, grads_Sigma)
        else:
            densify_and_prune(gaussians, optimizer, radii, grads_mu, grads_Sigma, prune_alpha=0.001)

    # 学习率调度
    if step in [7500, 15000]:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1

    # 日志
    if step % log_interval == 0:
        psnr_val = 10 * np.log10(1.0 / L1.item())
        print(f"Step {step:06d}: loss={loss:.4f}, PSNR={psnr_val:.2f}, "
              f"#gauss={len(gaussians)}, lr={optimizer.param_groups[0]['lr']:.2e}")

    # 保存
    if step % save_interval == 0:
        save_image(rendered, f"output/step_{step:06d}.png")
        save_gaussians_ply(gaussians, f"output/step_{step:06d}.ply")
```

---

### 9.2 评估

```python
def evaluate(gaussians, test_dataset):
    """在测试集上评估PSNR/SSIM"""
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            gt, camera = test_dataset[i]
            gt = gt.cuda()
            camera = {k: v.cuda() for k, v in camera.items()}
            rendered = render_gaussians_tiled(gaussians, camera, camera['height'], camera['width'])

            L1 = F.l1_loss(rendered, gt)
            psnr = 10 * torch.log10(1.0 / L1)
            psnr_list.append(psnr.item())

            # SSIM（如已安装torchmetrics）
            # ssim = structural_similarity_index_measure(rendered[None], gt[None])
            # ssim_list.append(ssim.item())

    return np.mean(psnr_list), np.mean(ssim_list) if ssim_list else None
```

---

## 十、调试与验证清单

### 10.1 阶段性检查点

| 阶段 | 目标 | 判断标准 |
|------|------|----------|
| 阶段1（数据加载） | 能读取图像和相机 | `dataset[0]` 返回合理张量 |
| 阶段2（初始化） | 高斯参数正确 | `gaussians.mu.shape == (N,3)` |
| 阶段3（慢速渲染） | 能出图 | 保存 `rendered.png` 不是全黑 |
| 阶段4（训练100步） | 损失下降 | loss从~1.0降到~0.5 |
| 阶段5（densify） | 高斯数量增长 | 从10k→50k |
| 阶段6（tile优化） | 单帧<100ms | `time.time()` 测量 |
| 阶段7（完整训练） | PSNR>25 | 测试集评估 |

---

### 10.2 常见问题速查

| 症状 | 可能原因 | 快速测试 | 解决 |
|------|----------|----------|------|
| 渲染全黑 | Σ太小 | `Sigma.diag().mean()` | 增大scale_factor |
| 渲染全白 | α太大 | `alpha.mean()` | 减小α初始值 |
| 梯度为0 | 高斯"死"了 | `mu.grad.norm()` | 检查α初始化 |
| 条纹伪影 | Σ奇异 | `torch.det(Sigma)` | 添加正则化 |
| 不收敛 | LR太高 | loss NaN/爆炸 | 降低所有LR 10倍 |
| 内存爆炸 | 高斯太多 | `len(gaussians)` | 提高prune阈值 |
| 速度慢 | 未用tile | 单帧>1s | 实现tile渲染 |

---

### 10.3 可视化调试

```python
def debug_visualize(gaussians, camera, step):
    """保存调试图像"""
    # 1. 渲染
    rendered = render_gaussians_tiled(gaussians, camera, H, W)

    # 2. 投影高斯中心
    mu_2d, _, _ = project_gaussian(gaussians.mu, gaussians.Sigma,
                                    camera['R'], camera['T'], camera['K'])

    # 3. 绘制
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(rendered.permute(1,2,0).cpu().numpy().clip(0,1))
    plt.title("Rendered")
    plt.subplot(132)
    plt.scatter(mu_2d[:,0].cpu(), mu_2d[:,1].cpu(), s=1, alpha=0.5)
    plt.xlim(0, W); plt.ylim(H, 0)
    plt.title(f"Gaussians ({len(gaussians)})")
    plt.subplot(133)
    scales = gaussians.get_scales().max(dim=1)[0].cpu()
    plt.hist(scales.numpy(), bins=50)
    plt.title("Scale distribution")
    plt.tight_layout()
    plt.savefig(f"debug/step_{step:06d}.png")
    plt.close()
```

---

## 十一、实现检查清单（完成度自评）

### 11.1 核心功能

- [ ] 数据加载（COLMAP / transforms.json）
- [ ] 高斯初始化（μ, Σ, α, c）
- [ ] 投影公式（透视 + 雅可比）
- [ ] 慢速渲染（验证正确性）
- [ ] Alpha Blending（排序 + 累加）
- [ ] 损失函数（L1 + SSIM + 正则化）
- [ ] 优化器（分组LR）
- [ ] 密度控制（densify + prune）
- [ ] Tile优化（预筛选 + 并行）
- [ ] 学习率调度（三阶段）

---

### 11.2 性能优化

- [ ] `torch.no_grad()` 推理
- [ ] float16 精度
- [ ] 梯度检查点（如内存不足）
- [ ] CUDA kernel融合（进阶）

---

### 11.3 工程化

- [ ] 配置文件（YAML/JSON）
- [ ] 日志系统（TensorBoard/WandB）
- [ ] 断点恢复（checkpoint）
- [ ] 多GPU支持（DDP）
- [ ] 模型导出（PLY/JSON）

---

## 十二、后续学习路径

完成基础实现后，可以：

1. **阅读官方代码**（https://github.com/graphdeco-inria/gaussian-splatting）
   - 对比自己的实现，学习工程技巧
   - 理解CUDA kernel优化

2. **尝试扩展方向**（第9章）
   - 动态场景：4D Gaussian
   - 压缩：量化 + 流式
   - 几何质量：各向同性约束

3. **应用到自己的数据**
   - 用手机拍摄场景
   - COLMAP SfM → 3DGS训练
   - 实时渲染展示

4. **贡献社区**
   - Bug fix / 性能优化
   - 新功能实现
   - 写博客分享经验

---

## 十三、最后提醒

- **先跑通，再优化**：阶段1-5的慢速版能出图，比优化但跑不通更重要
- **小数据开始**：用 `nerf_synthetic/chair`（100张图，1000个SfM点）
- **可视化驱动**：每100步保存渲染图，肉眼观察进步
- **梯度监控**：打印 `mu.grad.norm()`，确保梯度不为0
- **社区资源**：参考官方实现但**先自己写**

---

**现在，去烧起来吧！🔥** 遇到卡点，回到对应章节重推。
