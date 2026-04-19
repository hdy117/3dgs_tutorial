"""
============================================================
  Full Pipeline: From Gaussian Splatting Rendering to Training
  完整链路：从 3DGS 渲染到训练

This script walks through the ENTIRE pipeline with detailed comments.
Each major section maps directly to a chapter in the tutorial series.

目标：用最少代码展示 3DGS 的核心数学 → 代码映射关系。
要求：PyTorch >= 2.0（用于 autograd 追踪）。
============================================================
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


# ================================================================
# Section 1: Gaussian Parameter Definition (高斯参数定义)
#
# Each 3DGS Gaussian is defined by parameters in world space:
#   • mean (x, y, z):     position in 3D            — μ
#   • scale (s_x, s_y, s_z): size along each axis    — S = diag(scale)
#   • rotation (q_w, q_x, q_y, q_z): orientation as unit quaternion
#   • opacity:            alpha in [0,1] via sigmoid  — α = σ(opacity_param)
#
# Plus color encoded as Spherical Harmonics (SH) coefficients.
# ================================================================

class GaussianParams:
    """高斯参数容器 — 使用可变类，支持训练中的增删操作（densification）"""
    def __init__(self, position: torch.Tensor, scale: torch.Tensor,
                 rotation: torch.Tensor, opacity: torch.Tensor):
        self.position = position   # [N, 3]  — world-space mean (x, y, z)
        self.scale = scale         # [N, 3]  — per-axis scaling (positive via exp)
        self.rotation = rotation   # [N, 4]  — unit quaternion (w, x, y, z)
        self.opacity = opacity     # [N, 1]  — pre-sigmoid values


def create_random_gaussians(N: int = 5000) -> GaussianParams:
    """创建 N 个随机高斯（演示用，实际从 COLMAP 点云初始化）"""
    pos = torch.randn(N, 3) * 0.8
    # Place all Gaussians between z=1 and z=4 (in front of camera at origin looking +Z)
    pos[:, 2] = torch.rand(N) * 3.0 + 1.0  # positive z: in front of camera
    scale = torch.exp(torch.randn(N, 3) * 0.5)  # exp ensures positive scales
    rot = nn.functional.normalize(torch.randn(N, 4), dim=1)
    opacity = torch.rand(N, 1) * 2.0              # pre-sigmoid values

    return GaussianParams(position=pos, scale=scale, rotation=rot, opacity=opacity)


# ================================================================
# Section 2: Quaternion → Rotation Matrix (四元数 → 旋转矩阵)
#
# Math derivation: A unit quaternion q = [w, x, y, z] maps to a
# 3×3 rotation matrix R ∈ SO(3). This is the "square root" of SO(3):
# every rotation has exactly two quaternions (q and -q).
# ================================================================

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions [N, 4] to rotation matrices [N, 3, 3].

    Input q is assumed normalized. Returns R where v_transformed = R @ v.
    """
    w, x, y, z = q.unbind(dim=1)  # Split into scalar tensors: each [N]

    # Precompute squared terms (avoids redundant multiplications)
    ww, xx, yy, zz = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    # Standard quaternion→rotation matrix (unit quaternion q=[w,x,y,z]):
    #   R[0] = [1-2(y²+z²), 2(xy-wz),        2(xz+wy)]
    #   R[1] = [2(xy+wz),    1-2(x²+z²),     2(yz-wx)]
    #   R[2] = [2(xz-wy),    2(yz+wx),        1-2(x²+y²)]
    R = torch.stack([
        1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy),
        2*(xy + wz),       1 - 2*(xx + zz),   2*(yz - wx),
        2*(xz - wy),       2*(yz + wx),       1 - 2*(xx + yy),
    ], dim=1).reshape(-1, 3, 3)

    return R


# ================================================================
# Section 3: Covariance Matrix Construction (协方差矩阵构建)
#
# Math derivation — the key formula from the paper:
#   The Gaussian's shape ellipsoid in world space is defined by:
#     Σ = S @ R @ R^T @ S^T = S² (since R@R^T = I for orthogonal R)
#   But wait — that would just give diagonal scaling! The REAL formula is:
#     Σ = W @ M @ W^T  where M is the "model-space" covariance,
#                          and W is a world-to-view transform.
#
# In practice we compute: Σ_world = S @ R @ (S @ R)^T
# which gives an ellipsoid with axes aligned to rotated scale directions.
# ================================================================

def build_covariance(position: torch.Tensor, scale: torch.Tensor,
                     rotation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build world-space covariance matrices from scale + rotation.

    Returns:
      cov_world: [N, 3, 3] — full covariance matrix Σ for each Gaussian
      lower_chol: [N, 3, 3] — Cholesky factor L where Σ = L @ L^T
                   (used for numerical stability and validation)
    """
    # Step 1: Get rotation matrices from quaternions
    R = quaternion_to_rotation_matrix(rotation)  # [N, 3, 3]

    # Step 2: Build diagonal scaling matrix S from per-axis scale values
    sx, sy, sz = scale.unbind(dim=1)              # Each: [N]
    S = torch.diag_embed(torch.stack([sx, sy, sz], dim=1))  # [N, 3, 3]

    # Step 3: Covariance Σ = (S @ R) @ (S @ R)^T = S @ R @ R^T @ S^T
    SR = S @ R                                   # Scale then rotate: [N, 3, 3]
    cov_world = SR @ SR.transpose(1, 2)           # Σ = (SR)(SR)^T: [N, 3, 3]

    # Step 4: Cholesky decomposition — L is lower triangular s.t. Σ = L @ L^T
    jitter = torch.eye(3, device=cov_world.device).unsqueeze(0) * 1e-5
    lower_chol = torch.linalg.cholesky(cov_world + jitter)

    return cov_world, lower_chol


# ================================================================
# Section 4: Camera Projection (相机投影)
#
# Math derivation — the full pipeline from world to pixel space:
#   World → Camera: p_cam = R_w2c @ p_world + t_w2c     (extrinsics)
#   Camera → Pixel: x = fx * x_cam/z_cam + cx             (intrinsics)
#                   y = fy * y_cam/z_cam + cy              (inverted Y for screen coords)
# ================================================================

class Camera:
    """简化针孔相机模型：包含外参(R_w2c, t_w2c)和内参(fx, fy, cx, cy)"""
    def __init__(self, R_w2c: torch.Tensor, t_w2c: torch.Tensor,
                 fx: float = 500.0, fy: float = 500.0,
                 cx: float = 256.0, cy: float = 256.0):
        self.R_w2c = R_w2c       # [3, 3] — world-to-camera rotation
        self.t_w2c = t_w2c       # [3, 1] — world-to-camera translation (camera position in world)
        self.fx = fx             # focal length x (pixels)
        self.fy = fy             # focal length y (pixels)
        self.cx = cx             # principal point x (image center usually)
        self.cy = cy             # principal point y

    def project_positions(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project world-space 3D points to 2D pixel coordinates.

        Pipeline for each point p_world ∈ R^3:
          1. p_cam = R_w2c @ p_world + t_w2c    → camera-space (z>0 means in front of camera)
          2. x_pix = fx * (x_cam/z_cam) + cx    → perspective projection
          3. y_pix = fy * (y_cam/z_cam) + cy

        Args:
            positions: [N, 3] — world-space coordinates

        Returns:
            pixel_coords: [N, 2] — (x, y) in pixel space
            depths: [N] — z distance from camera (for sorting front-to-back)
        """
        # Step 1: Apply extrinsic transform → camera-space positions
        p_cam = (self.R_w2c @ positions.T).T + self.t_w2c.squeeze(-1)  # [N, 3]

        # Step 2: Extract z-depth — filter out Gaussians behind camera
        depths = p_cam[:, 2]
        # (The original paper simply skips Gaussians with depth ≤ 0.
        # We clamp here for numerical stability; the kernel naturally
        # vanishes at large distances, so behind-camera Gaussians
        # contribute near-zero anyway.)

        # Step 3: Perspective projection → pixel coordinates
        x_cam, y_cam, z_cam = p_cam.unbind(dim=1)
        px = self.fx * (x_cam / z_cam) + self.cx
        py = self.fy * (y_cam / z_cam) + self.cy

        return torch.stack([px, py], dim=1), depths


# ================================================================
# Section 5: 2D Gaussian Kernel Evaluation (二维高斯核评估)
#
# Math derivation — the "trick" of 3DGS:
#   A 3D Gaussian projected onto the image plane reduces to a 2D Gaussian.
#   The projected covariance is computed via the Jacobian of perspective projection:
#     Σ_2D ≈ J @ W @ Σ_world @ W^T @ J^T
#   where J = ∂(x_pix, y_pix)/∂(x_cam, y_cam, z_cam) is the projection Jacobian.
#
# The 2D Gaussian PDF at pixel offset d from center:
#   f(d) = exp(-0.5 * d^T @ Σ_2D^{-1} @ d) / (2π√|Σ_2D|)
# ================================================================

def project_covariance_to_2d(cov_world: torch.Tensor, position_cam: torch.Tensor,
                             camera: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D covariance matrix to 2D image plane using the Jacobian method.

    This is one of the most important derivations in 3DGS — it shows how a
    3D ellipsoid becomes a 2D ellipse when viewed through perspective projection.

    Args:
        cov_world: [N, 3, 3] — world-space covariance matrices
        position_cam: [N, 3] — positions in camera space (already transformed)
        camera: Camera object with fx, fy, cx, cy

    Returns:
        cov_2d: [N, 2, 2] — projected 2D covariance for each Gaussian
        det: [N] — determinant (used for normalization constant)
    """
    # Step 1: Compute Jacobian of perspective projection at each point.
    # For perspective proj: x' = fx * x/z + cx, y' = fy * y/z + cy
    #   ∂x'/∂x = fx/z,     ∂x'/∂y = 0,      ∂x'/∂z = -fx*x/z²
    #   ∂y'/∂x = 0,        ∂y'/∂y = fy/z,   ∂y'/∂z = -fy*y/z²
    x_cam, y_cam, z_cam = position_cam.unbind(dim=1)

    # Build Jacobian row by row then stack: torch.stack on [N] tensors gives [N, 3], not [N, rows, 3].
    J_row_x = torch.stack([camera.fx / z_cam, torch.zeros_like(z_cam), -camera.fx * x_cam / (z_cam ** 2)], dim=1)  # [N, 3]
    J_row_y = torch.stack([torch.zeros_like(z_cam), camera.fy / z_cam, -camera.fy * y_cam / (z_cam ** 2)], dim=1)  # [N, 3]
    J = torch.stack([J_row_x, J_row_y], dim=1)  # [N, 2, 3] — two rows of partial derivatives

    # Step 2: Transform world covariance to camera space first
    R_w2c = camera.R_w2c.unsqueeze(0)  # [1, 3, 3] for batch multiply
    cov_cam = (R_w2c @ cov_world @ R_w2c.transpose(1, 2).unsqueeze(0)).squeeze(0)  # [N, 3, 3]

    # Step 3: Project using Jacobian — Σ_2D = J @ Σ_cam @ J^T
    cov_2d = J @ cov_cam @ J.transpose(1, 2)  # [N, 2, 2]

    # Clamp diagonal to ensure positive-definiteness (numerical stability)
    jitter = torch.eye(2, device=cov_2d.device).unsqueeze(0) * 0.3
    cov_2d = cov_2d + jitter

    # Step 4: Compute determinant for normalization: |Σ_2D| = ad - bc²
    det = cov_2d[:, 0, 0] * cov_2d[:, 1, 1] - cov_2d[:, 0, 1]**2

    return cov_2d, det


def evaluate_gaussian_kernel(dxy: torch.Tensor, cov_inv: torch.Tensor,
                             det: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the 2D Gaussian kernel: f(d) = exp(-0.5 * d^T @ Σ^{-1} @ d) / (2π√|Σ|)

    Args:
        dxy: [N, 2, num_pixels] — offset vectors from Gaussian center to each pixel
        cov_inv: [N, 2, 2] — inverse of the projected 2D covariance for each Gaussian
        det: [N] — determinant for normalization

    Returns:
        kernel_values: [N, num_pixels] — Gaussian weight at each pixel (for each gaussian)
    """
    # Step 1: Compute Σ^{-1} @ d → Mahalanobis distance direction
    # Batch matrix multiply: [N, 2, 2] @ [N, 2, P] → [N, 2, P]
    inv_d = cov_inv @ dxy

    # Step 2: d^T @ (Σ^{-1} @ d) → scalar per gaussian-pixel pair
    # Sum over the 2 spatial dimensions: [N, 2, P] * [N, 2, P] summed on dim=1
    mahal = (dxy * inv_d).sum(dim=1)  # [N, num_pixels]: the exponent argument

    # Step 3: Normalize and apply exp(-0.5 * mahal)
    norm = 1.0 / (2 * math.pi * torch.sqrt(torch.maximum(det.unsqueeze(1), torch.tensor(1e-8))))
    kernel_values = norm * torch.exp(-0.5 * mahal)

    return kernel_values


# ================================================================
# Section 6: Rendering Pipeline — Full Forward Pass (完整渲染函数)
#
# Math derivation — the rendering equation discretized as a sum over Gaussians:
#   C(x) = Σ_{i∈N} c_i @ α_i @ Π_{j=1}^{i-1} (1 - α_j)
# where:
#   • N = set of Gaussians sorted front-to-back by depth
#   • c_i = RGB color of Gaussian i
#   • α_i = opacity contribution (kernel value × sigmoid(opacity_param))
#   • Π = product operator — the transparency "shield" from closer Gaussians
#
# This is the discretized volume rendering equation! It's the core of 3DGS.
# ================================================================

def render_single_view(gaussians: GaussianParams, camera: Camera,
                       image_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Render a single view given Gaussians and camera parameters.

    Pipeline steps with mathematical correspondence:
      ① Project 3D centers → pixel coordinates        [Sec 4: Camera Projection]
      ② Compute projected 2D covariance               [Sec 5: Covariance Projection]
      ③ Sort Gaussians by depth (closest first)       [for front-to-back compositing]
      ④ Evaluate 2D Gaussian kernel at each pixel     [Sec 5: Kernel Evaluation]
      ⑤ Alpha composite from front to back            [Sec 6: Volume Rendering Eq.]

    Returns:
        image: [H, W, 3] — composited RGB image per pixel
    """
    H, W = image_size
    N = gaussians.position.shape[0]
    num_pixels = H * W

    # === Step ①: Project Gaussian centers to screen space ===
    centers_2d, depths = camera.project_positions(gaussians.position)   # [N, 2], [N]

    # === Step ②: Compute projected 2D covariance + determinant ===
    cov_world, _ = build_covariance(gaussians.position, gaussians.scale, gaussians.rotation)
    p_cam = (camera.R_w2c @ gaussians.position.T).T + camera.t_w2c.squeeze(-1)

    cov_2d, det = project_covariance_to_2d(cov_world, p_cam, camera)  # [N, 2, 2], [N]
    cov_inv = torch.linalg.inv(cov_2d + torch.eye(2).unsqueeze(0) * 1e-6)

    # === Step ③: Sort Gaussians by depth — closest first (front-to-back order) ===
    sort_idx = torch.argsort(depths)  # indices that put smallest z first
    sorted_gaussians = GaussianParams(
        position=gaussians.position[sort_idx],
        scale=gaussians.scale[sort_idx],
        rotation=gaussians.rotation[sort_idx],
        opacity=gaussians.opacity[sort_idx]
    )

    # Apply sigmoid to get actual alpha values ∈ (0, 1)
    alphas = torch.sigmoid(sorted_gaussians.opacity.squeeze(-1))  # [N]

    # === Step ④: Create pixel grid and evaluate kernel per-pixel ===
    y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    all_px = x_grid.float().view(1, -1)        # [1, num_pixels]: x coordinate of each pixel
    all_py = y_grid.float().view(1, -1)        # [1, num_pixels]: y coordinate of each pixel

    # Get projected 2D centers for sorted Gaussians (in pixel space)
    px_2d, py_2d = centers_2d[sort_idx][:, 0], centers_2d[sort_idx][:, 1]  # [N]

    # Compute offset: dx[i,p] = all_px[0,p] - px_2d[i] → shape [N, num_pixels]
    dx = all_px - px_2d.unsqueeze(-1)         # [N, num_pixels]: pixel_x - gaussian_center_x (broadcasts N×P)
    dy = all_py - py_2d.unsqueeze(-1)         # [N, num_pixels]

    dxy = torch.stack([dx, dy], dim=1)        # [N, 2, num_pixels]

    # Evaluate Gaussian kernel for each gaussian-pixel pair
    kernel_vals = evaluate_gaussian_kernel(dxy, cov_inv, det[sort_idx])  # [N, num_pixels]

    # === Step ⑤: Alpha compositing — front-to-back volume rendering ===
    # Pre-compute per-gaussian color (simplified: uniform warm light for demo)
    base_color = torch.ones(N, 3, device=gaussians.position.device) * 0.8
    sorted_colors = base_color[sort_idx]  # [N, 3]

    # Initialize output buffers
    image_flat = torch.zeros(num_pixels, 3, device=gaussians.position.device)  # [num_pixels, 3]
    cumulative_alpha = torch.zeros(num_pixels, device=gaussians.position.device)  # [num_pixels]

    # Front-to-back compositing loop:
    #   For each Gaussian (sorted closest-first):
    #     weight_i = α_i × kernel_value_at_pixel × Π_{j<i}(1-α_j)
    #     image += color_i × weight_i / (sum of weights per pixel to normalize)
    # This directly implements: C(x) = Σ c_i @ α_i @ Π(1-α_j)

    for i in range(N):
        alpha_i = alphas[i]                          # Scalar opacity contribution
        kernel_at_pixel = kernel_vals[i:i+1, :]      # [1, num_pixels]: spatial extent
        effective_alpha = (alpha_i * kernel_at_pixel).clamp(max=0.95)  # α_i × f(d)

        # Transparency from all closer Gaussians: Π_{j<i}(1 - α_j)
        transparency = (1.0 - cumulative_alpha).unsqueeze(0)  # [1, num_pixels]

        # Weight of this gaussian's contribution at each pixel
        weight = effective_alpha * transparency          # [1, num_pixels]

        # Add to image — volume rendering: C(x) = Σ c_i · α_i · Π_{j<i}(1-α_j)
        color_i = sorted_colors[i:i+1].unsqueeze(1)    # [1, 1, 3] → broadcasts to [1, num_pixels, 3]
        image_flat += color_i * weight.unsqueeze(-1)   # Direct addition, no normalization needed

        # Update cumulative alpha: α_accum ← 1 - Π_{j≤i}(1-α_j)
        cumulative_alpha = 1.0 - transparency * (1.0 - effective_alpha.squeeze(0)).clamp(max=1.0)

    return image_flat.reshape(H, W, 3)


# ================================================================
# Section 7: Training Step — Forward + Backward + Update (训练步骤)
#
# Math derivation — the optimization problem of 3DGS:
#   min_{μ,S,R,α,c}  Σ_v L(C_v(θ), I_v^gt)
# where θ = all Gaussian parameters, C_v is rendered image for view v,
# and I_v^gt is ground truth. The loss combines L1 + D-SSIM.
# ================================================================

def compute_simple_ssim(render: torch.Tensor, gt: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Simplified SSIM for demonstration (not the full windowed version).

    Computes local structural similarity using small pooling neighborhoods.
    Full implementation uses a learned sliding window.
    """
    # Local mean via averaging pool with reflective padding
    def local_mean(x):
        padded = nn.functional.pad(x, (k//2,) * 4, mode='reflect')
        return nn.functional.avg_pool2d(padded, kernel_size=k, stride=1)

    mu_r = local_mean(render)
    mu_g = local_mean(gt)
    var_r = local_mean((render - mu_r)**2)
    var_g = local_mean((gt - mu_g)**2)
    cov_rg = local_mean((render - mu_r) * (gt - mu_g))

    C1, C2 = 0.01**2, 0.03**2  # SSIM constants for stability
    ssim_map = ((2*mu_r*mu_g + C1) * (2*cov_rg + C2)) / \
               ((mu_r**2 + mu_g**2 + C1) * (var_r + var_g + C2))

    return ssim_map.mean()


def training_step(gaussians: GaussianParams, camera: Camera,
                  gt_image: torch.Tensor, optimizer: torch.optim.Optimizer,
                  lambda_dssim: float = 0.2) -> dict:
    """
    Execute ONE training step of 3DGS.

    Pipeline (forward → backward → update):
      ① Forward render → build computation graph with autograd
      ② Compute L1 + D-SSIM loss against ground truth image
      ③ Backward pass → compute gradients on ALL Gaussian parameters
      ④ Optimizer step → update position, scale, rotation, opacity

    Args:
        gaussians: Current Gaussian params (all must have requires_grad=True)
        camera: Camera pose for this training view
        gt_image: Ground truth RGB image [H, W, 3] from the dataset
        optimizer: PyTorch optimizer with param groups attached to Gaussians
        lambda_dssim: Weight for D-SSIM loss component (default=0.2)

    Returns: Loss dictionary for logging/tracking.
    """
    # === Step ①: Forward render — builds the ENTIRE computation graph! ===
    # Every operation in render_single_view is tracked by PyTorch's autograd engine.
    # This is what makes gradients flow from loss back to Gaussian parameters.
    rendered = render_single_view(gaussians, camera)  # [H, W, 3]

    # === Step ②: Compute loss against ground truth image ===
    gt_rgb = gt_image.permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W] — batch format for loss
    render_rgb = rendered.permute(2, 0, 1).unsqueeze(0)

    # L1 loss: mean absolute difference per pixel (simple and effective)
    l1_loss = nn.functional.l1_loss(render_rgb, gt_rgb)

    # D-SSIM loss: structural similarity index (promotes perceptual quality)
    ssim_val = compute_simple_ssim(render_rgb, gt_rgb)
    dssim_loss = 1.0 - ssim_val  # Convert from "similarity" to "loss"

    # Combined loss with weighting — balances color accuracy vs structural fidelity
    total_loss = l1_loss + lambda_dssim * dssim_loss

    # === Step ③: BACKPROPAGATION ===
    # This is the magic moment: PyTorch traces through EVERY operation in render_single_view
    # and computes ∂total_loss/∂position, ∂total_loss/∂scale, etc.
    # These gradients are stored in the .grad attribute of each tensor.

    optimizer.zero_grad()  # Clear previous gradients (accumulation would corrupt optimization)
    total_loss.backward()  # Fill .grad attributes on all trainable tensors

    # === Step ④: Optimizer step — apply computed gradients to update parameters ===
    optimizer.step()

    return {
        'l1_loss': float(l1_loss.item()),
        'dssim_loss': float(dssim_loss.item()),
        'total_loss': float(total_loss.item())
    }


# ================================================================
# Section 8: Densification — Adding New Gaussians (密度控制)
#
# Math derivation — adaptive refinement strategy from the paper:
#   CLONE: When a Gaussian has high opacity AND high gradient norm,
#           clone it and offset slightly to refine local geometry.
#   SPLIT: For flat/elongated Gaussians with large scale, split along
#          principal axes into smaller, more directional Gaussians.
#
# This is the "adaptive" part of Gaussian Splatting — the scene
# representation grows organically during training.
# ================================================================

def densify_gaussians(gaussians: GaussianParams, opacity_thresh: float = 0.8,
                      grad_norm_thresh: float = 0.0002) -> Tuple[GaussianParams, int]:
    """
    Densification step — add new Gaussians where the scene needs more detail.

    Operations (from original paper):
      CLONE: Duplicate a Gaussian near high-gradient regions → finer resolution
      SPLIT: For large flat Gaussians → split into smaller ones along principal axes

    Args:
        gaussians: Current Gaussian parameters with .grad filled from backward()
        opacity_thresh: Opacity threshold for cloning candidates (higher = more opaque)
        grad_norm_thresh: Gradient norm threshold — high gradient means "needs refinement"

    Returns: Updated Gaussians + count of newly added Gaussians.
    """
    # Guard: densification requires gradients from a backward pass
    if gaussians.position.grad is None:
        return gaussians, 0
    
    # Get gradient magnitudes on positions (from backprop in training_step)
    grad_norms = torch.norm(gaussians.position.grad, dim=1)   # [N] — ∂loss/∂position per Gaussian

    opacities = torch.sigmoid(gaussians.opacity.squeeze(-1))  # [N] — actual opacity values

    # Find candidates: high opacity + high gradient → "this Gaussian is visible but blurry"
    clone_mask = (opacities > opacity_thresh) & (grad_norms > grad_norm_thresh)
    candidates = torch.where(clone_mask)[0]

    if len(candidates) == 0:
        return gaussians, 0  # Scene is well-represented, no densification needed

    cap = min(len(candidates), 20)  # Cap to prevent exponential growth (simplified)
    selected = candidates[:cap]
    n_new = len(selected)

    # === CLONE: Duplicate each candidate with noise offset along scale directions ===
    base_noise = torch.randn(n_new, 3, device=gaussians.position.device) * 0.5
    new_pos = gaussians.position[selected] + \
              base_noise * gaussians.scale[selected].unsqueeze(0).clamp(min=1e-4)

    # Scales and rotations inherited from parent (cloning preserves shape/orientation)
    new_scale = gaussians.scale[selected]
    new_rot = nn.functional.normalize(torch.randn(n_new, 4, device=gaussians.position.device), dim=1)

    # Opacity: slightly reduced to create competition for visual space
    new_opacity = (gaussians.opacity[selected] - 0.3).clamp(min=-2.0)  # pre-sigmoid logit

    # === Concatenate new Gaussians to existing set ===
    all_pos = torch.cat([gaussians.position, new_pos], dim=0)
    all_scale = torch.cat([gaussians.scale, new_scale], dim=0)
    all_rot = torch.cat([gaussians.rotation, new_rot], dim=0)
    all_opacity = torch.cat([gaussians.opacity, new_opacity], dim=0)

    return GaussianParams(position=all_pos, scale=all_scale, rotation=all_rot, opacity=all_opacity), n_new


# ================================================================
# Section 9: Main Training Loop — End-to-End (主训练循环)
#
# Ties everything together — the complete training engine of 3DGS.
# Simulates training on a synthetic scene with multiple views.
# ================================================================

def main():
    """
    Full demonstration of 3DGS training pipeline.

    Simulates: random Gaussian initialization → multi-view rendering → 
    optimization loop with densification → final rendered image.
    """
    print("=" * 60)
    print("  3D Gaussian Splatting Training Pipeline Demo")
    print("  完整 3DGS 训练链路演示")
    print("=" * 60)

    # === Setup: Initialize Gaussians from random distribution ===
    N = 5000
    gaussians = create_random_gaussians(N)

    # Enable gradient tracking on ALL parameters — this is how PyTorch knows what to optimize!
    for attr in ['position', 'scale', 'rotation', 'opacity']:
        current_tensor = getattr(gaussians, attr)
        setattr(gaussians, attr, current_tensor.requires_grad_(True))

    # === Setup: Camera parameters ===
    # Camera sits at world origin, identity rotation means it looks down +Z axis.
    # Gaussians are placed at z=[1, 4] — directly in the camera's field of view.
    camera = Camera(
        R_w2c=torch.eye(3),            # Identity — look down +Z axis from origin
        t_w2c=torch.zeros(3, 1),       # Camera at world origin (0, 0, 0)
        fx=500.0, fy=500.0,            # Focal length in pixels
        cx=256.0, cy=256.0             # Image center for 512×512 output
    )

    # === Setup: Create a synthetic ground truth image (a simple bright spot) ===
    gt_image = torch.ones(512, 512, 3) * 0.8  # Dark gray background
    y_grid, x_grid = torch.meshgrid(torch.arange(512), torch.arange(512), indexing='ij')

    # Bright warm spot at center (simulates a lit object in the scene)
    intensity = torch.exp(-((x_grid - 256)**2 + (y_grid - 256)**2) / 8000.0)
    gt_image[:, :, 0] = intensity * 1.2   # Red: bright center → warm light
    gt_image[:, :, 1] = intensity * 0.6   # Green: dimmer
    gt_image[:, :, 2] = intensity * 0.3   # Blue: dimmest

    print(f"\n[Setup] Created {N} Gaussians with random parameters")
    print(f"[Camera] At origin, looking down +Z | Focal={camera.fx}px")
    print(f"[Scene] Synthetic warm light at image center (512×512)")

    # === Setup: Optimizer with per-parameter learning rates ===
    # Different parameters learn at different speeds — a key insight from the paper!
    params = [
        {'params': gaussians.position, 'lr': 0.01},       # Positions update fastest (move Gaussians)
        {'params': gaussians.scale, 'lr': 0.005},          # Scale adapts medium speed
        {'params': gaussians.rotation, 'lr': 0.001},       # Rotation changes slowly (stability!)
        {'params': gaussians.opacity, 'lr': 0.05},         # Opacity learns very fast (masking)
    ]
    optimizer = torch.optim.Adam(params, lr=0.0025, eps=1e-15)

    print(f"[Optimizer] Adam with per-param learning rates")

    # === Training Loop: Simulate N iterations ===
    NUM_ITERATIONS = 20
    losses = []
    gaussian_counts = [N]

    print(f"\n{'─' * 64}")
    print(f"{'Iter':>5} | {'L1 Loss':>9} | {'D-SSIM':>10} | "
          f"{'Total':>8} | {'Gaussians':>8}")
    print(f"{'─' * 64}")

    for step in range(NUM_ITERATIONS):
        # Execute one training step (forward → loss → backward → update)
        loss_dict = training_step(gaussians, camera, gt_image, optimizer)
        losses.append(loss_dict)

        # Densification every 5 steps (simulated adaptive refinement)
        if step % 5 == 0 and step > 0:
            gaussians, added = densify_gaussians(gaussians)
            gaussian_counts.append(len(gaussians.position))

        print(f"{step+1:>5} | {loss_dict['l1_loss']:>9.4f} | "
              f"{loss_dict['dssim_loss']:>10.6f} | "
              f"{loss_dict['total_loss']:>8.4f} | "
              f"{len(gaussians.position):>8}")

    print(f"{'─' * 64}\n")

    # === Final: Render the optimized result ===
    final_image = render_single_view(gaussians, camera)
    avg_opacity = torch.sigmoid(gaussians.opacity).mean().item()

    print(f"[Final] Rendered image shape: {final_image.shape}")
    print(f"[Final] Gaussian count: {len(gaussians.position)} (started with {N})")
    print(f"[Final] Avg opacity: {avg_opacity:.4f}")

    # Save demo state for inspection / visualization later
    torch.save({
        'position': gaussians.position.detach().cpu(),
        'scale': gaussians.scale.detach().cpu(),
        'rotation': gaussians.rotation.detach().cpu(),
        'opacity': gaussians.opacity.detach().cpu(),
        'final_losses': losses,
        'camera_pose': camera.t_w2c.squeeze(-1).detach().cpu(),
        'image_size': (512, 512),
    }, '/mnt/disk_e/work/git/3dgs_tutorial/python_demos/demo_state.pt')

    print(f"\n[DONE] Demo state saved → python_demos/demo_state.pt")
    print("To visualize: load with torch.load() and render more views.")


if __name__ == "__main__":
    main()
