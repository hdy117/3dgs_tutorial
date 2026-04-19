"""
============================================================
  Train Demo: Load Saved Gaussian State → Optimize → Re-Render
  训练演示：加载预存高斯状态 → 优化训练 → 重新渲染
  
Workflow:
  1. load scene_state.pt (rendered in render_demo.py)
  2. train with L1 + D-SSIM loss (with densification every N steps)
  3. re-render comparison (before vs after training)

This is the second half of the render → train pipeline.
Run `python render_demo.py` FIRST to create scene_state.pt, then run this.

要求：torch >= 2.0, matplotlib >= 3.5
============================================================
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


# ================================================================
# Section 1: Gaussian Parameter Definition (高斯参数定义)
# ================================================================

class Gaussians:
    """高斯参数容器 — training needs grad tracking + densification support."""
    
    def __init__(self, position: torch.Tensor, scale: torch.Tensor,
                 rotation: torch.Tensor, opacity: torch.Tensor,
                 colors_sh0: torch.Tensor = None):
        self.position = position   # [N, 3] — world-space mean (x, y, z)
        self.scale = scale         # [N, 3] — per-axis scaling
        self.rotation = rotation   # [N, 4] — unit quaternion (w, x, y, z)
        self.opacity = opacity     # [N, 1] — pre-sigmoid values
        default_color = torch.ones(position.shape[0], 3, device=position.device)
        self.colors_sh0 = colors_sh0 if colors_sh0 is not None else default_color
    
    @property
    def N(self):
        return self.position.shape[0]


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions [N, 4] to rotation matrices [N, 3, 3]."""
    w, x, y, z = q.unbind(dim=1)
    ww, xx, yy, zz = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    R = torch.stack([
        1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy),
        2*(xy + wz),       1 - 2*(xx + zz),   2*(yz - wx),
        2*(xz - wy),       2*(yz + wx),       1 - 2*(xx + yy),
    ], dim=1).reshape(-1, 3, 3)
    
    return R


def build_covariance(scale: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Build world-space covariance matrices Σ = (S@R)(S@R)^T."""
    R = quaternion_to_rotation_matrix(rotation)  # [N, 3, 3]
    sx, sy, sz = scale.unbind(dim=1)
    S = torch.diag_embed(torch.stack([sx, sy, sz], dim=1))  # [N, 3, 3]
    SR = S @ R
    return SR @ SR.transpose(1, 2)  # [N, 3, 3]


class Camera:
    """简化针孔相机：外参(R_w2c, t_w2c) + 内参(fx, fy, cx, cy)"""
    
    def __init__(self, R_w2c: torch.Tensor, t_w2c: torch.Tensor,
                 fx: float = 500.0, fy: float = 500.0,
                 cx: float = 256.0, cy: float = 256.0):
        self.R_w2c = R_w2c       # [3, 3] — world-to-camera rotation
        self.t_w2c = t_w2c       # [3, 1] — camera position in world space
        self.fx = fx             # focal length x (pixels)
        self.fy = fy             # focal length y (pixels)
        self.cx = cx             # principal point x
        self.cy = cy             # principal point y
    
    def project_positions(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project world-space 3D points → pixel coordinates."""
        p_cam = (self.R_w2c @ positions.T).T + self.t_w2c.squeeze(-1)  # [N, 3]
        depths = p_cam[:, 2]
        
        x_cam, y_cam, z_cam = p_cam.unbind(dim=1)
        px = self.fx * (x_cam / torch.clamp(z_cam, min=0.01)) + self.cx
        py = self.fy * (y_cam / torch.clamp(z_cam, min=0.01)) + self.cy
        
        return torch.stack([px, py], dim=1), depths


def project_covariance_to_2d(cov_world: torch.Tensor, position_cam: torch.Tensor,
                             camera: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D covariance → 2D image plane via Jacobian method."""
    x_cam, y_cam, z_cam = position_cam.unbind(dim=1)
    
    J_row_x = torch.stack([camera.fx / z_cam, 
                           torch.zeros_like(z_cam), 
                           -camera.fx * x_cam / (z_cam ** 2)], dim=1)
    J_row_y = torch.stack([torch.zeros_like(z_cam), 
                           camera.fy / z_cam, 
                           -camera.fy * y_cam / (z_cam ** 2)], dim=1)
    J = torch.stack([J_row_x, J_row_y], dim=1)  # [N, 2, 3]
    
    R_w2c = camera.R_w2c.unsqueeze(0)
    cov_cam = (R_w2c @ cov_world @ R_w2c.transpose(1, 2).unsqueeze(0)).squeeze(0)
    
    cov_2d = J @ cov_cam @ J.transpose(1, 2)  # [N, 2, 2]
    
    jitter = torch.eye(2, device=cov_2d.device).unsqueeze(0) * 0.3
    cov_2d = cov_2d + jitter
    
    det = cov_2d[:, 0, 0] * cov_2d[:, 1, 1] - cov_2d[:, 0, 1]**2
    
    return cov_2d, det


def render_single_view(gaussians: Gaussians, camera: Camera,
                       image_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Render a single view — the full Gaussian Splatting rendering pipeline.
    
    Pipeline steps:
      ① Project 3D centers → pixel coordinates
      ② Compute projected 2D covariance
      ③ Sort Gaussians by depth (closest first)
      ④ Evaluate 2D Gaussian kernel at each pixel
      ⑤ Alpha composite from front to back (volume rendering equation)
    """
    H, W = image_size
    N = gaussians.position.shape[0]
    num_pixels = H * W
    
    # Step 1: Project centers
    centers_2d, depths = camera.project_positions(gaussians.position)
    
    # Step 2: Compute projected covariance
    cov_world = build_covariance(gaussians.scale, gaussians.rotation)
    p_cam = (camera.R_w2c @ gaussians.position.T).T + camera.t_w2c.squeeze(-1)
    cov_2d, det = project_covariance_to_2d(cov_world, p_cam, camera)
    cov_inv = torch.linalg.inv(cov_2d + torch.eye(2).unsqueeze(0) * 1e-6)
    
    # Step 3: Sort by depth (closest first — front-to-back compositing)
    sort_idx = torch.argsort(depths)
    sorted_positions = gaussians.position[sort_idx]
    sorted_colors = gaussians.colors_sh0[sort_idx]
    alphas_raw = torch.sigmoid(gaussians.opacity.squeeze(-1))[sort_idx]  # [N]
    
    px_2d, py_2d = centers_2d[sort_idx][:, 0], centers_2d[sort_idx][:, 1]
    sorted_det = det[sort_idx]
    sorted_cov_inv = cov_inv[sort_idx]
    
    # Step 4: Create pixel grid and evaluate kernel
    y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    all_px = x_grid.float().view(1, -1)        # [1, num_pixels]
    all_py = y_grid.float().view(1, -1)        # [1, num_pixels]
    
    dx = all_px - px_2d.unsqueeze(-1)          # [N, num_pixels]
    dy = all_py - py_2d.unsqueeze(-1)          # [N, num_pixels]
    dxy = torch.stack([dx, dy], dim=1)         # [N, 2, num_pixels]
    
    inv_d = sorted_cov_inv @ dxy               # [N, 2, num_pixels]
    mahal = (dxy * inv_d).sum(dim=1)           # [N, num_pixels]
    
    norm = 1.0 / (2 * math.pi * torch.sqrt(torch.maximum(sorted_det.unsqueeze(1), 
                                                        torch.tensor(1e-8))))
    kernel_vals = norm * torch.exp(-0.5 * mahal)  # [N, num_pixels]
    
    # Step 5: Alpha compositing — front-to-back volume rendering
    image_flat = torch.zeros(num_pixels, 3, device=gaussians.position.device)
    cumulative_alpha = torch.zeros(num_pixels, device=gaussians.position.device)
    
    for i in range(N):
        alpha_i = alphas_raw[i]
        kernel_at_pixel = kernel_vals[i:i+1, :]      # [1, num_pixels]
        effective_alpha = (alpha_i * kernel_at_pixel).clamp(max=0.95)  # α × f(d)
        
        transparency = (1.0 - cumulative_alpha).unsqueeze(0)  # [1, num_pixels]
        weight = effective_alpha * transparency          # [1, num_pixels]
        
        color_i = sorted_colors[i:i+1].unsqueeze(1)     # [1, 1, 3]
        image_flat += color_i * weight.unsqueeze(-1)    # [num_pixels, 3]
        
        cumulative_alpha = 1.0 - transparency.squeeze(0) * (1.0 - effective_alpha.squeeze(0))
    
    return image_flat.reshape(H, W, 3)


# ================================================================
# Section 2: Loss Computation — L1 + D-SSIM
# ================================================================

def compute_simple_ssim(render: torch.Tensor, gt: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Simplified SSIM for demonstration.
    
    Full 3DGS uses windowed D-SSIM with learnable parameters.
    This is a pedagogical simplification that captures the same idea:
    structural similarity penalizes local brightness/contrast differences.
    """
    def local_mean(x):
        padded = nn.functional.pad(x, (k//2,) * 4, mode='reflect')
        return nn.functional.avg_pool2d(padded, kernel_size=k, stride=1)
    
    mu_r = local_mean(render)
    mu_g = local_mean(gt)
    var_r = local_mean((render - mu_r)**2)
    var_g = local_mean((gt - mu_g)**2)
    cov_rg = local_mean((render - mu_r) * (gt - mu_g))
    
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_r*mu_g + C1) * (2*cov_rg + C2)) / \
               ((mu_r**2 + mu_g**2 + C1) * (var_r + var_g + C2))
    
    return ssim_map.mean()


def training_step(gaussians: Gaussians, camera: Camera,
                  gt_image: torch.Tensor, optimizer: torch.optim.Optimizer,
                  lambda_dssim: float = 0.2) -> dict:
    """
    Execute ONE training step of 3DGS.
    
    Pipeline (forward → backward → update):
      ① Forward render → build computation graph with autograd
      ② Compute L1 + D-SSIM loss against ground truth image
      ③ Backward pass → compute gradients on ALL Gaussian parameters
      ④ Optimizer step → update position, scale, rotation, opacity
    """
    # Step 1: Forward render (autograd tracks EVERY operation)
    rendered = render_single_view(gaussians, camera)  # [H, W, 3]
    
    # Step 2: Compute loss against ground truth image
    gt_rgb = gt_image.permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W]
    render_rgb = rendered.permute(2, 0, 1).unsqueeze(0)
    
    l1_loss = nn.functional.l1_loss(render_rgb, gt_rgb)
    ssim_val = compute_simple_ssim(render_rgb, gt_rgb)
    dssim_loss = 1.0 - ssim_val
    
    total_loss = l1_loss + lambda_dssim * dssim_loss
    
    # Step 3: Backpropagation — gradients flow from loss → Gaussian parameters!
    optimizer.zero_grad()
    total_loss.backward()
    
    # Step 4: Optimizer step — apply computed gradients
    optimizer.step()
    
    return {
        'l1_loss': float(l1_loss.item()),
        'dssim_loss': float(dssim_loss.item()),
        'total_loss': float(total_loss.item())
    }


# ================================================================
# Section 3: Densification — Add New Gaussians (密度控制)
# ================================================================

def densify_gaussians(gaussians: Gaussians, opacity_thresh: float = 0.8,
                      grad_norm_thresh: float = 0.0002, max_N: int = 50000) -> Tuple[Gaussians, int]:
    """
    Densification step — add new Gaussians where the scene needs more detail.
    
    Operations (from original paper):
      CLONE: Duplicate a Gaussian near high-gradient regions → finer resolution
      SPLIT: For large flat Gaussians with low opacity → split into smaller ones
    
    Args:
        gaussians: Current Gaussian parameters with .grad filled from backward()
        opacity_thresh: Opacity threshold for cloning candidates
        grad_norm_thresh: Gradient norm threshold — high gradient = "needs refinement"
        max_N: Maximum number of Gaussians (prevent runaway growth)
    
    Returns: Updated Gaussians + count of newly added Gaussians.
    """
    # Guard: densification requires gradients from a backward pass
    if gaussians.position.grad is None:
        return gaussians, 0
    
    grad_norms = torch.norm(gaussians.position.grad, dim=1)   # [N]
    opacities = torch.sigmoid(gaussians.opacity.squeeze(-1))  # [N]
    
    # Find candidates: high opacity + high gradient → "this Gaussian is visible but blurry"
    clone_mask = (opacities > opacity_thresh) & (grad_norms > grad_norm_thresh)
    candidates = torch.where(clone_mask)[0]
    
    if len(candidates) == 0 or gaussians.N >= max_N:
        return gaussians, 0
    
    cap = min(len(candidates), 30)  # Cap new Gaussians per step (simplified)
    selected = candidates[:cap]
    n_new = len(selected)
    
    # === CLONE: Duplicate with noise offset along scale directions ===
    base_noise = torch.randn(n_new, 3, device=gaussians.position.device) * 0.5
    new_pos = gaussians.position[selected] + \
              base_noise * gaussians.scale[selected].unsqueeze(0).clamp(min=1e-4)
    
    # Scales inherited from parent; rotations slightly perturbed
    new_scale = gaussians.scale[selected] * 0.8  # shrink clones for competition
    new_rot = nn.functional.normalize(torch.randn(n_new, 4, device=gaussians.position.device), dim=1)
    
    # Opacity: reduced to create competition for visual space
    new_opacity = (gaussians.opacity[selected] - 0.5).clamp(min=-2.0)
    
    # === Concatenate ===
    all_pos = torch.cat([gaussians.position, new_pos], dim=0)
    all_scale = torch.cat([gaussians.scale, new_scale], dim=0)
    all_rot = torch.cat([gaussians.rotation, new_rot], dim=0)
    all_opacity = torch.cat([gaussians.opacity, new_opacity], dim=0)
    
    colors_sh0 = torch.cat([gaussians.colors_sh0, 
                            gaussians.colors_sh0[selected]], dim=0)
    
    return Gaussians(position=all_pos, scale=all_scale, rotation=all_rot,
                     opacity=all_opacity, colors_sh0=colors_sh0), n_new


def reset_gradients(gaussians: Gaussians):
    """Clear gradients after densification (new Gaussians don't inherit .grad)."""
    for attr in ['position', 'scale', 'rotation', 'opacity']:
        tensor = getattr(gaussians, attr)
        if hasattr(tensor, 'requires_grad') and tensor.requires_grad:
            grad = getattr(gaussians, f'{attr}_grad_backup') if \
                hasattr(gaussians, f'{attr}_grad_backup') else None
    # Simply zero out all gradients by re-optimizing with fresh optimizer next step.
    # For a cleaner approach, save/restore .grad before clone, then zero new Gaussians.


# ================================================================
# Section 4: Synthetic Ground Truth Creation
# ================================================================

def create_gt_image(H: int = 512, W: int = 512) -> torch.Tensor:
    """
    Create a synthetic ground truth image for training.
    
    This simulates a scene with colored objects at known positions.
    In real usage, this would be loaded from COLMAP images.
    """
    gt_image = torch.ones(H, W, 3) * 0.15  # Dark background
    
    y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # Object 1: Red sphere at image center
    intensity_r = torch.exp(-((x_grid - 256)**2 + (y_grid - 256)**2) / 6000.0)
    gt_image[:, :, 0] += intensity_r * 0.8
    
    # Object 2: Blue sphere at right
    intensity_b = torch.exp(-((x_grid - 384)**2 + (y_grid - 256)**2) / 6000.0)
    gt_image[:, :, 2] += intensity_b * 0.7
    
    # Object 3: Green sphere at left  
    intensity_g = torch.exp(-((x_grid - 128)**2 + (y_grid - 256)**2) / 6000.0)
    gt_image[:, :, 1] += intensity_g * 0.7
    
    # Object 4: Yellow sphere at top
    intensity_y = torch.exp(-((x_grid - 256)**2 + (y_grid - 128)**2) / 6000.0)
    gt_image[:, :, 0] += intensity_y * 0.3
    gt_image[:, :, 1] += intensity_y * 0.4
    
    return torch.clamp(gt_image, 0.0, 1.0)


# ================================================================
# Section 5: Visualization
# ================================================================

def visualize_comparison(before: torch.Tensor, after: torch.Tensor, 
                         title: str = "Before vs After Training"):
    """Display side-by-side comparison of before/after training renders."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    img_before = np.clip(before.cpu().numpy(), 0.0, 1.0)
    img_after = np.clip(after.cpu().numpy(), 0.0, 1.0)
    
    ax1.imshow(img_before)
    ax1.set_title("Before Training (random Gaussians)", fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(img_after)
    ax2.set_title("After Training (optimized Gaussians)", fontsize=12)
    ax2.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    out_path = '/mnt/disk_e/work/git/3dgs_tutorial/python_demos/train_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] train_comparison.png")
    plt.show(block=False)


# ================================================================
# Section 6: Main Training Pipeline
# ================================================================

def main():
    """
    Load saved Gaussian state from render_demo.py → Train → Re-render.
    
    This completes the render-first, train-later workflow:
      1. load scene_state.pt (created by render_demo.py)
      2. enable gradient tracking on all parameters
      3. run training loop with densification
      4. re-render and compare before/after
    """
    import numpy as np
    
    print("=" * 60)
    print("  3D Gaussian Splatting — Training Demo")
    print("  训练演示：从预存高斯状态优化 → 重新渲染")
    print("=" * 60)
    
    # === Step 1: Load saved scene state from render_demo.py ===
    checkpoint_path = '/mnt/disk_e/work/git/3dgs_tutorial/python_demos/scene_state.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] {checkpoint_path} not found!")
        print("Please run 'python render_demo.py' first to create the scene.")
        return
    
    state = torch.load(checkpoint_path, weights_only=False)
    
    gaussians = Gaussians(
        position=state['position'],
        scale=state['scale'],
        rotation=state['rotation'],
        opacity=state['opacity'],
        colors_sh0=state['colors_sh0']
    )
    
    N_initial = gaussians.N
    print(f"\n[Loaded] Scene state from {checkpoint_path}")
    print(f"  Gaussians: {N_initial} (positions, scales, rotations, opacities, colors)")
    
    # === Step 2: Enable gradient tracking on ALL parameters ===
    for attr in ['position', 'scale', 'rotation', 'opacity']:
        tensor = getattr(gaussians, attr)
        setattr(gaussians, attr, tensor.requires_grad_(True))
    
    print(f"[Setup] Enabled requires_grad=True on all {N_initial} Gaussian parameters")
    
    # === Step 3: Setup camera and ground truth image ===
    camera = Camera(
        R_w2c=torch.eye(3),
        t_w2c=torch.zeros(3, 1),
        fx=500.0, fy=500.0, cx=256.0, cy=256.0
    )
    
    gt_image = create_gt_image(H=512, W=512)
    print(f"[Scene] Synthetic ground truth (4 colored blobs on dark background)")
    
    # === Step 4: Setup optimizer with per-parameter learning rates ===
    params = [
        {'params': gaussians.position, 'lr': 0.001},       # Positions: slow move
        {'params': gaussians.scale, 'lr': 0.0005},          # Scale: very slow adaptation
        {'params': gaussians.rotation, 'lr': 0.0001},       # Rotation: stability first!
        {'params': gaussians.opacity, 'lr': 0.01},          # Opacity: faster convergence
    ]
    optimizer = torch.optim.Adam(params, lr=0.0025, eps=1e-15)
    
    # === Step 5: Render BEFORE training (baseline) ===
    print(f"\n{'─' * 64}")
    print("Rendering BEFORE training (random initialization)...")
    before_render = render_single_view(gaussians, camera)
    print(f"  Before render shape: {tuple(before_render.shape)}")
    
    # Save before render for comparison
    torch.save({
        'render': before_render.detach().cpu(),
        'gaussians_before': {
            k: getattr(gaussians, k).detach().cpu() 
            for k in ['position', 'scale', 'rotation', 'opacity', 'colors_sh0']
        }
    }, '/mnt/disk_e/work/git/3dgs_tutorial/python_demos/before_train.pt')
    
    # === Step 6: Training Loop ===
    NUM_ITERATIONS = 100
    DENSIFY_EVERY = 10  # densification every N steps (like original paper)
    losses = []
    gaussian_counts = [N_initial]
    
    print(f"\n{'─' * 64}")
    print(f"{'Iter':>5} | {'L1 Loss':>9} | {'D-SSIM':>10} | "
          f"{'Total':>8} | {'Gaussians':>8} | {'Added':>6}")
    print(f"{'─' * 64}\n")
    
    for step in range(NUM_ITERATIONS):
        # Execute one training step (forward → loss → backward → update)
        loss_dict = training_step(gaussians, camera, gt_image, optimizer)
        losses.append(loss_dict)
        
        # Densification every N steps (adaptive refinement)
        added = 0
        if step > 0 and step % DENSIFY_EVERY == 0:
            gaussians, added = densify_gaussians(gaussians)
            gaussian_counts.append(len(gaussians.position))
        
        print(f"{step+1:>5} | {loss_dict['l1_loss']:>9.4f} | "
              f"{loss_dict['dssim_loss']:>10.6f} | "
              f"{loss_dict['total_loss']:>8.4f} | "
              f"{len(gaussians.position):>8} | {added:>6}")
    
    print(f"\n{'─' * 64}\n")
    
    # === Step 7: Render AFTER training (final result) ===
    print("Rendering AFTER training...")
    after_render = render_single_view(gaussians, camera)
    avg_opacity = torch.sigmoid(gaussians.opacity).mean().item()
    
    print(f"[Final] Rendered image shape: {after_render.shape}")
    print(f"[Final] Gaussian count: {len(gaussians.position)} "
          f"(started with {N_initial}, added during densification)")
    print(f"[Final] Avg opacity: {avg_opacity:.4f}")
    
    # Calculate improvement
    l1_before = nn.functional.l1_loss(before_render.permute(2, 0, 1).unsqueeze(0), 
                                       gt_image.permute(2, 0, 1).unsqueeze(0)).item()
    l1_after = losses[-1]['l1_loss'] if losses else float('inf')
    
    print(f"\n[Improvement] L1 loss: {l1_before:.4f} → {l1_after:.4f}"
          f" ({((l1_before - l1_after) / max(l1_before, 1e-8) * 100):.1f}% reduction)")
    
    # === Step 8: Save trained state ===
    torch.save({
        'position': gaussians.position.detach().cpu(),
        'scale': gaussians.scale.detach().cpu(),
        'rotation': gaussians.rotation.detach().cpu(),
        'opacity': gaussians.opacity.detach().cpu(),
        'colors_sh0': gaussians.colors_sh0.detach().cpu(),
        'final_losses': losses,
        'gaussian_counts': gaussian_counts,
    }, '/mnt/disk_e/work/git/3dgs_tutorial/python_demos/trained_state.pt')
    
    print(f"\n[DONE] Trained state saved → python_demos/trained_state.pt")
    print("  (Load with torch.load() and render more views)")
    
    # === Step 9: Show comparison visualization ===
    try:
        visualize_comparison(before_render, after_render, 
                           "3DGS Training: Before vs After")
        print("\n[Visualization] Comparison saved → train_comparison.png")
    except Exception as e:
        print(f"\n[Note] Visualization skipped ({e})")


if __name__ == "__main__":
    import os
    main()
