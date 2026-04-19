"""
============================================================
  Render Demo: 3D Gaussian Splatting — Pure PyTorch Rendering
  渲染演示：纯 PyTorch 实现 3DGS 渲染管线
  
This script demonstrates the rendering pipeline in isolation.
No training, no optimization — just Gaussians → Pixels.

目标：用最少代码展示 3DGS 渲染的完整数学推导 → 代码映射。
要求：torch >= 2.0, matplotlib >= 3.5
============================================================
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple


# ================================================================
# Section 1: Gaussian Parameter Definition (高斯参数定义)
# ================================================================

class Gaussians:
    """高斯参数容器 — 渲染阶段不需要增删，训练阶段需要"""
    def __init__(self, position: torch.Tensor, scale: torch.Tensor,
                 rotation: torch.Tensor, opacity: torch.Tensor,
                 colors_sh0: torch.Tensor = None):
        self.position = position   # [N, 3] — world-space mean (x, y, z)
        self.scale = scale         # [N, 3] — per-axis scaling (positive via exp)
        self.rotation = rotation   # [N, 4] — unit quaternion (w, x, y, z)
        self.opacity = opacity     # [N, 1] — pre-sigmoid values
        default_color = torch.ones(position.shape[0], 3, device=position.device)
        self.colors_sh0 = colors_sh0 if colors_sh0 is not None else default_color


# ================================================================
# Section 2: Quaternion → Rotation Matrix (四元数 → 旋转矩阵)
# ================================================================

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions [N, 4] to rotation matrices [N, 3, 3]."""
    w, x, y, z = q.unbind(dim=1)
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
# ================================================================

def build_covariance(scale: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Build world-space covariance matrices.
    
    Math: Σ_world = S @ R @ (S @ R)^T = S @ R @ R^T @ S^T
    
    This creates an ellipsoid with axes aligned to the rotated scale directions.
    Think of it as: first stretch along principal axes (scale), then rotate orientation.
    """
    R = quaternion_to_rotation_matrix(rotation)  # [N, 3, 3]
    
    sx, sy, sz = scale.unbind(dim=1)
    S = torch.diag_embed(torch.stack([sx, sy, sz], dim=1))  # [N, 3, 3]
    
    SR = S @ R                                    # Scale then rotate: [N, 3, 3]
    cov_world = SR @ SR.transpose(1, 2)           # Σ = (SR)(SR)^T: [N, 3, 3]
    
    return cov_world


# ================================================================
# Section 4: Camera Model (相机模型)
# ================================================================

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
        """
        Project world-space 3D points → pixel coordinates.
        
        Pipeline per point p_world:
          1. p_cam = R_w2c @ p_world + t_w2c    (extrinsics)
          2. x_pix = fx * (x_cam/z_cam) + cx    (intrinsics, perspective divide)
          3. y_pix = fy * (y_cam/z_cam) + cy
        """
        p_cam = (self.R_w2c @ positions.T).T + self.t_w2c.squeeze(-1)  # [N, 3]
        
        depths = p_cam[:, 2]
        
        x_cam, y_cam, z_cam = p_cam.unbind(dim=1)
        px = self.fx * (x_cam / torch.clamp(z_cam, min=0.01)) + self.cx
        py = self.fy * (y_cam / torch.clamp(z_cam, min=0.01)) + self.cy
        
        return torch.stack([px, py], dim=1), depths


# ================================================================
# Section 5: 2D Gaussian Kernel Evaluation (二维高斯核评估)
# ================================================================

def project_covariance_to_2d(cov_world: torch.Tensor, position_cam: torch.Tensor,
                             camera: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D covariance → 2D image plane via Jacobian method.
    
    Math (the key trick of 3DGS):
      A 3D Gaussian projected onto the image plane becomes a 2D Gaussian.
      
      Σ_2D ≈ J @ R_w2c^T @ Σ_world @ R_w2c @ J^T
      
      where J = ∂(x_pix, y_pix)/∂(x_cam, y_cam, z_cam) is the perspective projection Jacobian:
        ∂x'/∂x = fx/z,   ∂x'/∂y = 0,     ∂x'/∂z = -fx*x/z²
        ∂y'/∂x = 0,      ∂y'/∂y = fy/z,  ∂y'/∂z = -fy*y/z²
    
    This is where the beautiful math of Gaussian Splatting lives:
    a 3D ellipsoid viewed through perspective becomes a 2D ellipse.
    """
    x_cam, y_cam, z_cam = position_cam.unbind(dim=1)
    
    # Jacobian rows (∂pixel / ∂camera_coords): [N, 3] each
    J_row_x = torch.stack([camera.fx / z_cam, 
                           torch.zeros_like(z_cam), 
                           -camera.fx * x_cam / (z_cam ** 2)], dim=1)
    J_row_y = torch.stack([torch.zeros_like(z_cam), 
                           camera.fy / z_cam, 
                           -camera.fy * y_cam / (z_cam ** 2)], dim=1)
    J = torch.stack([J_row_x, J_row_y], dim=1)  # [N, 2, 3]
    
    # Transform covariance to camera space: Σ_cam = R_w2c @ Σ_world @ R_w2c^T
    R_w2c = camera.R_w2c.unsqueeze(0)  # [1, 3, 3] for batch multiply
    cov_cam = (R_w2c @ cov_world @ R_w2c.transpose(1, 2).unsqueeze(0)).squeeze(0)  # [N, 3, 3]
    
    # Project to 2D: Σ_2D = J @ Σ_cam @ J^T
    cov_2d = J @ cov_cam @ J.transpose(1, 2)  # [N, 2, 2]
    
    # Clamp diagonal for positive-definiteness (numerical stability)
    jitter = torch.eye(2, device=cov_2d.device).unsqueeze(0) * 0.3
    cov_2d = cov_2d + jitter
    
    # Determinant: |Σ_2D| = ad - bc²
    det = cov_2d[:, 0, 0] * cov_2d[:, 1, 1] - cov_2d[:, 0, 1]**2
    
    return cov_2d, det


def render_single_view(gaussians: Gaussians, camera: Camera,
                       image_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Render a single view — the full Gaussian Splatting rendering pipeline.
    
    Mathematically this implements the discretized volume rendering equation:
      C(x) = Σ_{i∈N} c_i · α_i · Π_{j=1}^{i-1} (1 - α_j)
    
    Where:
      • N = Gaussians sorted front-to-back by depth
      • c_i = RGB color of Gaussian i
      • α_i = opacity contribution = sigmoid(opacity_param) × kernel_value
    
    Step-by-step pipeline:
      ① Project 3D centers → pixel coordinates     [Sec 4]
      ② Compute projected 2D covariance            [Sec 5]
      ③ Sort Gaussians by depth (closest first)    
      ④ Evaluate 2D Gaussian kernel at each pixel  
      ⑤ Alpha composite from front to back         [Volume Rendering Eq.]
    """
    H, W = image_size
    N = gaussians.position.shape[0]
    num_pixels = H * W
    
    # === Step ①: Project Gaussian centers to screen space ===
    centers_2d, depths = camera.project_positions(gaussians.position)
    
    # === Step ②: Compute projected 2D covariance + determinant ===
    cov_world = build_covariance(gaussians.scale, gaussians.rotation)
    p_cam = (camera.R_w2c @ gaussians.position.T).T + camera.t_w2c.squeeze(-1)
    
    cov_2d, det = project_covariance_to_2d(cov_world, p_cam, camera)
    
    # Inverse covariance for kernel evaluation
    cov_inv = torch.linalg.inv(cov_2d + torch.eye(2).unsqueeze(0) * 1e-6)
    
    # === Step ③: Sort Gaussians by depth — closest first (front-to-back) ===
    sort_idx = torch.argsort(depths)
    
    sorted_positions = gaussians.position[sort_idx]
    sorted_scale = gaussians.scale[sort_idx]
    sorted_colors = gaussians.colors_sh0[sort_idx]  # [N, 3]
    alphas_raw = torch.sigmoid(gaussians.opacity.squeeze(-1))[sort_idx]  # [N]
    
    px_2d, py_2d = centers_2d[sort_idx][:, 0], centers_2d[sort_idx][:, 1]
    sorted_det = det[sort_idx]
    sorted_cov_inv = cov_inv[sort_idx]
    
    # === Step ④: Create pixel grid and evaluate kernel per-pixel ===
    y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    all_px = x_grid.float().view(1, -1)        # [1, num_pixels]
    all_py = y_grid.float().view(1, -1)        # [1, num_pixels]
    
    # Offset vectors: dxy[i, :, p] = pixel_p - gaussian_center_i  → [N, 2, num_pixels]
    dx = all_px - px_2d.unsqueeze(-1)          # [N, num_pixels]: broadcast N×P
    dy = all_py - py_2d.unsqueeze(-1)          # [N, num_pixels]
    dxy = torch.stack([dx, dy], dim=1)         # [N, 2, num_pixels]
    
    # Evaluate Gaussian kernel: f(d) = exp(-0.5 * d^T @ Σ^{-1} @ d) / (2π√|Σ|)
    inv_d = sorted_cov_inv @ dxy               # [N, 2, num_pixels]: Σ^{-1} @ d
    mahal = (dxy * inv_d).sum(dim=1)           # [N, num_pixels]: Mahalanobis distance²
    
    norm = 1.0 / (2 * math.pi * torch.sqrt(torch.maximum(sorted_det.unsqueeze(1), 
                                                        torch.tensor(1e-8))))
    kernel_vals = norm * torch.exp(-0.5 * mahal)  # [N, num_pixels]
    
    # === Step ⑤: Alpha compositing — front-to-back volume rendering ===
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
# Section 6: Scene Construction — Create a Colorful Scene
# ================================================================

def create_sphere_scene(num_gaussians_per_sphere=200, radius=0.4):
    """
    Create a scene with multiple colored spheres using surface-sampled Gaussians.
    
    This is the tutorial's simplified initialization — in practice you'd use COLMAP point clouds.
    
    Math: Sample uniformly on sphere surface, then add slight inward perturbation.
      p = center + r · (2·u - 1) where u ~ Uniform(0,1), |p-center| ≈ radius
    """
    all_positions = []
    all_scales = []
    all_rotations = []
    all_opacities = []
    all_colors = []
    
    # Define spheres with different colors and positions
    sphere_configs = [
        {"center": (0.0, 0.3, 2.5), "color": (1.0, 0.2, 0.2)},   # Red sphere
        {"center": (-0.8, -0.2, 3.0), "color": (0.2, 0.6, 1.0)},  # Blue sphere  
        {"center": (0.7, -0.4, 2.8),  "color": (0.2, 0.9, 0.3)},  # Green sphere
        {"center": (0.0, 0.5, 1.8),   "color": (1.0, 0.9, 0.1)},  # Yellow sphere (closer)
    ]
    
    for config in sphere_configs:
        cx, cy, cz = config["center"]
        cr, cg, cb = config["color"]
        
        n = num_gaussians_per_sphere
        # Sample points on sphere surface
        theta = torch.rand(n) * 2 * math.pi
        phi = torch.acos(1 - 2 * torch.rand(n))  # Uniform on sphere
        
        pos_x = cx + radius * torch.sin(phi) * torch.cos(theta)
        pos_y = cy + radius * torch.sin(phi) * torch.sin(theta)
        pos_z = cz + radius * torch.cos(phi)
        
        all_positions.append(torch.stack([pos_x, pos_y, pos_z], dim=1))
        
        # Scale: small isotropic Gaussians for smooth surface
        s = torch.ones(n, 3) * (radius * 0.4)
        all_scales.append(s)
        
        # Random rotations (Gaussians point outward from sphere center)
        rot = nn.functional.normalize(torch.randn(n, 4), dim=1)
        all_rotations.append(rot)
        
        # High opacity — spheres are fully visible
        op = torch.ones(n, 1) * 3.0  # pre-sigmoid: sigmoid(3) ≈ 0.95
        
        all_opacities.append(op)
        
        # Color from config
        color = torch.full((n, 3), 0.7)
        color[:, 0] += cr - 0.6
        color[:, 1] += cg - 0.6
        color[:, 2] += cb - 0.6
        all_colors.append(color.clamp(0.0, 1.0))
    
    # Add a ground plane (flat green surface)
    n_ground = 1500
    gx = torch.randn(n_ground) * 3 + 0
    gy = torch.randn(n_ground) * 3 - 1
    gz = torch.ones(n_ground) * (-0.6)  # flat ground at z=-0.6
    
    all_positions.append(torch.stack([gx, gy, gz], dim=1))
    all_scales.append(torch.full((n_ground, 3), 0.15))
    all_rotations.append(nn.functional.normalize(torch.randn(n_ground, 4), dim=1))
    all_opacities.append(torch.ones(n_ground, 1) * 2.5)
    ground_color = torch.tensor([0.15, 0.35, 0.15]).view(1, 3).expand(n_ground, -1)
    all_colors.append(ground_color)
    
    # Concatenate everything
    positions = torch.cat(all_positions, dim=0)
    scales = torch.cat(all_scales, dim=0)
    rotations = torch.cat(all_rotations, dim=0)
    opacities = torch.cat(all_opacities, dim=0)
    colors = torch.cat(all_colors, dim=0)
    
    return Gaussians(position=positions, scale=scales, rotation=rotations,
                     opacity=opacities, colors_sh0=colors)


# ================================================================
# Section 7: Camera View Control (相机视角控制)
# ================================================================

def make_orthogonal_view(azimuth: float = 0.0, elevation: float = 0.0) -> Camera:
    """
    Create an orthogonal camera view.
    
    Math: The world-to-camera rotation is built from spherical coordinates:
      R_w2c rotates the world so that the camera's viewing direction maps to +Z axis.
      
      View direction d = (cos(elev)*sin(az), sin(elev), cos(elev)*cos(az))
      Camera position = -d * distance
    """
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    
    # Build camera position from spherical coordinates
    dist = 5.0
    cam_x = dist * math.cos(el_rad) * math.sin(az_rad)
    cam_y = dist * math.sin(el_rad)
    cam_z = dist * math.cos(el_rad) * math.cos(az_rad)
    
    t_w2c = torch.tensor([[cam_x], [cam_y], [cam_z]])
    
    # Build rotation matrix: world → camera frame
    # Forward vector (toward origin from camera)
    forward = -t_w2c.squeeze(-1) / dist  # normalized view direction
    
    # Compute right and up vectors via cross products
    world_up = torch.tensor([0.0, 1.0, 0.0]).float()
    if abs(forward[1].item()) > 0.99:  # Avoid gimbal lock at poles
        right = torch.tensor([1.0, 0.0, 0.0]).float()
    else:
        right = torch.cross(world_up, forward)
        if right.norm().item() > 0:
            right = right / right.norm()
        else:
            right = torch.tensor([1.0, 0.0, 0.0]).float()

    up = torch.cross(forward, right)
    
    # R_w2c rows are the camera basis vectors expressed in world coordinates
    R_w2c = torch.stack([-right, -up, forward], dim=1).T  # [3, 3]
    
    return Camera(R_w2c=R_w2c, t_w2c=t_w2c, fx=500.0, fy=500.0, cx=256.0, cy=256.0)


# ================================================================
# Section 8: Visualization (可视化)
# ================================================================

def visualize_image(image: torch.Tensor, title: str = "Rendered Image"):
    """Display rendered image using matplotlib."""
    import matplotlib.pyplot as plt
    
    img_np = image.cpu().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.clip(img_np, 0.0, 1.0))
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('/home/dhu/shared/work/git/3dgs_tutorial/python_demos/render_output.png', dpi=150, bbox_inches='tight')
    print(f"[Saved] render_output.png")
    plt.show(block=False)


# ================================================================
# Section 9: Main — Rendering Demo (渲染演示)
# ================================================================

def main():
    """
    Run the rendering demo.
    
    Creates a scene with colored spheres and renders from multiple camera angles.
    No training needed — just pure math → pixels.
    """
    print("=" * 60)
    print("  3D Gaussian Splatting — Rendering Demo")
    print("  渲染演示：高斯体 → 像素 (无训练)")
    print("=" * 60)
    
    # === Create scene with colored spheres ===
    gaussians = create_sphere_scene(num_gaussians_per_sphere=200, radius=0.4)
    N = gaussians.position.shape[0]
    print(f"\n[Scene] Created {N} Gaussians (3 colored spheres + ground plane)")
    print(f"  Positions: [{gaussians.position[:, 0].min():.2f}, {gaussians.position[:, 0].max():.2f}]")
    print(f"  Colors: RGB ∈ [0,1]")
    
    # === Render from multiple angles ===
    view_angles = [
        (0, 5),      # Front view
        (90, 0),     # Right side
        (45, 30),    # Isometric view
        (-60, -10),  # Low angle
    ]
    
    print(f"\n{'─' * 60}")
    for az, el in view_angles:
        camera = make_orthogonal_view(azimuth=az, elevation=el)
        
        image = render_single_view(gaussians, camera, image_size=(512, 512))
        
        print(f"Rendered [{az:3d}°, {el:3d}°] | Shape: {tuple(image.shape)}")
        print(f"  Pixel range: [{image.min():.4f}, {image.max():.4f}]")
    
    # Render the front view and save
    camera = make_orthogonal_view(azimuth=0, elevation=5)
    final_image = render_single_view(gaussians, camera, image_size=(512, 512))
    
    print(f"\n{'─' * 60}")
    print("[Final] Rendering front view (0°, 5°)...")
    visualize_image(final_image, "3DGS Render — Colored Spheres + Ground")
    
    # Save scene state for training demo
    torch.save({
        'position': gaussians.position.detach().cpu(),
        'scale': gaussians.scale.detach().cpu(),
        'rotation': gaussians.rotation.detach().cpu(),
        'opacity': gaussians.opacity.detach().cpu(),
        'colors_sh0': gaussians.colors_sh0.detach().cpu(),
    }, '/home/dhu/shared/work/git/3dgs_tutorial/python_demos/scene_state.pt')
    
    print(f"\n[DONE] Scene saved → python_demos/scene_state.pt")
    print("  (Load in train_demo.py for training)")


if __name__ == "__main__":
    main()
