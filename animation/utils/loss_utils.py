#  Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from third_partys.Video_Depth_Anything.video_depth_anything.video_depth import VideoDepthAnything
import torch
import torch.nn as nn
import numpy as np
import igl
import cv2
import time
import torch.nn.functional as F
from utils.quat_utils import quat_inverse, quat_log, quat_multiply, normalize_quaternion
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch
import os
from pathlib import Path

class DepthModule:
    def __init__(self, encoder='vitl', device='cuda', input_size=518, fp32=False):
        """
        Initialize the depth loss module with Video Depth Anything
        
        Args:
            encoder: 'vitl' or 'vits'
            device: device to run the model on
            input_size: input size for the model
            fp32: whether to use float32 for inference
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size = input_size
        self.fp32 = fp32
        
        # Initialize model configuration
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        # Load Video Depth Anything model
        self.video_depth_model = VideoDepthAnything(**model_configs[encoder])
        self.video_depth_model.load_state_dict(
            torch.load(f'./third_partys/Video_Depth_Anything/ckpt/video_depth_anything_{encoder}.pth', map_location='cpu'), 
            strict=True
        )
        self.video_depth_model = self.video_depth_model.to(self.device).eval()
        for param in self.video_depth_model.parameters():
            param.requires_grad = False 
    
    def get_depth_maps(self, frames, target_fps=30):
        """
        Get depth maps for video frames
        """
        depths, _ = self.video_depth_model.infer_video_depth(
            frames, 
            target_fps, 
            input_size=self.input_size, 
            device=self.device, 
            fp32=self.fp32
        )
        return depths
    
def save_depth_as_images(depth_np, output_dir='./depth_images'):
    """save depth images"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, depth_map in enumerate(depth_np):
        depth_map = depth_map.detach().cpu().numpy()  
        valid_mask = (depth_map > 0)
        if not valid_mask.any():
            continue
            
        valid_min = depth_map[valid_mask].min()
        valid_max = depth_map[valid_mask].max()
        
        normalized = np.zeros_like(depth_map)
        normalized[valid_mask] = 255.0 * (depth_map[valid_mask] - valid_min) / (valid_max - valid_min)
    
        depth_img = normalized.astype(np.uint8)
        
        cv2.imwrite(os.path.join(output_dir, f'depth_{i:04d}.png'), depth_img)
        
    print(f"Save {len(depth_np)} depth images to {output_dir}")

def compute_visibility_mask_igl(ray_origins, ray_dirs, distances, verts, faces, distance_tolerance=1e-6, for_vertices=False):
    """
    Compute visibility mask using IGL ray-mesh intersection.
    """
    num_rays = ray_origins.shape[0]
    visibility_mask = np.ones(num_rays, dtype=bool)

    for i in range(num_rays):
        ray_origin = ray_origins[i].reshape(1, 3)
        ray_dir = ray_dirs[i].reshape(1, 3)
        intersections = igl.ray_mesh_intersect(ray_origin, ray_dir, verts, faces)
        if intersections:
            # Count intersections that occur before the target point
            count = sum(1 for h in intersections if h[4] < distances[i] - distance_tolerance)
            # count=0 → ray completely missed the mesh; count=1 → ray stops exactly at the face containing the joint
            # count>1 → ray was blocked by other faces along the way
            if for_vertices:
                if count != 1:
                    visibility_mask[i] = False
            else: # for joints
                if count > 2:
                    visibility_mask[i] = False

    return visibility_mask

def compute_reprojection_loss(renderer, vis_mask, predicted_joints, tracked_joints_2d, image_size):
    """
    Compute reprojection loss between predicted 3D points and tracked 2D points.
    """
    if predicted_joints.dim() != 3:
        raise ValueError(f"predicted_joints must be 3D tensor, got shape {predicted_joints.shape}")
    
    B, J, _ = predicted_joints.shape
    device = predicted_joints.device
    
    # Project 3D joints to 2D screen coordinates
    projected = renderer.camera.transform_points_screen(
        predicted_joints, 
        image_size=[image_size, image_size]
    )
    projected_2d = projected[..., :2]  # (B, J, 2)
   
    # Convert and validate tracked joints
    if not isinstance(tracked_joints_2d, torch.Tensor):
        tracked_joints_2d = torch.from_numpy(tracked_joints_2d).float()
    tracked_joints_2d = tracked_joints_2d.to(device)
   
    if tracked_joints_2d.dim() == 2:
        tracked_joints_2d = tracked_joints_2d.unsqueeze(0).expand(B, -1, -1)

    vis_mask = vis_mask.to(device).float()

    num_visible = vis_mask.sum()
    if num_visible == 0:
        # No visible joints - return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    squared_diff = (projected_2d - tracked_joints_2d).pow(2).sum(dim=-1)  # (B, J)
    
    vis_mask_expanded = vis_mask.unsqueeze(0)  # (1, J)
    masked_loss = squared_diff * vis_mask_expanded  # (B, J)
    per_frame_loss = masked_loss.sum(dim=1) / num_visible  # (B,)
    final_loss = per_frame_loss.mean()  # scalar

    return final_loss

def geodesic_loss(q1, q2, eps=1e-6):
    """
    Compute geodesic distance loss between batches of quaternions for rot smooth loss.
    """
    q1_norm = normalize_quaternion(q1, eps=eps)
    q2_norm = normalize_quaternion(q2, eps=eps)
   
    dot_product = (q1_norm * q2_norm).sum(dim=-1, keepdim=True)
    q2_corrected = torch.where(dot_product < 0, -q2_norm, q2_norm)
    inner_product = (q1_norm * q2_corrected).sum(dim=-1)
   
    # Clamp to valid range for arccos to avoid numerical issues
    inner_product_clamped = torch.clamp(inner_product, min=-1.0 + eps, max=1.0 - eps)
    theta = 2.0 * torch.arccos(torch.abs(inner_product_clamped))

    return theta

def root_motion_reg(root_quats, root_pos):
    return ((root_pos[1:] - root_pos[:-1])**2).mean(), (geodesic_loss(root_quats[1:], root_quats[:-1])**2).mean()

def joint_motion_coherence(quats_normed, parent_idx):
    """
    Compute joint motion coherence loss to enforce smooth relative motion between parent-child joints.
    """
    coherence_loss = 0
    
    for j, parent in enumerate(parent_idx):
        if parent != -1:  # Skip root joint
            parent_rot = quats_normed[:, parent]  # (T, 4)
            child_rot = quats_normed[:, j]        # (T, 4)
            
            # Compute relative rotation of child w.r.t. parent's local frame
            # local_rot = parent_rot^(-1) * child_rot
            local_rot = quat_multiply(quat_inverse(parent_rot), child_rot)
            local_rot_velocity = local_rot[1:] - local_rot[:-1]  #  (T-1, 4)

            coherence_loss += local_rot_velocity.pow(2).mean()
    
    return coherence_loss

def read_flo_file(file_path):
    """
    Read optical flow from .flo format file.
    """
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if len(magic) == 0 or magic[0] != 202021.25:
               raise ValueError(f'Invalid .flo file format: magic number {magic}')
           
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        flow = data.reshape(h, w, 2)
    return flow

def load_optical_flows(flow_dir, num_frames):
    """
    Load sequence of optical flow files.
    """
    flow_dir = Path(flow_dir)
    flows = []
    
    for i in range(num_frames - 1):
        flow_path = flow_dir / f'flow_{i:04d}.flo'
        if flow_path.exists():
            flow = read_flo_file(flow_path)
            flows.append(flow)
        else:
            raise ValueError("No flow files found")
    
    return np.stack(flows, axis=0)

def rasterize_vertex_flow(flow_vertices, meshes, faces, image_size, renderer, eps = 1e-8):
    """
    Rasterize per-vertex flow to dense flow field using barycentric interpolation.
    """
    B, V, _ = flow_vertices.shape
    device = flow_vertices.device
    
    if isinstance(image_size, int):
        H = W = image_size
    else:
        H, W = image_size
    
    batch_meshes = join_meshes_as_batch([join_meshes_as_scene(m) for m in meshes]).to(device)
    fragments    = renderer.renderer.rasterizer(batch_meshes)
   
    pix_to_face = fragments.pix_to_face    # (B, H, W, K)
    bary_coords = fragments.bary_coords    # (B, H, W, K, 3)

    flow_scene_list = []
    for mesh_idx in range(B):
        mesh = meshes[mesh_idx]
        V_mesh = mesh.verts_packed().shape[0]
        
        if V_mesh > flow_vertices.shape[1]:
            raise ValueError(f"Mesh {mesh_idx} has {V_mesh} vertices but flow has {flow_vertices.shape[1]}")
        
        flow_scene_list.append(flow_vertices[mesh_idx, :V_mesh])
    

    flow_vertices_scene = torch.cat(flow_scene_list, dim=0).to(device)
    faces_scene = batch_meshes.faces_packed() 

    flow_pred = torch.zeros(B, H, W, 2, device=device)
    valid = pix_to_face[..., 0] >= 0
    
    for b in range(B):
        b_valid = valid[b]  # (H,W)
        if torch.count_nonzero(b_valid) == 0:
            print(f"No valid pixels found for batch {b}")
            continue
            
        valid_indices = torch.nonzero(b_valid, as_tuple=True)
        h_indices, w_indices = valid_indices
        
        face_idxs = pix_to_face[b, h_indices, w_indices, 0]  # (N,)
        bary = bary_coords[b, h_indices, w_indices, 0]       # (N,3)
        
        max_face_idx = faces_scene.shape[0] - 1
        if face_idxs.max() > max_face_idx:
            raise RuntimeError(f"Face index {face_idxs.max()} exceeds max {max_face_idx}")
       
        face_verts = faces_scene[face_idxs]  # (N, 3)
        f0, f1, f2 = face_verts.unbind(-1)  # Each (N,)
       
        max_vert_idx = flow_vertices_scene.shape[0] - 1
        if max(f0.max(), f1.max(), f2.max()) > max_vert_idx:
            raise RuntimeError(f"Vertex index exceeds flow_vertices_scene size {max_vert_idx}")
       
        v0_flow = flow_vertices_scene[f0]  # (N, 2)
        v1_flow = flow_vertices_scene[f1]  # (N, 2)
        v2_flow = flow_vertices_scene[f2]  # (N, 2)
        
        # Interpolate using barycentric coordinates
        b0, b1, b2 = bary.unbind(-1)  # Each (N,)
        
        # Ensure barycentric coordinates sum to 1 (numerical stability)
        bary_sum = b0 + b1 + b2
        b0 = b0 / (bary_sum + eps)
        b1 = b1 / (bary_sum + eps)
        b2 = b2 / (bary_sum + eps)
        
        flow_interpolated = (
            b0.unsqueeze(-1) * v0_flow + 
            b1.unsqueeze(-1) * v1_flow + 
            b2.unsqueeze(-1) * v2_flow
        )  # (N, 2)
        
        # Update flow prediction
        flow_pred[b, h_indices, w_indices] = flow_interpolated
    
    return flow_pred

def calculate_flow_loss(flow_dir, device, mask, renderer, model):
    """
    Calculate optical flow loss with improved error handling and flexibility.
    """
    if device is None:
        device = mask.device
    
    T = mask.shape[0]
    H, W = mask.shape[1:3]

    if mask.shape[0] == T:
        flow_mask = mask[1:] # Use frames 1 to T-1
    else:
        flow_mask = mask 
        
    flows_np = load_optical_flows(flow_dir, T)
    flow_gt = torch.from_numpy(flows_np).float().to(device)  # [T-1, H, W, 2]
    
    vertices = model.deformed_vertices[0] # (T,V,3)
    # Project vertices to get 2D flow
    proj_t  = renderer.project_points(vertices[:-1])    # (T-1,V,2) in pixels
    proj_tp = renderer.project_points(vertices[1:])
    vertex_flow  = proj_tp - proj_t                      # (T-1,V,2) Δx,Δy

    meshes = [model.get_mesh(t) for t in range(T)]
    flow_pred = rasterize_vertex_flow(vertex_flow, meshes, model.faces[0], (H,W), renderer)  # (B,H,W,2)
    
    eps = 1e-3
    diff = (flow_pred - flow_gt) * flow_mask.unsqueeze(-1) # (T-1, H, W, 2)
    loss = torch.sqrt(diff.pow(2).sum(dim=-1) + eps**2)  # Charbonnier loss
    loss = loss.sum() / (flow_mask.sum() + 1e-6)
    
    return loss

def normalize_depth_from_reference(depth_maps, reference_idx=0, invalid_value=-1.0, invert=False, eps = 1e-8):
    """
    Normalize depth maps based on a reference frame with improved robustness.
    """
    if depth_maps.dim() != 3:
        raise ValueError(f"Expected depth_maps with 3 dimensions, got {depth_maps.dim()}")
    
    T, H, W = depth_maps.shape
    device = depth_maps.device
    
    reference_depth = depth_maps[reference_idx]
    valid_mask = (
        (reference_depth != invalid_value) & 
        (reference_depth > 1e-8) &  # Avoid very small positive values
        torch.isfinite(reference_depth)  # Exclude inf/nan
    )

    valid_values = reference_depth[valid_mask]
    min_depth = torch.quantile(valid_values, 0.01)  # 1st percentile
    max_depth = torch.quantile(valid_values, 0.99)  # 99th percentile
    
    depth_range = max_depth - min_depth
    if depth_range < eps:
        logger.warning(f"Very small depth range ({depth_range:.6f}), using fallback normalization")
        min_depth = valid_values.min()
        max_depth = valid_values.max()
        depth_range = max(max_depth - min_depth, eps)
    
    scale = 1.0 / (max_depth - min_depth)
    offset = -min_depth * scale
    
    all_valid_mask = (
        (depth_maps != invalid_value) & 
        (depth_maps > eps) &
        torch.isfinite(depth_maps)
    )

    normalized_depths = torch.full_like(depth_maps, invalid_value)
    
    if all_valid_mask.any():
        normalized_values = depth_maps[all_valid_mask] * scale + offset
        
        if invert:
            normalized_values = 1.0 - normalized_values
        
        normalized_depths[all_valid_mask] = normalized_values
    
    return normalized_depths, scale.item(), offset.item()

def compute_depth_loss_normalized(mono_depths, zbuf_depths, mask):
    """
    Compute normalized depth loss.
    """ 
    device = zbuf_depths.device
    # Normalize both depth types
    zbuf_norm, z_scale, z_offset = normalize_depth_from_reference(zbuf_depths)
    mono_norm, m_scale, m_offset = normalize_depth_from_reference(mono_depths, invert=True)
    
    valid_zbuf = (zbuf_norm >= 0) & (zbuf_norm <= 1)
    valid_mono = (mono_norm >= 0) & (mono_norm <= 1)
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    combined_mask = mask & valid_zbuf & valid_mono
    
    num_valid = combined_mask.sum().item()
    if num_valid == 0:
        print("No valid pixels for depth loss computation")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    depth_diff = (zbuf_norm - mono_norm) * combined_mask.float()
    loss = (depth_diff**2).sum() / num_valid
    
    return loss