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

import torch
from typing import List, Tuple, Optional

EPS = 1e-8

def normalize_quaternion(quat: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Normalize quaternions to unit length.
    
    Args:
        quat: Quaternion tensor of shape (..., 4) with (w, x, y, z) format
        eps: Small value for numerical stability
        
    Returns:
        Normalized quaternions of same shape
    """
    norm = torch.norm(quat, dim=-1, keepdim=True)
    return quat / torch.clamp(norm, min=eps)

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions using Hamilton product.
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack((w, x, y, z), dim=-1)

def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate.
    """
    w, xyz = quat[..., :1], quat[..., 1:]
    return torch.cat([w, -xyz], dim=-1)

def quat_inverse(quat: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Compute quaternion inverse.
    """
    conjugate = quat_conjugate(quat)
    norm_squared = torch.sum(quat * quat, dim=-1, keepdim=True)
    return conjugate / torch.clamp(norm_squared, min=eps)

def quat_log(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute quaternion logarithm, mapping to rotation vectors (axis-angle).
    """
    # quat_norm = normalize_quaternion(quat, eps)
    q_norm = torch.sqrt(torch.sum(quat * quat, dim=-1, keepdim=True))
    quat_norm = quat / torch.clamp(q_norm, min=eps)
    
    w = quat_norm[..., 0:1]    # Scalar part
    xyz = quat_norm[..., 1:]   # Vector part
    
    xyz_norm = torch.norm(xyz, dim=-1, keepdim=True)
    w_clamped = torch.clamp(w, min=-1.0 + eps, max=1.0 - eps)
    
    # half-angle
    half_angle = torch.acos(torch.abs(w_clamped))
    
    safe_xyz_norm = torch.clamp(xyz_norm, min=eps)
    
    # Scale factor
    scale = torch.where(
        xyz_norm < eps,
        torch.ones_like(xyz_norm), 
        half_angle / safe_xyz_norm 
    )
    
    # Handle quaternion sign ambiguity (q and -q represent same rotation)
    sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    
    rotation_vector = sign * scale * xyz
    
    return rotation_vector

def quat_rotate_vector(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate a 3D vector by a quaternion.
    """
    q_vec = quat[..., 1:]  # vector part
    q_w = quat[..., 0:1]   # scalar part
    
    cross1 = torch.cross(q_vec, vec, dim=-1)
    cross2 = torch.cross(q_vec, cross1, dim=-1)
    
    # Apply the rotation formula
    rotated_vec = vec + 2.0 * q_w * cross1 + 2.0 * cross2
    
    return rotated_vec

def quat_to_rotation_matrix(quat: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    """
    quat_norm = normalize_quaternion(quat, eps)
    w, x, y, z = torch.unbind(quat_norm, dim=-1)
    
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    
    rotation_matrix = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=-1)
    
    return rotation_matrix.reshape(quat.shape[:-1] + (3, 3))

def quat_to_transform_matrix(quat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion and position to 4x4 transformation matrix.
    """
    # rotation part
    rotation = quat_to_rotation_matrix(quat)
    batch_shape = rotation.shape[:-2]
    
    # homogeneous transformation matrix
    transform = torch.zeros(batch_shape + (4, 4), dtype=rotation.dtype, device=rotation.device)
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = pos
    transform[..., 3, 3] = 1.0
    
    return transform

def compute_rest_local_positions(
    joint_positions: torch.Tensor,
    parent_indices: List[int]
) -> torch.Tensor:
    """
    Compute local positions relative to parent joints from global joint positions.
    """
    
    num_joints = joint_positions.shape[0]
    local_positions = torch.zeros_like(joint_positions)
    
    for j in range(num_joints):
        parent_idx = parent_indices[j]
        
        if parent_idx >= 0 and parent_idx != j and parent_idx < num_joints:
            # Child joint: local offset = global_pos - parent_global_pos
            local_positions[j] = joint_positions[j] - joint_positions[parent_idx]
        else:
            # Root joint: use global position as local position
            local_positions[j] = joint_positions[j]
    
    return local_positions