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
from typing import List, Optional, Tuple, Union
from collections import deque
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex, TexturesUV

from utils.quat_utils import quat_to_transform_matrix, quat_multiply, quat_rotate_vector

class RiggingModel:
    """
    A 3D rigged model supporting skeletal animation.
    
    Handles mesh geometry, skeletal hierarchy, skinning weights, and 
    linear blend skinning (LBS) deformation.
    """
    def __init__(self, device = "cuda:0"):
        self.device = device
        # Mesh data
        self.vertices: List[torch.Tensor] = []    
        self.faces: List[torch.Tensor] = []        
        self.textures: List[Union[TexturesVertex, TexturesUV]] = [] 
        
        # Skeletal data
        self.bones: Optional[torch.Tensor] = None       # (N, 2) [parent, child] pairs
        self.parent_indices: Optional[torch.Tensor] = None  # (J,) parent index for each joint
        self.root_index: Optional[int] = None           # Root joint index
        self.joints_rest: Optional[torch.Tensor] = None # (J, 3) rest pose positions
        self.skin_weights: List[torch.Tensor] = [] # List of (V_i, J) skinning weights

        # Fixed local positions
        self.rest_local_positions: Optional[torch.Tensor] = None  # (J, 3)
        
        # Computed data
        self.bind_matrices_inv: Optional[torch.Tensor] = None  # (J, 4, 4) inverse bind matrices
        self.deformed_vertices: Optional[List[torch.Tensor]] = None  # List of (T, V_i, 3)
        self.joint_positions: Optional[torch.Tensor] = None    # (T, J, 3) current joint positions
        
        # Validation flags
        self._bind_matrices_initialized = False

    def initialize_bind_matrices(self, rest_local_pos):
        """Initialize bind matrices and store rest local positions."""
        self.rest_local_positions = rest_local_pos.to(self.device)
        
        J = rest_local_pos.shape[0]
        rest_global_quats, rest_global_pos = self.forward_kinematics(
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * J], device=self.device),  # unit quaternion
            self.parent_indices,
            self.root_index
        )
        
        bind_matrices = quat_to_transform_matrix(rest_global_quats, rest_global_pos)  # (1,J,4,4)
        self.bind_matrices_inv = torch.inverse(bind_matrices.squeeze(0))  # (J,4,4)
        
        self._bind_matrices_initialized = True

    def animate(self, local_quaternions, root_quaternion = None, root_position = None):
        """
        Animate the model using local joint transformations.
        
        Args:
            local_quaternions: (T, J, 4) local rotations per frame
            root_quaternion: (T, 4) global root rotation
            root_position: (T, 3) global root translation
        """
        if not self._bind_matrices_initialized:
            raise RuntimeError("Bind matrices not initialized. Call initialize_bind_matrices() first.")
        
        # Forward kinematics
        global_quats, global_pos = self.forward_kinematics(
            local_quaternions,
            self.parent_indices,
            self.root_index
        )
        self.joint_positions = global_pos

        joint_transforms = quat_to_transform_matrix(global_quats, global_pos)  # (T, J, 4, 4)
        
        # Apply global root transformation if provided
        if root_quaternion is not None and root_position is not None:
            root_transform = quat_to_transform_matrix(root_quaternion, root_position)
            joint_transforms = root_transform[:, None] @ joint_transforms
            self.joint_positions = joint_transforms[..., :3, 3]

        # Linear blend skinning
        self.deformed_vertices = []
        for i, vertices in enumerate(self.vertices):
            deformed = self._linear_blend_skinning(
                vertices,
                joint_transforms,
                self.skin_weights[i],
                self.bind_matrices_inv
            )
            self.deformed_vertices.append(deformed)


    def get_mesh(self, frame_idx=None):
        meshes = []
        for i in range(len(self.vertices)):
            mesh = Meshes(
                verts=[self.vertices[i]] if frame_idx is None or self.deformed_vertices is None else [self.deformed_vertices[i][frame_idx]],
                faces=[self.faces[i]],
                textures=self.textures[i]
            )
            meshes.append(mesh)
        return join_meshes_as_scene(meshes)
    
    def _linear_blend_skinning(self, vertices, joint_transforms, skin_weights, bind_matrices_inv):
        """
        Apply linear blend skinning to vertices.
        
        Args:
            vertices: (V, 3) vertex positions
            joint_transforms: (T, J, 4, 4) joint transformation matrices
            skin_weights: (V, J) per-vertex joint weights
            bind_matrices_inv: (J, 4, 4) inverse bind matrices
            
        Returns:
            (T, V, 3) deformed vertices
        """
        # Compute final transformation matrices
        transforms = torch.matmul(joint_transforms, bind_matrices_inv)  # (T, J, 4, 4)
        
        # Weight and blend transformations
        weighted_transforms = torch.einsum('vj,tjab->tvab', skin_weights, transforms)  # (T, V, 4, 4)
        
        # Apply to vertices
        vertices_hom = torch.cat([vertices, torch.ones(vertices.shape[0], 1, device=vertices.device)], dim=-1)
        deformed = torch.matmul(weighted_transforms, vertices_hom.unsqueeze(-1)).squeeze(-1)
        
        return deformed[..., :3]
    
    def forward_kinematics(self, local_quaternions, parent_indices, root_index = 0):
        """
        Compute global joint transformations from local ones.
        
        Args:
            local_quaternions: (B, J, 4) local rotations
            parent_indices: (J,) parent index for each joint
            root_index: Root joint index
            
        Returns:
            Tuple of (global_quaternions, global_positions)
        """
        B, J = local_quaternions.shape[:2]
        local_positions = self.rest_local_positions.unsqueeze(0).expand(B, -1, -1)
        
        
        # Initialize storage
        global_quats = [None] * J
        global_positions = [None] * J
        
        # Build children mapping
        children = [[] for _ in range(J)]
        for child_idx in range(J):
            parent_idx = parent_indices[child_idx]
            if parent_idx >= 0:
                children[parent_idx].append(child_idx)
        
        # Breadth-first traversal from root
        queue = deque([root_index])
        visited = {root_index}
        
        # Process root
        global_quats[root_index] = local_quaternions[:, root_index]
        global_positions[root_index] = local_positions[:, root_index]
        
        while queue:
            current = queue.popleft()
            current_quat = global_quats[current]
            current_pos = global_positions[current]
            
            for child in children[current]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
                    
                    # Transform child to global space
                    child_quat = quat_multiply(current_quat, local_quaternions[:, child])
                    child_pos = quat_rotate_vector(current_quat, local_positions[:, child]) + current_pos
                    
                    global_quats[child] = child_quat
                    global_positions[child] = child_pos
        
        return torch.stack(global_quats, dim=1), torch.stack(global_positions, dim=1)
