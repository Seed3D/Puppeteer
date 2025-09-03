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

import os
import numpy as np
import torch
import random
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import TexturesAtlas
from pytorch3d.structures import Meshes
from model import RiggingModel

def prepare_depth(depth_path, input_frames, device, depth_model):
    os.makedirs(depth_path, exist_ok=True)
    depth_path  = f"{depth_path}/depth_gt_raw.pt" 
    if os.path.exists(depth_path):
        print("load GT depth...")
        depth_gt_raw = torch.load(depth_path, map_location=device)
    else:
        print("run VideoDepthAnything and save.")
        with torch.no_grad():
            depth_gt_raw = depth_model.get_depth_maps(input_frames)
        torch.save(depth_gt_raw.cpu(), depth_path)
        depth_gt_raw = depth_gt_raw.to(device)
    return depth_gt_raw

def normalize_vertices(verts):
    """Normalize vertices to a unit cube."""
    vmin, vmax = verts.min(dim=0).values, verts.max(dim=0).values
    center = (vmax + vmin) / 2.0
    scale = (vmax - vmin).max()
    verts_norm = (verts - center) / scale
    return verts_norm, center, scale

def build_atlas_texture(obj_path, atlas_size, device):
    """Load OBJ + materials and bake all textures into a single atlas."""
    verts, faces, aux = load_obj(
        obj_path,
        device=device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=atlas_size,
        texture_wrap="repeat",
    )
    atlas = aux.texture_atlas  # (F, R, R, 3)
    verts_norm, _, _ = normalize_vertices(verts)
    mesh_atlas = Meshes(
        verts=[verts_norm],
        faces=[faces.verts_idx],
        textures=TexturesAtlas(atlas=[atlas]),
    )
    return mesh_atlas

def read_rig_file(file_path):
    """
    Read rig from txt file, our format is the same as RigNet: 
    joints joint_name x y z
    root root_joint_name
    skin vertex_idx joint_name weight joint_name weight ...
    hier parent_joint_name child_joint_name
    """
    joints = []
    bones = []
    joint_names = [] 

    joint_mapping = {}
    joint_index = 0
    
    skinning_data = {}  # Dictionary to store vertex index -> [(joint_idx, weight), ...]

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.split()
        if line.startswith('joints'):
            name = parts[1]
            position = [float(parts[2]), float(parts[3]), float(parts[4])]
            joints.append(position)
            joint_names.append(name) 
            joint_mapping[name] = joint_index
            joint_index += 1
        elif line.startswith('hier'):
            parent_joint = joint_mapping[parts[1]]
            child_joint = joint_mapping[parts[2]]
            bones.append([parent_joint, child_joint])
        elif line.startswith('root'):
            root = joint_mapping[parts[1]]
        elif line.startswith('skin'):
            vertex_idx = int(parts[1])
            
            if vertex_idx not in skinning_data:
                skinning_data[vertex_idx] = []

            for i in range(2, len(parts), 2):
                if i+1 < len(parts):  
                    joint_name = parts[i]
                    weight = float(parts[i+1])
                    
                    if joint_name in joint_mapping:
                        joint_idx = joint_mapping[joint_name]
                        skinning_data[vertex_idx].append((joint_idx, weight))

    return np.array(joints), np.array(bones), root, joint_names, skinning_data

def load_model_from_obj_and_rig(
    mesh_path: str,
    rig_path: str,
    device: str | torch.device = "cuda",
    use_skin_color: bool = True,
    atlas_size: int = 8,
):
    """Load a 3D model from OBJ and rig files."""

    # 1) read raw mesh
    raw_mesh = load_objs_as_meshes([mesh_path], device=device)
    verts_raw = raw_mesh.verts_packed()      # (V,3)
    faces_idx = raw_mesh.faces_packed()      # (F,3)

    # 2) read rig data
    joints_np, bones_np, root_idx, joint_names, skinning_data = read_rig_file(rig_path)
    J = joints_np.shape[0]

    # parent indices, default -1
    parent_idx = [-1] * J
    for p, c in bones_np:
        parent_idx[c] = p

    verts_norm, center, scale = normalize_vertices(verts_raw)
    joints_t = torch.as_tensor(joints_np, dtype=torch.float32, device=device)
    joints_norm = (joints_t - center) / scale

    # skin weights tensor (V,J)
    V = verts_raw.shape[0]
    skin_weights = torch.zeros(V, J, dtype=torch.float32, device=device)
    for v_idx, lst in skinning_data.items():
        for j_idx, w in lst:
            skin_weights[v_idx, j_idx] = w

    # 3) texture strategy
    mesh_norm = build_atlas_texture(mesh_path, atlas_size, device)
    tex = mesh_norm.textures

    # 4) pack into Model class
    model = RiggingModel(device=device)
    model.vertices = [mesh_norm.verts_packed()]
    model.faces = [faces_idx]
    model.textures = [tex]

    # rig meta
    model.bones = bones_np  # (B,2)
    model.parent_indices = parent_idx
    model.root_index = root_idx
    model.skin_weights = [skin_weights]

    model.bind_matrices_inv = torch.eye(4, device=device).unsqueeze(0).expand(J, -1, -1).contiguous()
    model.joints_rest = joints_norm

    return model
