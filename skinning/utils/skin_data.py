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
import h5py
import numpy as np
import torch
import torch.utils.data as data
import trimesh
from collections import deque
from utils.util import process_mesh_to_pc, read_obj_file, read_rig_file, normalize_to_unit_cube, build_adjacency_list, \
                        compute_graph_distance, get_tpl_edges, triangulate_faces

class SkinData(data.Dataset):
    def __init__(self, args, mode, query_num=4096):
        self.args = args
        self.query_num = query_num
        self.mode = mode  # train, eval, generate
        
        if mode == 'eval':
            self._init_h5_data(args)
        elif mode == 'generate':
            self._init_file_data(args)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def _init_h5_data(self, args):
        """Initialize for H5 file-based evaluation"""
        self.data_source = 'h5'
        self.eval_data_path = args.eval_data_path
        self.h5_file = None
        
        with h5py.File(self.eval_data_path, 'r') as f:
            self.num_samples = len(f.keys())
        print(f"[SkinData] found {self.num_samples} samples in the dataset.")
    
    def _init_file_data(self, args):
        """Initialize for mesh/rig file-based generation"""
        self.data_source = 'files'
        self.mesh_folder = args.mesh_folder
        self.rig_files_dir = args.input_skel_folder
        
        # Get list of available samples
        self.sample_files = []
        for obj_file in os.listdir(self.mesh_folder):
            if obj_file.endswith('.obj'):
                file_name = os.path.splitext(obj_file)[0]
                rig_file_path = os.path.join(self.rig_files_dir, f'{file_name}.txt')
                if os.path.exists(rig_file_path):
                    self.sample_files.append((obj_file, rig_file_path, file_name))
        
        self.num_samples = len(self.sample_files)
        print(f"[SkinData] found {self.num_samples} samples for generation.")

    def _load_h5_data(self, idx):
        """Load data from H5 file"""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.eval_data_path, 'r')

        data = self.h5_file[f'sample_{idx}']
        
        sample_points = data['pc_w_norm'][:, :3]
        normal = data['pc_w_norm'][:, 3:]
        joints = data['joints'][:]
        bones = data['bones'][:]
        root_index = data['root_index'][()]
        graph_dist = data['graph_dist'][:]
        
        file_name = data['file_name'][()].decode('utf-8')
        vertices = data['vertices'][:]
        edges = data['edges'][:]
        gt_skin = data['skin'][:]
        
        return {
            'sample_points': sample_points,
            'normal': normal,
            'joints': joints,
            'bones': bones,
            'root_index': root_index,
            'graph_dist': graph_dist,
            'file_name': file_name,
            'vertices': vertices,
            'edges': edges,
            'gt_skin': gt_skin
        }
    
    def _load_file_data(self, idx):
        """Load data from mesh and rig files"""
        obj_file, rig_file_path, file_name = self.sample_files[idx]
        
        # Load mesh
        mesh_file_path = os.path.join(self.mesh_folder, obj_file)
        vertices, faces = read_obj_file(mesh_file_path)

        triangulated_faces = triangulate_faces(faces) # if faces are not triangles, triangulate them
        
        # Create trimesh object and process to point cloud
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangulated_faces)
        pc_w_norm, _ = process_mesh_to_pc(mesh, sample_num=8192)
        sample_points = pc_w_norm[:, :3]
        normal = pc_w_norm[:, 3:]
        
        # Load rig data
        joints, bones, root_index = read_rig_file(rig_file_path)
        
        # Normalize mesh and joints
        vertices, center, scale = normalize_to_unit_cube(vertices, 0.9995)
        joints -= center
        joints *= scale
        
        # Get edges
        edges = get_tpl_edges(vertices, faces)
        
        # Compute graph distance
        num_joints = joints.shape[0]
        adjacency = build_adjacency_list(num_joints, bones)
        graph_dist = compute_graph_distance(num_joints, adjacency)
        
        return {
            'sample_points': sample_points,
            'normal': normal,
            'joints': joints,
            'bones': bones,
            'root_index': root_index,
            'graph_dist': graph_dist,
            'file_name': file_name,
            'vertices': vertices,
            'edges': edges
        }

    def _process_data(self, data_dict):
        """Common processing for both data sources"""
        sample_points = data_dict['sample_points']
        normal = data_dict['normal']
        joints = data_dict['joints']
        bones = data_dict['bones']
        root_index = data_dict['root_index']
        graph_dist = data_dict['graph_dist']
        file_name = data_dict['file_name']
        vertices = data_dict['vertices']
        edges = data_dict['edges']
        if 'gt_skin' in data_dict:
            gt_skin = data_dict['gt_skin']
        
        # Random sampling for query points
        ind = np.random.default_rng().choice(sample_points.shape[0], self.query_num, replace=False)
        query_points = sample_points[ind]
        query_normal = normal[ind]
        
        # Normalize to (-0.5, 0.5)
        bounds = np.array([sample_points.min(axis=0), sample_points.max(axis=0)])
        center = (bounds[0] + bounds[1]) / 2
        scale = (bounds[1] - bounds[0]).max() + 1e-5
        
        sample_points = (sample_points - center) / scale
        query_points = (query_points - center) / scale
        joints = (joints - center) / scale
        vertices = (vertices - center) / scale

        # Normalize normals
        pc_coor = sample_points
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / (normal_norm + 1e-8)

        query_points = query_points.clip(-0.5, 0.5)
        joints = joints.clip(-0.5, 0.5)
        
        # Process joints to bone coordinates format
        j = joints.shape[0]
        bone_coor = np.zeros((j, 6))
        bone_coor[:, 3:] = joints
        
        # Create parent indices array
        parent_indices = np.ones(j, dtype=np.int32) * -1
        
        # Fill parent information using bones array
        for parent, child in bones:
            if parent_indices[child] == -1:
                parent_indices[child] = parent
        
        # Set root node parent to itself
        parent_indices[root_index] = root_index
        
        # Get parent coordinates
        valid_mask = parent_indices != -1
        bone_coor[valid_mask, :3] = joints[parent_indices[valid_mask]]
        
        # Convert to tensors
        query_points = torch.from_numpy(query_points).float()
        query_points_normal = torch.from_numpy(np.concatenate([query_points, query_normal], axis=-1)).float()
        bone_coor = torch.from_numpy(bone_coor).float()
        graph_dist = torch.from_numpy(graph_dist).float()
        edges = torch.from_numpy(edges).long()
        vertices = torch.from_numpy(vertices).float()
        if 'gt_skin' in data_dict:
            gt_skin = torch.from_numpy(gt_skin).float()
        
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
        pc_w_norm = torch.from_numpy(np.concatenate([pc_coor, normal], axis=-1)).float()
        
        # Handle joint padding
        max_joints = self.args.max_joints
        num_joints = bone_coor.shape[0]
        padding_size = max_joints - num_joints

        if padding_size > 0:
            bone_coor = torch.nn.functional.pad(bone_coor, (0, 0, 0, padding_size), 'constant', 0)
            graph_dist = torch.nn.functional.pad(
                graph_dist, 
                pad=(0, padding_size, 0, padding_size), 
                mode='constant', 
                value=999
            )
            if 'gt_skin' in data_dict:
                gt_skin = torch.nn.functional.pad(gt_skin, (0, padding_size), 'constant', 0)
        
        # Create valid joints mask
        valid_joints_mask = torch.zeros(max_joints, dtype=torch.bool)
        valid_joints_mask[:num_joints] = True
        
        if 'gt_skin' in data_dict:
            return query_points_normal, pc_w_norm, bone_coor, valid_joints_mask, graph_dist, vertices, file_name, edges, gt_skin
        else: 
            return query_points_normal, pc_w_norm, bone_coor, valid_joints_mask, graph_dist, vertices, file_name, edges

    def __getitem__(self, idx):
        # Load data based on source
        if self.data_source == 'h5':
            data_dict = self._load_h5_data(idx)
        else:  # files
            data_dict = self._load_file_data(idx)
        
        # data processing
        return self._process_data(data_dict)
            
    def __len__(self):
        return self.num_samples