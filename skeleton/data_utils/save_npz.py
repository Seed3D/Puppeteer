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
"""
This python script shows how we process the meshes and rigs from the input folders and save them in a compressed npz file.
"""
import os
import numpy as np
import glob
import pickle
from concurrent.futures import ProcessPoolExecutor
import skimage.measure
import trimesh
import mesh2sdf.core
import scipy.sparse as sp

def read_obj_file(file_path):
    vertices = []
    faces = []
    normals = []  # Added normals list
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()[1:]
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif line.startswith('vn '):  # Added reading normals
                parts = line.split()[1:]
                normals.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # OBJ format is 1-based, we need 0-based for npz
                face = [int(part.split('//')[0]) - 1 for part in parts]
                faces.append(face)
    
    return np.array(vertices), np.array(faces), np.array(normals) 

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

def convert_to_sparse_skinning(skinning_data, num_vertices, num_joints):
    """Convert skinning weights to sparse matrix format."""
    rows = []
    cols = []
    data = []
    
    for vertex_idx, weights in skinning_data.items():
        for joint_idx, weight in weights:
            rows.append(vertex_idx)
            cols.append(joint_idx)
            data.append(weight)
    
    sparse_skinning = sp.coo_matrix((data, (rows, cols)), shape=(num_vertices, num_joints))
    
    # Return as tuple of arrays which can be serialized
    return (sparse_skinning.data, sparse_skinning.row, sparse_skinning.col, sparse_skinning.shape)

def normalize_to_unit_cube(vertices, normals=None, scale_factor=1.0):
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (max_coords + min_coords) / 2.0
    
    vertices -= center
    scale = 1.0 / np.abs(vertices).max() * scale_factor
    vertices *= scale
    
    if normals is not None:
        # Normalize each normal vector to unit length
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms+1e-8)
    
        return vertices, normals, center, scale
    else:
        return vertices, center, scale

def normalize_vertices(vertices, scale=0.9):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, scale

def export_to_watertight(normalized_mesh, octree_depth: int = 7):
    """
        Convert the non-watertight mesh to watertight.

        Args:
            input_path (str): normalized path
            octree_depth (int):

        Returns:
            mesh(trimesh.Trimesh): watertight mesh

        """
    size = 2 ** octree_depth
    level = 2 / size

    scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(normalized_mesh.vertices)

    sdf = mesh2sdf.core.compute(scaled_vertices, normalized_mesh.faces, size=size)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)

    # watertight mesh
    vertices = vertices / size * 2 - 1 # -1 to 1
    vertices = vertices / to_orig_scale + to_orig_center
    mesh = trimesh.Trimesh(vertices, faces, normals=normals)

    return mesh

def process_mesh_to_pc(mesh, marching_cubes = True, sample_num = 8192):
    if marching_cubes:
        mesh = export_to_watertight(mesh)
    return_mesh = mesh
    points, face_idx = mesh.sample(sample_num, return_index=True)
    points, _, _ = normalize_to_unit_cube(points, scale_factor=0.9995)
    normals = mesh.face_normals[face_idx]

    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
    return pc_normal, return_mesh

def process_single_file(args):
    mesh_file, rig_file = args
    mesh_name = os.path.basename(mesh_file).split('.')[0]
    rig_name = os.path.basename(rig_file).split('.')[0]

    if mesh_name != rig_name:
        print(f"Skipping files {mesh_file} and {rig_file} because their names do not match.")
        return None
    
    vertices, faces, normals = read_obj_file(mesh_file)

    joints, bones, root, joint_names, skinning_data = read_rig_file(rig_file)

    # Normalize the mesh to the unit cube centered at the origin
    vertices, normals, center, scale = normalize_to_unit_cube(vertices, normals, scale_factor=0.5)
    
    # Apply the same transformation to joints
    joints -= center
    joints *= scale

    # Create trimesh object for processing
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Process into point cloud with normals
    pc_normal, _ = process_mesh_to_pc(mesh)
    
    # Convert skinning data to sparse format
    sparse_skinning = convert_to_sparse_skinning(skinning_data, len(vertices), len(joints))

    return {
        'vertices': vertices,
        'faces': faces,
        'normals': normals,
        'joints': joints,
        'bones': bones,
        'root_index': root,
        'uuid': mesh_name,
        'pc_w_norm': pc_normal,
        'joint_names': joint_names,
        'skinning_weights_value': sparse_skinning[0],  # values
        'skinning_weights_rows': sparse_skinning[1],  # row indices
        'skinning_weights_cols': sparse_skinning[2],  # column indices
        'skinning_weights_shape': sparse_skinning[3]  # shape of matrix
    }
        
def process_files(mesh_folder, rig_folder, output_file, num_workers=8):
    file_pairs = []

    for root, _, files in os.walk(rig_folder):
        for file in files:
            if file.endswith('.txt'):
                rig_file = os.path.join(root, file)
                obj_base_name = os.path.splitext(file)[0]
                mesh_file = os.path.join(mesh_folder, obj_base_name + '.obj')
                if os.path.exists(mesh_file):
                    file_pairs.append((mesh_file, rig_file))
                else:
                    print(f"Mesh file not found: {mesh_file}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        data_list = list(executor.map(process_single_file, file_pairs))
    
    data_list = [data for data in data_list if data is not None]
    
    np.savez_compressed(output_file, data_list, allow_pickle=True) 

def main():
    # Example usage
    mesh_folder = 'meshes/'
    rig_folder = 'rigs/'
    output_file = 'results.npz'

    process_files(mesh_folder, rig_folder, output_file)

if __name__ == "__main__":
    main()