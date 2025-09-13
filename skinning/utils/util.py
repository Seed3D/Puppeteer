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
import cv2
import json
import trimesh
import skimage.measure
import trimesh
import mesh2sdf.core

from utils.rig_parser import Info
from collections import deque, defaultdict
from scipy.cluster.hierarchy import linkage, fcluster

def read_obj_file(file_path):
    """Read OBJ file and return vertices and faces"""
    vertices = []
    faces = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()[1:]
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = [int(part.split('/')[0]) - 1 for part in parts]
                faces.append(face)
    
    return np.array(vertices), faces

def triangulate_faces(faces):
    """Convert quads and n-gons to triangles"""
    triangulated_faces = []
    for face in faces:
        if len(face) == 3:
            triangulated_faces.append(face)
        elif len(face) == 4:
            # for quad mesh
            triangulated_faces.append([face[0], face[1], face[2]])
            triangulated_faces.append([face[0], face[2], face[3]])
        elif len(face) > 4:
            for i in range(1, len(face) - 1):
                triangulated_faces.append([face[0], face[i], face[i + 1]])
    return np.array(triangulated_faces)

def read_rig_file(file_path):
    """Read rig file and return joints, bones, and root index"""
    joints = []
    bones = []
    joint_mapping = {}
    joint_index = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith('joints'):
            parts = line.split()
            name = parts[1]
            position = [float(parts[2]), float(parts[3]), float(parts[4])]
            joints.append(position)
            joint_mapping[name] = joint_index
            joint_index += 1
        elif line.startswith('hier'):
            parts = line.split()
            parent_joint = joint_mapping[parts[1]]
            child_joint = joint_mapping[parts[2]]
            bones.append([parent_joint, child_joint])
        elif line.startswith('root'):
            parts = line.split()
            root = joint_mapping[parts[1]]

    return np.array(joints), np.array(bones), root

def normalize_to_unit_cube(vertices, scale_factor=1.0):
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (max_coords + min_coords) / 2.0
    
    vertices -= center
    scale = 1.0 / np.abs(vertices).max() * scale_factor
    vertices *= scale
    
    return vertices, center, scale

def build_adjacency_list(num_joints, bones):
    """Build adjacency list for graph distance computation"""
    adjacency = [[] for _ in range(num_joints)]
    for (p, c) in bones:
        adjacency[p].append(c)
        adjacency[c].append(p)
    return adjacency

def compute_graph_distance(num_joints, adjacency):
    """Compute graph distance using BFS"""
    graph_dist = np.full((num_joints, num_joints), np.inf, dtype=np.float32)

    for start in range(num_joints):
        queue = deque()
        queue.append((start, 0))
        graph_dist[start, start] = 0.0

        while queue:
            current, dist = queue.popleft()
            for nbr in adjacency[current]:
                if graph_dist[start, nbr] == np.inf:
                    graph_dist[start, nbr] = dist + 1
                    queue.append((nbr, dist + 1))

    return graph_dist
    
def get_tpl_edges(vertices, faces):
    """Get topology edges from mesh (handles any polygon type)"""
    edges = set()
    
    for face in faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            edge = tuple(sorted([v1, v2]))
            edges.add(edge)
    
    return np.array(list(edges))

def save_args(args, output_dir, filename="config.json"):
    args_dict = vars(args)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

def save_skin_weights_to_rig(rig_path, skin_weights, output_path):
    """
    save skinning weights to rig file, keeping the original joints, root and hier information unchanged.
  
    parameters:
    rig_path: original rig path
    skin_weights: predicted skinning weights
    output_path: output rig path
    """
   
    original_rig = Info(rig_path)
    
    joints_name = list(original_rig.joint_pos.keys())
    
    skin_lines = []
    for v in range(len(skin_weights)):
        vi_skin = [str(v)]
        skw = skin_weights[v]
        skw = skw / (np.sum(skw))
        
        for i in range(len(skw)):
            if i == len(joints_name):
                break
            if skw[i] > 1e-5: 
                bind_joint_name = joints_name[i]
                bind_weight = skw[i]
                vi_skin.append(bind_joint_name)
                vi_skin.append(str(bind_weight))
        skin_lines.append(vi_skin)
    
    with open(rig_path, 'r') as f_in:
        original_lines = f_in.readlines()
    
    preserved_lines = []
    for line in original_lines:
        word = line.split()
        if word[0] in ['joints', 'root', 'hier']:
            preserved_lines.append(line)
    
    with open(output_path, 'w') as f_out:
        for line in preserved_lines:
            f_out.write(line)
        
        for skw in skin_lines:
            cur_line = 'skin {0} '.format(skw[0])
            for cur_j in range(1, len(skw), 2):
                cur_line += '{0} {1:.6f} '.format(skw[cur_j], float(skw[cur_j+1]))
            cur_line += '\n'
            f_out.write(cur_line)

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

def process_mesh_to_pc(mesh, marching_cubes = True, sample_num = 4096):
   
    if marching_cubes:
        mesh = export_to_watertight(mesh)
        print("MC over!")
    return_mesh = mesh
    points, face_idx = mesh.sample(sample_num, return_index=True)
    points, _, _ = normalize_to_unit_cube(points, 0.9995)
    normals = mesh.face_normals[face_idx]

    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
    return pc_normal, return_mesh

def post_filter(skin_weights, topology_edge, num_ring=1):
    """
    Post-process skinning weights by averaging over multi-ring neighbors.
    
    Parameters:
    skin_weights: (num_vertices, num_joints) array of skinning weights
    topology_edge: (num_edges, 2) array of edges defining the mesh topology
    num_ring: number of rings for neighbor averaging
    
    Returns:
    skin_weights_new: (num_vertices, num_joints) array of post-processed skin
    """
    skin_weights_new = np.zeros_like(skin_weights)
    num_vertices = skin_weights.shape[0]
    
    adjacency_list = [[] for _ in range(num_vertices)]
    for e in range(topology_edge.shape[0]): 
        v1, v2 = topology_edge[e, 0], topology_edge[e, 1]
        adjacency_list[v1].append(v2)
    
    for v in range(num_vertices):
        adj_verts_multi_ring = set()
        visited = {v}
        current_ring = {v}
        
        for r in range(num_ring):
            next_ring = set()
            for seed in current_ring:
                for neighbor in adjacency_list[seed]:
                    if neighbor not in visited:
                        next_ring.add(neighbor)
                        visited.add(neighbor)
            
            adj_verts_multi_ring.update(next_ring)
            if not next_ring:
                break
                
            current_ring = next_ring
        
        # calculate the average skinning weights
        adj_verts_multi_ring.discard(v)
        if adj_verts_multi_ring:
            skin_weights_neighbor = skin_weights[list(adj_verts_multi_ring), :]
            skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)
        else:
            skin_weights_new[v, :] = skin_weights[v, :]
    
    return skin_weights_new