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

from collections import deque, defaultdict
from scipy.cluster.hierarchy import linkage, fcluster

from data_utils.pyrender_wrapper import PyRenderWrapper
from data_utils.data_loader import DataLoader

def save_mesh(vertices, faces, filename):

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)    
    mesh.export(filename, file_type='obj')

def pred_joints_and_bones(bone_coor):
    """
    get joints (j,3) and bones (b,2) from (b,2,3), preserve the parent-child relationship
    """
    parent_coords = bone_coor[:, 0, :]  # (b, 3)
    child_coords = bone_coor[:, 1, :]   # (b, 3)

    all_coords = np.vstack([parent_coords, child_coords])  # (2b, 3)
    pred_joints, indices = np.unique(all_coords, axis=0, return_inverse=True)

    b = bone_coor.shape[0]
    parent_indices = indices[:b]
    child_indices = indices[b:]

    pred_bones = np.column_stack([parent_indices, child_indices])
    
    valid_bones = pred_bones[parent_indices != child_indices]
    
    return pred_joints, valid_bones

def find_connected_components(joints, bones):
    """Find connected components in the skeleton graph."""
    n_joints = len(joints)
    graph = defaultdict(list)
    
    # Build adjacency list
    for parent, child in bones:
        graph[parent].append(child)
        graph[child].append(parent)
    
    visited = [False] * n_joints
    components = []
    
    for i in range(n_joints):
        if not visited[i]:
            component = []
            queue = deque([i])
            visited[i] = True
            
            while queue:
                node = queue.popleft()
                component.append(node)
                
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            components.append(component)
    
    return components

def ensure_skeleton_connectivity(joints, bones, root_index=None, merge_distance_threshold=0.01):
    """
    Ensure skeleton is fully connected.
    - If distance < merge_distance_threshold: merge joints
    - If distance >= merge_distance_threshold: connect with bone
    """
    current_joints = joints.copy()
    current_bones = list(bones)
    current_root = root_index
    
    iteration = 0
    while True:
        components = find_connected_components(current_joints, current_bones)
        if len(components) == 1:
            # print("Successfully ensured skeleton connectivity")
            break
            
        # if iteration == 0:
        #     print(f"Found {len(components)} disconnected components, connecting them progressively...")
        
        # Find the globally closest pair of components
        min_distance = float('inf')
        best_pair = None
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp1_joints = current_joints[components[i]]
                comp2_joints = current_joints[components[j]]
                
                distances = cdist(comp1_joints, comp2_joints)
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                distance = distances[min_idx]
                
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (i, j, components[i][min_idx[0]], components[j][min_idx[1]], min_idx)
        
        if best_pair is None:
            print("Warning: Could not find valid component pair to connect")
            break
        
        comp1_idx, comp2_idx, joint1_idx, joint2_idx, min_idx = best_pair
        
        if min_distance < merge_distance_threshold:
            # Merge the joints
            # print(f"Iteration {iteration + 1}: Merging closest joints {joint1_idx} and {joint2_idx} "
            #       f"(distance: {min_distance:.4f})")
            
            # Always merge joint2 into joint1
            merge_map = {joint2_idx: joint1_idx}
            
            # Update bones
            updated_bones = []
            for parent, child in current_bones:
                new_parent = merge_map.get(parent, parent)
                new_child = merge_map.get(child, child)
                if new_parent != new_child:  # Remove self-loops
                    updated_bones.append([new_parent, new_child])
            
            # Update root
            if current_root == joint2_idx:
                current_root = joint1_idx
            
            # Remove the merged joint and update indices
            joint_to_remove = joint2_idx
            mask = np.ones(len(current_joints), dtype=bool)
            mask[joint_to_remove] = False
            current_joints = current_joints[mask]
            
            # Create index mapping for remaining joints
            old_to_new = {}
            new_idx = 0
            for old_idx in range(len(mask)):
                if mask[old_idx]:
                    old_to_new[old_idx] = new_idx
                    new_idx += 1
            
            # Update bone indices
            current_bones = [[old_to_new[parent], old_to_new[child]] 
                           for parent, child in updated_bones 
                           if parent in old_to_new and child in old_to_new]
            
            # Update root index
            if current_root is not None and current_root in old_to_new:
                current_root = old_to_new[current_root]
            
        else:
            # Connect with bone
            # print(f"Iteration {iteration + 1}: Connecting closest components with bone {joint1_idx} -> {joint2_idx} "
            #       f"(distance: {min_distance:.4f})")
            current_bones.append([joint1_idx, joint2_idx])
        
        iteration += 1
        
        # prevent infinite loops
        if iteration > len(joints):
            print(f"Warning: Maximum iterations reached ({iteration}), stopping")
            break
    
    current_bones = np.array(current_bones) if len(current_bones) > 0 else np.array([]).reshape(0, 2)
    
    # Final connectivity verification
    final_components = find_connected_components(current_joints, current_bones)
    if len(final_components) == 1:
        pass
    else:
        print(f"Warning: Still have {len(final_components)} disconnected components after {iteration} iterations")
    
    return current_joints, current_bones, current_root

def merge_duplicate_joints_and_fix_bones(joints, bones, tolerance=0.0025, root_index=None):
    """
    merge duplicate joints that are within a certain tolerance distance, and fix bones to maintain connectivity.
    Also merge bones that become duplicates after joint merging.
    """
    n_joints = len(joints)
    
    # find merge joint groups
    merge_groups = []
    used = [False] * n_joints
    
    for i in range(n_joints):
        if used[i]:
            continue
            
        # find all joints within tolerance distance to joint i
        group = [i]
        for j in range(i + 1, n_joints):
            if not used[j]:
                dist = np.linalg.norm(joints[i] - joints[j])
                if dist < tolerance:
                    group.append(j)
                    used[j] = True
        
        used[i] = True
        merge_groups.append(group)
        
        # if len(group) > 1:
        #     print(f"find duplicate joints group: {group}")
    
    # build merge map: choose representative joint
    merge_map = {}
    for group in merge_groups:
        if root_index is not None and root_index in group:
            representative = root_index
        else:
            representative = group[0]  # else choose the first one as representative
        for joint_idx in group:
            merge_map[joint_idx] = representative
    
    # track root joint change
    intermediate_root_index = None
    if root_index is not None:
        intermediate_root_index = merge_map.get(root_index, root_index)
        # if intermediate_root_index != root_index:
        #     print(f"root joint index changed from {root_index} to {intermediate_root_index}")
    
    # update bones: remove self-loop bones, and merge duplicate bones
    updated_bones = []
    
    for parent, child in bones:
        new_parent = merge_map.get(parent, parent)
        new_child = merge_map.get(child, child)
        
        if new_parent != new_child: # remove self-loop bones
            updated_bones.append([new_parent, new_child])
    
    # remove duplicate bones
    unique_bones = []
    seen_bones = set()
    
    for bone in updated_bones:
        bone_key = tuple(bone)  # keep the order of [parent, child]
        if bone_key not in seen_bones:
            seen_bones.add(bone_key)
            unique_bones.append(bone)
    
    # re-index joints to remove unused joints
    used_joint_indices = set()
    for parent, child in unique_bones:
        used_joint_indices.add(parent)
        used_joint_indices.add(child)
    if intermediate_root_index is not None:
        used_joint_indices.add(intermediate_root_index)
    
    
    used_joint_indices = sorted(list(used_joint_indices))
    
    # new index for used joints
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_joint_indices)}
    
    final_joints = joints[used_joint_indices]
    final_bones = np.array([[old_to_new[parent], old_to_new[child]] 
                           for parent, child in unique_bones])
    
    final_root_index = None
    if intermediate_root_index is not None:
        final_root_index = old_to_new[intermediate_root_index]
        if root_index is not None and final_root_index != root_index:
            print(f"final root index: {root_index} -> {final_root_index}")
    
    removed_joints = n_joints - len(final_joints)
    removed_bones = len(bones) - len(final_bones)
    
    # print
    # if removed_joints > 0 or removed_bones > 0:
    #     print(f"merge results:")
    #     print(f"  joint number: {n_joints} -> {len(final_joints)} (remove {removed_joints})")
    #     print(f"  bone number: {len(bones)} -> {len(final_bones)} (remove {removed_bones})")

    # Ensure skeleton connectivity with relaxed threshold
    final_joints, final_bones, final_root_index = ensure_skeleton_connectivity(
        final_joints, final_bones, final_root_index, 
        merge_distance_threshold=tolerance*8  # More relaxed threshold for connectivity
    )
    
    if root_index is not None:
        return final_joints, final_bones, final_root_index
    else:
        return final_joints, final_bones

def save_skeleton_to_txt(pred_joints, pred_bones, pred_root_index, hier_order, vertices, filename='skeleton.txt'):
    """
    save skeleton to txt file, the format follows Rignet (joints, root, hier)
    
    if hier_order: the first joint index in bone is root joint index, and parent-child relationship is established in bones.
    else: we set the joint nearest to the mesh center as the root joint, and then build hierarchy starting from root.
    """
    
    num_joints = pred_joints.shape[0]
    
    # assign joint names
    joint_names = [f'joint{i}' for i in range(num_joints)]
    
    adjacency = defaultdict(list)
    for bone in pred_bones:
        idx_a, idx_b = bone
        adjacency[idx_a].append(idx_b)
        adjacency[idx_b].append(idx_a)
    
    # find root joint
    if hier_order:
        root_idx = pred_root_index
    else:
        centroid = np.mean(vertices, axis=0)
        distances = np.linalg.norm(pred_joints - centroid, axis=1)
        root_idx = np.argmin(distances)
    
    root_name = joint_names[root_idx]
    
    # build hierarchy
    parent_map = {}
    
    if hier_order:
        visited = set()
        
        for parent_idx, child_idx in pred_bones:
            if child_idx not in parent_map:
                parent_map[child_idx] = parent_idx
                visited.add(child_idx)
                visited.add(parent_idx)

        parent_map[root_idx] = None

    else:
        visited = set([root_idx])
        queue = deque([root_idx])
        parent_map[root_idx] = None
        
        while queue:
            current_idx = queue.popleft()
            for neighbor_idx in adjacency[current_idx]:
                if neighbor_idx not in visited:
                    parent_map[neighbor_idx] = current_idx
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
                
    if len(visited) != num_joints:
        print(f"bones are not fully connected, leaving {num_joints - len(visited)} joints unconnected.")
    
    # save joints
    joints_lines = []
    for idx, coord in enumerate(pred_joints):
        name = joint_names[idx]
        joints_line = f'joints {name} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}'
        joints_lines.append(joints_line)
    
    # save root name
    root_line = f'root {root_name}'
    
    # save hierarchy
    hier_lines = []
    for child_idx, parent_idx in parent_map.items():
        if parent_idx is not None:
            parent_name = joint_names[parent_idx]
            child_name = joint_names[child_idx]
            hier_line = f'hier {parent_name} {child_name}'
            hier_lines.append(hier_line)
    
    with open(filename, 'w') as file:
        for line in joints_lines:
            file.write(line + '\n')

        file.write(root_line + '\n')

        for line in hier_lines:
            file.write(line + '\n')

def save_skeleton_to_txt_joint(pred_joints, pred_bones, filename='skeleton.txt'):
    """
    save skeleton to txt file, the format follows Rignet (joints, root, hier)
    """
    
    num_joints = pred_joints.shape[0]
    
    # assign joint names
    joint_names = [f'joint{i}' for i in range(num_joints)]
    
    # find potential root joints
    all_parents = set([bone[0] for bone in pred_bones])
    all_children = set([bone[1] for bone in pred_bones])
    potential_roots = all_parents - all_children
    
    # determine root joint
    if not potential_roots:
        print("Warning: No joint is only a parent, choosing the first joint as root.")
        root_idx = pred_bones[0, 0]
    else:
        if len(potential_roots) > 1:
            print(f"Warning: Multiple potential root joints found ({len(potential_roots)}), choosing the first one.")
        root_idx = list(potential_roots)[0]
    
    root_name = joint_names[root_idx]
    
    # build hierarchy
    parent_map = {}
    visited = set()
    
    for parent_idx, child_idx in pred_bones:
        if child_idx not in parent_map:
            parent_map[child_idx] = parent_idx
            visited.add(child_idx)
            visited.add(parent_idx)

    parent_map[root_idx] = None
    
    if len(visited) != num_joints:
        print(f"Warning: bones are not fully connected, leaving {num_joints - len(visited)} joints unconnected.")
    
    # save joints
    joints_lines = []
    for idx, coord in enumerate(pred_joints):
        name = joint_names[idx]
        joints_line = f'joints {name} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}'
        joints_lines.append(joints_line)
    
    # save root name
    root_line = f'root {root_name}'
    
    # save hierarchy
    hier_lines = []
    for child_idx, parent_idx in parent_map.items():
        if parent_idx is not None:
            parent_name = joint_names[parent_idx]
            child_name = joint_names[child_idx]
            hier_line = f'hier {parent_name} {child_name}'
            hier_lines.append(hier_line)
    
    with open(filename, 'w') as file:
        for line in joints_lines:
            file.write(line + '\n')

        file.write(root_line + '\n')

        for line in hier_lines:
            file.write(line + '\n')
    return root_idx
     

def save_skeleton_obj(joints, bones, save_path, root_index=None, radius_sphere=0.01, 
                     radius_bone=0.005, segments=16, stacks=16, use_cone=False):
    """
    Save skeletons to obj file, each connection contains two red spheres (joint) and one blue cylinder (bone).
    if root index is known, set root sphere to green.
    """
    
    all_vertices = []
    all_colors = []
    all_faces = []
    vertex_offset = 0
    
    # create spheres for joints
    for i, joint in enumerate(joints):
        # define color
        if root_index is not None and i == root_index:
            color = (0, 1, 0)  # green for root joint
        else:
            color = (1, 0, 0)  # red for other joints
        
        # create joint sphere
        sphere_vertices, sphere_faces = create_sphere(joint, radius=radius_sphere, segments=segments, stacks=stacks)
        all_vertices.extend(sphere_vertices)
        all_colors.extend([color] * len(sphere_vertices))
        
        # adjust face index
        adjusted_sphere_faces = [(v1 + vertex_offset, v2 + vertex_offset, v3 + vertex_offset) for (v1, v2, v3) in sphere_faces]
        all_faces.extend(adjusted_sphere_faces)
        vertex_offset += len(sphere_vertices)
    
    # create bones
    for bone in bones:
        parent_idx, child_idx = bone
        parent = joints[parent_idx]
        child = joints[child_idx]
        
        try:
            bone_vertices, bone_faces = create_bone(parent, child, radius=radius_bone, segments=segments, use_cone=use_cone)
        except ValueError as e:
            print(f"Skipping connection {parent_idx}-{child_idx}, reason: {e}")
            continue
            
        all_vertices.extend(bone_vertices)
        all_colors.extend([(0, 0, 1)] * len(bone_vertices))  # blue
        
        # adjust face index
        adjusted_bone_faces = [(v1 + vertex_offset, v2 + vertex_offset, v3 + vertex_offset) for (v1, v2, v3) in bone_faces]
        all_faces.extend(adjusted_bone_faces)
        vertex_offset += len(bone_vertices)

    # save to obj
    obj_lines = []
    for v, c in zip(all_vertices, all_colors):
        obj_lines.append(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}")
    obj_lines.append("") 

    for face in all_faces:
        obj_lines.append(f"f {face[0]} {face[1]} {face[2]}")
        
    with open(save_path, 'w') as obj_file:
        obj_file.write("\n".join(obj_lines))

def create_sphere(center, radius=0.01, segments=16, stacks=16):
    vertices = []
    faces = []
    for i in range(stacks + 1):
        lat = np.pi / 2 - i * np.pi / stacks
        xy = radius * np.cos(lat)
        z = radius * np.sin(lat)
        for j in range(segments):
            lon = j * 2 * np.pi / segments
            x = xy * np.cos(lon) + center[0]
            y = xy * np.sin(lon) + center[1]
            vertices.append((x, y, z + center[2]))
    for i in range(stacks):
        for j in range(segments):
            first = i * segments + j
            second = first + segments
            third = first + 1 if (j + 1) < segments else i * segments
            fourth = second + 1 if (j + 1) < segments else (i + 1) * segments
            faces.append((first + 1, second + 1, fourth + 1))
            faces.append((first + 1, fourth + 1, third + 1))
    return vertices, faces
    
def create_bone(start, end, radius=0.005, segments=16, use_cone=False):
    dir_vector = np.array(end) - np.array(start)
    height = np.linalg.norm(dir_vector)
    if height == 0:
        raise ValueError("Start and end points cannot be the same for a cone.")
    dir_vector = dir_vector / height

    z = np.array([0, 0, 1])
    if np.allclose(dir_vector, z):
        R = np.identity(3)
    elif np.allclose(dir_vector, -z):
        R = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    else:
        v = np.cross(z, dir_vector)
        s = np.linalg.norm(v)
        c = np.dot(z, dir_vector)
        kmat = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
        R = np.identity(3) + kmat + np.matmul(kmat, kmat) * ((1 - c) / (s**2))

    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    base_circle = np.array([np.cos(theta), np.sin(theta), np.zeros(segments)]) * radius
    
    vertices = []
    for point in base_circle.T:
        rotated = np.dot(R, point) + np.array(start)
        vertices.append(tuple(rotated))
        

    faces = []
    
    if use_cone:
        vertices.append(tuple(end))

        apex_idx = segments + 1 
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((i + 1, next_i + 1, apex_idx))
    else:
        top_circle = np.array([np.cos(theta), np.sin(theta), np.ones(segments)]) * radius
        for point in top_circle.T:
            point_scaled = np.array([point[0], point[1], height])
            rotated = np.dot(R, point_scaled) + np.array(start)
            vertices.append(tuple(rotated))
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((i + 1, next_i + 1, next_i + segments + 1))
            faces.append((i + 1, next_i + segments + 1, i + segments + 1))
    
    return vertices, faces

def render_mesh_with_skeleton(joints, bones, vertices, faces, output_dir, filename, prefix='pred', root_idx=None):
    """
    Render the mesh with skeleton using PyRender.
    """
    loader = DataLoader()
    
    raw_size = (960, 960)
    renderer = PyRenderWrapper(raw_size)
    
    save_dir = os.path.join(output_dir, 'render_results')
    os.makedirs(save_dir, exist_ok=True)
    
    loader.joints = joints
    loader.bones = bones
    loader.root_idx = root_idx
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.vertex_colors[:, 3] = 100  # set transparency
    loader.mesh = mesh
    v = mesh.vertices
    xmin, ymin, zmin = v.min(axis=0)
    xmax, ymax, zmax = v.max(axis=0)
    loader.bbox_center = np.array([(xmax + xmin)/2, (ymax + ymin)/2, (zmax + zmin)/2])
    loader.bbox_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    loader.bbox_scale = max(xmax - xmin, ymax - ymin, zmax - zmin)
    loader.normalize_coordinates()
    
    input_dict = loader.query_mesh_rig()
    
    angles = [0, np.pi/2, np.pi, 3*np.pi/2] 
    distance = np.max(loader.bbox_size) * 2
    
    subfolder_path = os.path.join(save_dir, filename + '_' + prefix)
    
    os.makedirs(subfolder_path, exist_ok=True)
    
    for i, angle in enumerate(angles):
        renderer.set_camera_view(angle, loader.bbox_center, distance)
        renderer.align_light_to_camera()

        color = renderer.render(input_dict)[0]
        
        output_filename = f"{filename}_{prefix}_view{i+1}.png"
        output_filepath = os.path.join(subfolder_path, output_filename)
        cv2.imwrite(output_filepath, color)
    

def save_args(args, output_dir, filename="config.json"):
    args_dict = vars(args)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4)