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
Blender script for extracting rig (.txt) and mesh (.obj) from glbs. 
This code currently supports GLB files only, but it can be easily modified to load other formats (e.g., FBX, DAE) with minimal changes.
"""

import bpy
import os
import re
import json
import pickle

def get_hierarchy_root_joint(joint):
    """
    Function to find the top parent joint node from the given 
    'joint' Blender node (armature bone).
    """
    root_joint = joint
    while root_joint.parent is not None:
        root_joint = root_joint.parent
    return root_joint

def get_meshes_and_armatures():
    """
    Function to get all meshes and armatures in the scene
    """
    default_objects = ['Cube', 'Light', 'Camera', 'Icosphere']
    for obj_name in default_objects:
        if obj_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
    
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    return meshes, armatures

def get_joint_dict(root):
    """
    Function to create a dictionary of joints from the root joint
    """
    joint_pos = {}
    def traverse_bone(bone):
        joint_pos[bone.name] = {
            'pos': bone.head_local,
            'pa': bone.parent.name if bone.parent else 'None',
            'ch': [child.name for child in bone.children]
        }
        for child in bone.children:
            traverse_bone(child)

    traverse_bone(root)
    return joint_pos

def record_info(root, joint_dict, meshes, mesh_vert_offsets, file_info):
    """
    - root: root joint
    - joint_dict
    - meshes
    - mesh_vert_offsets: for multi-geometry
    - file_info
    """
    skin_records = {}

    def replace_special_characters(name):
        return re.sub(r'\W+', '_', name)

    for key, val in joint_dict.items():
        modified_key = replace_special_characters(key)
        file_info.write(f'joints {modified_key} {val["pos"][0]:.8f} {val["pos"][1]:.8f} {val["pos"][2]:.8f}\n')
    file_info.write(f'root {replace_special_characters(root.name)}\n')
    
    for mesh_index, mesh in enumerate(meshes):
        vert_offset = mesh_vert_offsets[mesh_index]
        if mesh.type == 'MESH':
            for vtx in mesh.data.vertices:
                weights = {}
                for group in vtx.groups:
                    bone_name = replace_special_characters(mesh.vertex_groups[group.group].name)
                    weights[bone_name] = group.weight

                global_vertex_index = vert_offset + vtx.index

                skin_record = f"skin {global_vertex_index} " + " ".join(f"{bone} {weight:.4f}" for bone, weight in weights.items())
                
                if global_vertex_index not in skin_records:
                    skin_records[global_vertex_index] = skin_record
                    file_info.write(skin_record + "\n")
    
    for key, val in joint_dict.items():
        if val['pa'] != 'None':
            parent_name = replace_special_characters(val['pa'])
            child_name = replace_special_characters(key)
            file_info.write(f'hier {parent_name} {child_name}\n')


def record_obj(meshes, file_obj):
    vert_offset = 0
    norm_offset = 0
    mesh_vert_offsets = []

    for mesh in meshes:
        mesh_vert_offsets.append(vert_offset)
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # vertex
        for v in mesh.data.vertices:
            file_obj.write(f"v {v.co[0]} {v.co[1]} {v.co[2]}\n")
        file_obj.write("\n")
        
        # normal
        for vn in mesh.data.vertices:
            normal = vn.normal
            file_obj.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
        file_obj.write("\n")
        
        # face
        for poly in mesh.data.polygons:
            verts = [v + 1 + vert_offset for v in poly.vertices] 
            file_obj.write(f"f {verts[0]}//{verts[0]} {verts[1]}//{verts[1]} {verts[2]}//{verts[2]}\n")
        
        vert_count = len(mesh.data.vertices)
        vert_offset += vert_count
        norm_offset += vert_count

    return mesh_vert_offsets

def process_glb(glb_path, rigs_dir, meshes_dir):
    base_name = os.path.splitext(os.path.basename(glb_path))[0]
    
    obj_name = os.path.join(meshes_dir, f'{base_name}.obj')
    info_name = os.path.join(rigs_dir, f'{base_name}.txt')
    
    # Skip processing if rig info file already exists
    if os.path.exists(info_name):
        print(f"{info_name} already exists. Skipping...")
        return
    
    if os.path.exists(obj_name):
        print(f"{obj_name} already exists. Skipping...")
        return
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    
    meshes, armatures = get_meshes_and_armatures()
    
    if not armatures:
        print(f"No armatures found in {glb_path}. Skipping...")
        return
    
    root = armatures[0].data.bones[0]
    root_name = get_hierarchy_root_joint(root)
    joint_dict = get_joint_dict(root_name)
    
    #  save meshes
    with open(obj_name, 'w') as file_obj:
        mesh_vert_offsets = record_obj(meshes, file_obj)
    
    # save rigs
    with open(info_name, 'w') as file_info:
        record_info(root_name, joint_dict, meshes, mesh_vert_offsets, file_info)
    
    print(f"Processed {glb_path}")

if __name__ == '__main__':
    
    src_dir = 'glbs'
    rigs_dir = 'rigs'
    meshes_dir = 'meshes'
    # Ensure rigs directory exists
    if not os.path.exists(rigs_dir):
        os.makedirs(rigs_dir)
    if not os.path.exists(meshes_dir):
        os.makedirs(meshes_dir)

    glb_paths = [os.path.join(src_dir, file) for file in os.listdir(src_dir) if file.endswith('.glb')]
    
    print(len(glb_paths))
    
    for glb_path in glb_paths:
        try:
            process_glb(glb_path, rigs_dir, meshes_dir)
        except Exception as e:
            with open('error.txt', 'a') as error_file:
                error_file.write(f"{glb_path}: {str(e)}\n")