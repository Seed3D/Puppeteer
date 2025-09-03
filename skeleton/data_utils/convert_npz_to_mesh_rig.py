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
You can convert npz file back to obj(mesh) and txt(rig) files using this python script.
"""
import os
import numpy as np
import scipy.sparse as sp

def export_obj(vertices, faces, normals, output_path):
    with open(output_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for n in normals:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        for i, face in enumerate(faces):
            # OBJ format is 1-based, so we add 1 to all indices
            f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")

def export_rig_txt(joints, bones, root_index, joint_names, skinning_weights, output_path):
    """
    joints [joint_name] [x] [y] [z]
    root [root_joint_name]
    skin [vertex_index] [joint_name1] [weight1] [joint_name2] [weight2] ...
    hier [parent_joint_name] [child_joint_name]
    """
    n_joints = len(joints)
    n_verts = skinning_weights.shape[0]  # (n_vertex, n_joints)

    with open(output_path, 'w') as f:
        # 1) joints
        for i in range(n_joints):
            x, y, z = joints[i]
            jn = joint_names[i]
            f.write(f"joints {jn} {x} {y} {z}\n")

        # 2) root
        root_name = joint_names[root_index]
        f.write(f"root {root_name}\n")

        # 3) skin 
        for vidx in range(n_verts):
            row_weights = skinning_weights[vidx]
            non_zero_indices = np.where(row_weights != 0)[0]
            if len(non_zero_indices) == 0:
                continue 
            
            line_parts = [f"skin {vidx}"]  # vertex_idx
            for jidx in non_zero_indices:
                w = row_weights[jidx]
                jn = joint_names[jidx]
                line_parts.append(jn)
                line_parts.append(str(w))
            
            f.write(" ".join(line_parts) + "\n")

        # 4) hier
        for p_idx, c_idx in bones:
            p_name = joint_names[p_idx]
            c_name = joint_names[c_idx]
            f.write(f"hier {p_name} {c_name}\n")

if __name__ == "__main__":
    
    data = np.load('articulation_xlv2_test.npz', allow_pickle=True)
    data_list = data['arr_0'] 

    print(f"Loaded {len(data_list)} data entries")

    model_data = data_list[0]
    print("Data keys:", model_data.keys())
    # 'vertices', 'faces', 'normals', 'joints', 'bones', 'root_index', 'uuid', 'joint_names',
    # 'skinning_weights_value', 'skinning_weights_row', 'skinning_weights_col', 'skinning_weights_shape'

    vertices = model_data['vertices']          # (n_vertex, 3)
    faces = model_data['faces']                # (n_faces, 3)
    normals = model_data['normals']            # (n_vertex, 3)
    joints = model_data['joints']              # (n_joints, 3)
    bones = model_data['bones']                # (n_bones, 2)
    root_index = model_data['root_index']      # int
    joint_names = model_data['joint_names']    # list of str
    uuid_str = model_data['uuid']             

    skin_val = model_data['skinning_weights_value']
    skin_row = model_data['skinning_weights_row']
    skin_col = model_data['skinning_weights_col']
    skin_shape = model_data['skinning_weights_shape']
    skin_sparse = sp.coo_matrix((skin_val, (skin_row, skin_col)), shape=skin_shape)
    skinning_weights = skin_sparse.toarray()  # (n_vertex, n_joints)

    obj_path = f"{uuid_str}.obj" 
    export_obj(vertices, faces, normals, obj_path)
    rig_txt_path = f"{uuid_str}.txt"
    export_rig_txt(joints, bones, root_index, joint_names, skinning_weights, rig_txt_path)

    print("Done!")
