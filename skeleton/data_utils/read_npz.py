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
import numpy as np
import scipy.sparse as sp

# Load the NPZ file
data = np.load('articulation_xlv2_test.npz', allow_pickle=True)
data_list = data['arr_0']

print(f"Loaded {len(data_list)} data entries")
print(f"Data keys: {data_list[0].keys()}")
# 'vertices', 'faces', 'normals', 'joints', 'bones', 'root_index', 'uuid', 'pc_w_norm', 'joint_names', 'skinning_weights_value', 
# 'skinning_weights_row', 'skinning_weights_col', 'skinning_weights_shape'

data = data_list[0] # check the first data

vertices = data['vertices'] # (n_vertex, 3)
faces = data['faces'] # (n_faces, 3)
normals = data['normals'] # (n_vertex, 3)
joints = data['joints'] # (n_joints, 3)
bones = data['bones'] # (n_bones, 2)
pc_w_norm = data['pc_w_norm'] # (8192, 6)

# Extract the sparse skinning weights components
skinning_data = data['skinning_weights_value']
skinning_rows = data['skinning_weights_row']
skinning_cols = data['skinning_weights_col']
skinning_shape = data['skinning_weights_shape']

skinning_sparse = sp.coo_matrix((skinning_data, (skinning_rows, skinning_cols)), shape=skinning_shape)
skinning_weights = skinning_sparse.toarray()  # (n_vertex, n_joints)

