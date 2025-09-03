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
from torch import is_tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from data_utils.save_npz import normalize_to_unit_cube

import numpy as np  
    
class SkeletonData(Dataset): 
    """
    A PyTorch Dataset to load and process skeleton data. 
    """
    def __init__(self, data, args, is_training): 
        self.data = data 
    
        self.input_pc_num = args.input_pc_num
        self.is_training = is_training
    
        self.hier_order = args.hier_order
        print(f"[Dataset] Created from {len(self.data)} entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        data = self.data[idx] 
        
        joints = data['joints']
        vertices = data['vertices']
        pc_normal = data['pc_w_norm']
        
        indices = np.random.choice(pc_normal.shape[0], self.input_pc_num, replace=False)
        pc_normal = pc_normal[indices, :]

        pc_coor = pc_normal[:, :3]
        normal = pc_normal[:, 3:]
        if np.linalg.norm(normal, axis=1, keepdims=True).min() < 0.99:
            print("normal reroll")
            return self.__getitem__(np.random.randint(0, len(self.data)))

        data_dict = {}
        
        # normalize normal
        normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)

        # scale to -0.5 to 0.5
        _, center, scale = normalize_to_unit_cube(vertices.copy(), scale_factor=0.9995)
        joints = (joints - center) * scale # align joints with pc first

        bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
        pc_center = (bounds[0] + bounds[1])[None, :]  / 2
        pc_scale = (bounds[1] - bounds[0]).max() + 1e-5
        pc_coor = (pc_coor - pc_center) / pc_scale
        joints = (joints - pc_center) / pc_scale

        joints = joints.clip(-0.5, 0.5)
       
        data_dict['joints'] = torch.from_numpy(np.asarray(joints).astype(np.float16))
        data_dict['bones'] = torch.from_numpy(data['bones'].astype(np.int64))
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
        data_dict['pc_normal'] = torch.from_numpy(np.concatenate([pc_coor, normal], axis=-1).astype(np.float16))
        data_dict['vertices'] = torch.from_numpy(data['vertices'].astype(np.float16))
        data_dict['faces'] = torch.from_numpy(data['faces'].astype(np.int64))
        data_dict['uuid'] = data['uuid']
        data_dict['root_index'] = str(data['root_index'])
        data_dict['transform_params'] = torch.tensor([
            center[0], center[1], center[2],
            scale,
            pc_center[0][0], pc_center[0][1], pc_center[0][2], 
            pc_scale
        ], dtype=torch.float32)
   
        return data_dict
    
    @classmethod
    def load(cls, args, is_training=True):   
        loaded_data = np.load(args.dataset_path, allow_pickle=True)  
        data = []
        for item in loaded_data["arr_0"]:
            data.append(item)  
        print(f"[Dataset] Loaded {len(data)} entries")
        return cls(data, args, is_training) 
        
 