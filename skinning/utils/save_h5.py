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
import h5py
from tqdm import tqdm
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from utils.util import read_obj_file, read_rig_file, normalize_to_unit_cube, build_adjacency_list, compute_graph_distance, get_tpl_edges

@dataclass
class ProcessedSample:
    """Data structure for a processed sample."""
    vertices: np.ndarray
    faces: np.ndarray
    joints: np.ndarray
    bones: np.ndarray
    root_index: int
    pc_w_norm: np.ndarray
    file_name: str
    skin: np.ndarray
    graph_dist: np.ndarray
    edges: np.ndarray

def process_sample(data: Dict[str, Any]) -> Optional[ProcessedSample]:
    """
    Process a single sample from the dataset.
    
    Args:
        data: Dictionary containing sample data
        
    Returns:
        ProcessedSample object or None if processing fails
    """
    vertices = data['vertices'].copy()
    joints = data['joints'].copy()
    if len(joints) > 70: # filter out data with too many joints
        return None
    
    vertices, center, scale = normalize_to_unit_cube(vertices, 0.9995)
    joints = (joints - center) * scale

    # Build skinning weights matrix
    skinning_data = data['skinning_weights_value']
    skinning_rows = data['skinning_weights_row']
    skinning_cols = data['skinning_weights_col']
    skinning_shape = data['skinning_weights_shape']

    skinning_sparse = sp.coo_matrix(
        (skinning_data, (skinning_rows, skinning_cols)), 
        shape=skinning_shape
    )
    skinning_weights = skinning_sparse.toarray()  # (n_vertex, n_joints)
    
    # Compute topology and graph features
    edges = get_tpl_edges(data['vertices'], data['faces'])
    num_joints = len(data['joints'])
    adjacency = build_adjacency_list(num_joints, data['bones'])
    graph_dist = compute_graph_distance(num_joints, adjacency)

    return ProcessedSample(
        vertices=vertices,
        faces=data['faces'],
        joints=joints,
        bones=data['bones'],
        root_index=data['root_index'],
        pc_w_norm=data['pc_w_norm'],
        file_name=data['uuid'],
        skin=skinning_weights,
        graph_dist=graph_dist,
        edges=edges
    )

def parallel_process_samples(
    data_list: List[Dict[str, Any]], 
    max_workers: Optional[int] = None
) -> List[ProcessedSample]:
    """
    Process multiple samples in parallel.
    
    Args:
        data_list: List of sample dictionaries
        max_workers: Maximum number of worker processes
        
    Returns:
        List of successfully processed samples
    """
    processed_samples = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_sample, data): data for data in data_list}
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing samples'):
            try:
                result = future.result()
                if result is not None:
                    processed_samples.append(result)
                else:
                    original_data = futures[future]
            except Exception as e:
                original_data = futures[future]
                print(f"Exception in processing {original_data.get('file_name', 'unknown')}: {e}")
                    
    return processed_samples

def save_to_h5(processed_samples: List[ProcessedSample], output_path: str) -> None:
    """
    Save processed samples to HDF5 file.
    
    Args:
        processed_samples: List of processed samples
        output_path: Output HDF5 file path
    """
    with h5py.File(output_path, 'w') as f:
        # Add metadata
        f.attrs['num_samples'] = len(processed_samples)
        f.attrs['version'] = '1.0'
        
        for i, sample in enumerate(tqdm(processed_samples, desc='Saving to HDF5')):
            grp = f.create_group(f'sample_{i}')

            # Save arrays with compression
            grp.create_dataset('joints', data=sample.joints, compression='gzip')
            grp.create_dataset('bones', data=sample.bones, compression='gzip')
            grp.create_dataset('root_index', data=sample.root_index, dtype='i')
            grp.create_dataset('pc_w_norm', data=sample.pc_w_norm, compression='gzip')
            grp.create_dataset('vertices', data=sample.vertices, compression='gzip')
            grp.create_dataset('faces', data=sample.faces, compression='gzip')
            grp.create_dataset('edges', data=sample.edges, compression='gzip')
            grp.create_dataset('skin', data=sample.skin, compression='gzip')
            grp.create_dataset('graph_dist', data=sample.graph_dist, compression='gzip')
            string_dtype = h5py.special_dtype(vlen=str)
            grp.create_dataset('file_name', data=sample.file_name, dtype=string_dtype)


def main(npz_file_path, h5_file_path, max_workers):
    loaded_data = np.load(npz_file_path, allow_pickle=True)
    data_list = loaded_data['arr_0']

    num_samples = len(data_list)
    print(f"Total samples: {num_samples}")

    processed_samples = parallel_process_samples(
        data_list=data_list,
        max_workers=max_workers
    )
    save_to_h5(processed_samples, h5_file_path)
    print("Processing complete!")

if __name__ == '__main__':
    npz_file_path = 'articulation_xlv2_test.npz'
    h5_file_path = 'articulation_xlv2_test.h5'
    main(npz_file_path, h5_file_path, max_workers=8)
