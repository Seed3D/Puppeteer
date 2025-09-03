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
import torch
import trimesh
import argparse
import numpy as np

from tqdm import tqdm
from trimesh import Scene

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs

from skeleton_models.skeletongen import SkeletonGPT
from data_utils.save_npz import normalize_to_unit_cube
from utils.mesh_to_pc import MeshProcessor
from utils.save_utils import save_mesh, pred_joints_and_bones, save_skeleton_to_txt, save_skeleton_to_txt_joint, save_args, \
                        merge_duplicate_joints_and_fix_bones, save_skeleton_obj, render_mesh_with_skeleton
                       
class Dataset:
    def __init__(self, input_list, input_pc_num = 8192, apply_marching_cubes = True, octree_depth = 7, output_dir = None):
        super().__init__()
        self.data = []
        self.output_dir = output_dir
    
        mesh_list = []
        for input_path in input_list:
            ext = os.path.splitext(input_path)[1].lower()
            if ext in ['.ply', '.stl', '.obj']: 
                cur_data = trimesh.load(input_path, force='mesh')
                mesh_list.append(cur_data)
            else:
                print(f"Unsupported file type: {ext}")
        if apply_marching_cubes:
            print("First apply Marching Cubes and then sample point cloud, need time...")
        pc_list = MeshProcessor.convert_meshes_to_point_clouds(mesh_list, input_pc_num, apply_marching_cubes = apply_marching_cubes, octree_depth = octree_depth)
        for input_path, cur_data, mesh in zip(input_list, pc_list, mesh_list):
            self.data.append({'pc_normal': cur_data, 'faces': mesh.faces, 'vertices': mesh.vertices, 'file_name': os.path.splitext(os.path.basename(input_path))[0]})
        print(f"dataset total data samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}
        data_dict['pc_normal'] = self.data[idx]['pc_normal']
        # normalize pc coor
        pc_coor = data_dict['pc_normal'][:, :3]
        normals = data_dict['pc_normal'][:, 3:]
        pc_coor, center, scale = normalize_to_unit_cube(pc_coor, scale_factor=0.9995)

        data_dict['file_name'] = self.data[idx]['file_name']
        pc_coor = pc_coor.astype(np.float32)
        normals = normals.astype(np.float32)

        point_cloud = trimesh.PointCloud(pc_coor)
        point_cloud.metadata['normals'] = normals 
        
        try:
            point_cloud.export(os.path.join(self.output_dir, f"{data_dict['file_name']}.ply"))
        except Exception as e:
            print(f"fail to save point clouds: {e}")

        assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), "normals should be unit vectors, something wrong"
        data_dict['pc_normal'] = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
        
        vertices = self.data[idx]['vertices']
        faces = self.data[idx]['faces']
        bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
        pc_center = (bounds[0] + bounds[1])[None, :] / 2
        pc_scale = ((bounds[1] - bounds[0]).max() + 1e-5)
        data_dict['transform_params'] = torch.tensor([
            center[0], center[1], center[2],
            scale,
            pc_center[0][0], pc_center[0][1], pc_center[0][2], 
            pc_scale
        ], dtype=torch.float32)
        data_dict['vertices'] = vertices
        data_dict['faces']= faces
        return data_dict
    
def get_args():
    parser = argparse.ArgumentParser("SkeletonGPT", add_help=False)

    parser.add_argument("--input_pc_num", default=8192, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument('--input_dir', default=None, type=str, help="input mesh directory")
    parser.add_argument('--input_path', default=None, type=str, help="input mesh path")
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument('--llm', default="facebook/opt-350m", type=str, help="The LLM backend")
    parser.add_argument("--pad_id", default=-1, type=int, help="padding id")
    parser.add_argument("--n_discrete_size", default=128, type=int, help="discretized 3D space")
    parser.add_argument("--n_max_bones", default=100, type=int, help="max number of bones")
    parser.add_argument('--dataset_path', default="combine_256_updated", type=str, help="data path")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)
    parser.add_argument('--pretrained_weights', default=None, type=str)
    parser.add_argument('--save_name', default="infer_results", type=str)
    parser.add_argument("--save_render", default=False, action="store_true", help="save rendering results of mesh with skel")
    parser.add_argument("--apply_marching_cubes", default=False, action="store_true")
    parser.add_argument("--octree_depth", default=7, type=int)
    parser.add_argument("--hier_order", default=False, action="store_true")
    parser.add_argument("--joint_token", default=False, action="store_true", help="use joint_based tokenization")
    parser.add_argument("--seq_shuffle", default=False, action="store_true", help="shuffle the skeleton sequence")

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = get_args()
    
    output_dir = f'{args.output_dir}/{args.save_name}'
    os.makedirs(output_dir, exist_ok=True)
    save_args(args, output_dir)
    
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
        mixed_precision=args.precision,
    )
    
    model = SkeletonGPT(args).cuda()
    
    if args.pretrained_weights is not None:
        pkg = torch.load(args.pretrained_weights, map_location=torch.device("cpu"))
        model.load_state_dict(pkg["model"])
    else:
        raise ValueError("Pretrained weights must be provided.")
    model.eval()
    set_seed(args.seed)
    
    # create dataset
    if args.input_dir is not None:
        input_list = sorted(os.listdir(args.input_dir))
        input_list = [os.path.join(args.input_dir, x) for x in input_list if x.endswith('.ply') or x.endswith('.obj') or x.endswith('.stl')]
        dataset = Dataset(input_list, args.input_pc_num, args.apply_marching_cubes, args.octree_depth, output_dir)
    elif args.input_path is not None:
        dataset = Dataset([args.input_path], args.input_pc_num, args.apply_marching_cubes, args.octree_depth, output_dir)
    else:
        raise ValueError("input_dir or input_path must be provided.")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= 1,
        drop_last = False,
        shuffle = False,
    )

    dataloader, model = accelerator.prepare(dataloader, model)

    for curr_iter, batch_data_label in tqdm(enumerate(dataloader), total=len(dataloader)):
        with accelerator.autocast():
            pred_bone_coords = model.generate(batch_data_label)
        
        # determine the output file name
        file_name = os.path.basename(batch_data_label['file_name'][0])
        pred_skel_filename = os.path.join(output_dir, f'{file_name}_skel.obj')
        pred_rig_filename = os.path.join(output_dir, f"{file_name}_pred.txt")
        mesh_filename = os.path.join(output_dir, f"{file_name}_mesh.obj")
        
        transform_params = batch_data_label['transform_params'][0].cpu().numpy()
        trans = transform_params[:3]
        scale = transform_params[3]
        pc_trans = transform_params[4:7]
        pc_scale = transform_params[7]
        vertices = batch_data_label['vertices'][0].cpu().numpy()
        faces = batch_data_label['faces'][0].cpu().numpy()
        
        skeleton = pred_bone_coords[0].cpu().numpy() 
        pred_joints, pred_bones = pred_joints_and_bones(skeleton.squeeze())
        
        # Post process: merge duplicate or nearby joints and deduplicate bones.
        if args.hier_order: # for MagicArticulate hier order
            pred_root_index = pred_bones[0][0]
            pred_joints, pred_bones, pred_root_index = merge_duplicate_joints_and_fix_bones(pred_joints, pred_bones, root_index=pred_root_index) 
        else: # for Puppeteer or MagicArticulate spaital order
            pred_joints, pred_bones = merge_duplicate_joints_and_fix_bones(pred_joints, pred_bones)
            pred_root_index = None

        # when save rig to txt, denormalize the skeletons to the same scale with input meshes
        pred_joints_denorm = pred_joints * pc_scale + pc_trans # first align with point cloud
        pred_joints_denorm = pred_joints_denorm / scale + trans # then align with original mesh
        
        if args.joint_token:
            pred_root_index = save_skeleton_to_txt_joint(pred_joints_denorm, pred_bones, pred_rig_filename)
        else:
            save_skeleton_to_txt(pred_joints_denorm, pred_bones, pred_root_index, args.hier_order, vertices, pred_rig_filename)
        
        # save skeletons
        if args.hier_order or args.joint_token:
            save_skeleton_obj(pred_joints, pred_bones, pred_skel_filename, pred_root_index, use_cone=True)
        else:
            save_skeleton_obj(pred_joints, pred_bones, pred_skel_filename, use_cone=False)
        
        # when saving mesh and rendering, use normalized vertices (-0.5,0.5)
        vertices_norm = (vertices - trans) * scale
        vertices_norm = (vertices_norm - pc_trans) / pc_scale
        save_mesh(vertices_norm, faces, mesh_filename)
        
        # render mesh w/ skeleton
        if args.save_render:
            if args.hier_order or args.joint_token:
                render_mesh_with_skeleton(pred_joints, pred_bones, vertices_norm, faces, output_dir, file_name, prefix='pred', root_idx=pred_root_index)
            else:
                render_mesh_with_skeleton(pred_joints, pred_bones, vertices_norm, faces, output_dir, file_name, prefix='pred')