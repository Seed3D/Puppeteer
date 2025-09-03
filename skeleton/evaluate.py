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
import math
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs

from skeleton_models.skeletongen import SkeletonGPT
from utils.skeleton_data_loader import SkeletonData
from utils.save_utils import save_mesh, pred_joints_and_bones, save_skeleton_to_txt, save_skeleton_to_txt_joint, save_args, \
                       merge_duplicate_joints_and_fix_bones, save_skeleton_obj, render_mesh_with_skeleton
from utils.eval_utils import chamfer_dist, joint2bone_chamfer_dist, bone2bone_chamfer_dist
    

def get_args():
    parser = argparse.ArgumentParser("SkeletonGPT", add_help=False)

    parser.add_argument("--input_pc_num", default=8192, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument('--llm', default="facebook/opt-350m", type=str, help="The LLM backend")
    parser.add_argument("--pad_id", default=-1, type=int, help="padding id")
    parser.add_argument("--n_discrete_size", default=128, type=int, help="size of discretized 3D space")
    parser.add_argument("--n_max_bones", default=100, type=int, help="max number of bones")
    parser.add_argument('--dataset_path', default="Articulation_xlv2.npz", type=str, help="data path")
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument('--save_name', default="infer_results", type=str)
    parser.add_argument("--save_render", default=False, action="store_true", help="save rendering results of mesh with skel")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)
    parser.add_argument('--pretrained_weights', default=None, type=str, help="path of pretrained models")
    parser.add_argument("--hier_order", default=False, action="store_true", help="use hier order")
    parser.add_argument("--joint_token", default=False, action="store_true", help="use joint_based tokenization")
    parser.add_argument("--seq_shuffle", default=False, action="store_true", help="shuffle the skeleton sequence")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    
    dataset = SkeletonData.load(args, is_training=False)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last = False,
        shuffle = False,
    )
    
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
    
    set_seed(args.seed)
    dataloader, model = accelerator.prepare(
        dataloader,
        model,
    )
    
    model.eval()
    
    output_dir = f'{args.output_dir}/{args.save_name}'
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    save_args(args, output_dir)
    
    gt_samples, pred_samples = [], []
    avg_j2j_cd, avg_j2b_cd, avg_b2b_cd = 0.0, 0.0, 0.0
    infer_all_time = []
    num_valid = 0
    results_file = f'{output_dir}/evaluate_results.txt'
    
    for curr_iter, batch_data_label in tqdm(enumerate(dataloader), total=len(dataloader)):
        start_time = time.time()
        with accelerator.autocast():
            pred_bone_coords = model.generate(batch_data_label)
        infer_time_pre_mesh = time.time() - start_time
        infer_all_time.append(infer_time_pre_mesh)
            
        if pred_bone_coords is None:
            continue
        print(pred_bone_coords.shape)
        
        if pred_bone_coords.shape[1] > 0:
            gt_joints = batch_data_label['joints'].squeeze(0).cpu().numpy()
            gt_bones = batch_data_label['bones'].squeeze(0).cpu().numpy()
            
            pred_joints, pred_bones = pred_joints_and_bones(pred_bone_coords.cpu().numpy().squeeze(0))
            if pred_bones.shape[0] == 0:
                continue

            # Post process: merge duplicate or nearby joints and deduplicate bones.
            if args.hier_order: # for MagicArticulate hier order
                pred_root_index = pred_bones[0][0]
                pred_joints, pred_bones, pred_root_index = merge_duplicate_joints_and_fix_bones(pred_joints, pred_bones, root_index=pred_root_index) 
            else: # for Puppeteer or MagicArticulate spaital order
                pred_joints, pred_bones = merge_duplicate_joints_and_fix_bones(pred_joints, pred_bones)
                pred_root_index = None

            gt_root_index = int(batch_data_label['root_index'][0])
            gt_joints, gt_bones, gt_root_index = merge_duplicate_joints_and_fix_bones(gt_joints, gt_bones, root_index=gt_root_index) # also merge duplicate joints/bones for GT to prevent NaNs in CD computation.

            if gt_bones.shape[0] == 0 or pred_bones.shape[0] == 0:
                continue

            ### calculate CD
            j2j_cd = chamfer_dist(pred_joints, gt_joints) 
            j2b_cd = joint2bone_chamfer_dist(pred_joints, pred_bones, gt_joints, gt_bones)
            b2b_cd = bone2bone_chamfer_dist(pred_joints, pred_bones, gt_joints, gt_bones)
            
            if math.isnan(j2j_cd) or math.isnan(j2b_cd) or math.isnan(b2b_cd):
                print("NaN cd")
            else:
                num_valid += 1
                avg_j2j_cd += j2j_cd
                avg_j2b_cd += j2b_cd 
                avg_b2b_cd += b2b_cd
                print(f"For {batch_data_label['uuid'][0]}, J2J Chamfer Distance: {j2j_cd:.7f}, J2B Chamfer Distance: {j2b_cd:.7f}, B2B Chamfer Distance: {b2b_cd:.7f}, infer time: {infer_time_pre_mesh:.7f}")
                with open(results_file, 'a') as f:
                    f.write(f"For {batch_data_label['uuid'][0]}, J2J Chamfer Distance: {j2j_cd:.7f}, J2B Chamfer Distance: {j2b_cd:.7f}, B2B Chamfer Distance: {b2b_cd:.7f}, infer time: {infer_time_pre_mesh:.7f}\n")

            if len(gt_samples) <= 30: # only save the first 30 results now, change to 2000 to save all
                pred_samples.append((pred_joints, pred_bones, pred_root_index))
                gt_samples.append((gt_joints, gt_bones, batch_data_label['vertices'][0], batch_data_label['faces'][0], batch_data_label['transform_params'][0], batch_data_label['uuid'][0], gt_root_index))
        
    with open(results_file, 'a') as f:
        f.write(f"Average J2J Chamfer Distance: {avg_j2j_cd/num_valid:.7f}\n")
        f.write(f"Average J2B Chamfer Distance: {avg_j2b_cd/num_valid:.7f}\n")
        f.write(f"Average B2B Chamfer Distance: {avg_b2b_cd/num_valid:.7f}\n")
        f.write(f"Average inference time: {np.mean(infer_all_time):.7f}\n")
    print(f"Valid generation: {num_valid}, Average J2J Chamfer Distance: {avg_j2j_cd/num_valid:.7f}, average J2B Chamfer Distance: {avg_j2b_cd/num_valid:.7f}, average B2B Chamfer Distance: {avg_b2b_cd/num_valid:.7f}, average infer time: {np.mean(infer_all_time):.7f}")

    # save results
    for i, ((pred_joints, pred_bones, pred_root_index), (gt_joints, gt_bones, vertices, faces, transform_params, file_name, gt_root_index)) in enumerate(zip(pred_samples, gt_samples)):
        pred_skel_filename = f'{output_dir}/{file_name}_skel_pred.obj'
        gt_skel_filename = f'{output_dir}/{file_name}_skel_gt.obj'
        mesh_filename = f'{output_dir}/{file_name}.obj'
        pred_rig_filename = f'{output_dir}/{file_name}_pred.txt'
        
        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()
        trans = transform_params[:3].cpu().numpy()
        scale = transform_params[3].cpu().numpy()
        pc_trans = transform_params[4:7].cpu().numpy()
        pc_scale = transform_params[7].cpu().numpy()

        # save skeleton to .txt, denormalize the skeletons to align with input meshes
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
        save_skeleton_obj(gt_joints, gt_bones, gt_skel_filename, gt_root_index, use_cone=True)
        
        # save mesh
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
            render_mesh_with_skeleton(gt_joints, gt_bones, vertices_norm, faces, output_dir, file_name, prefix='gt', root_idx=gt_root_index)
            