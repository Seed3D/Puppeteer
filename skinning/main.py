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

import argparse
import json
import numpy as np
import os
import time
from pathlib import Path
from scipy.spatial import cKDTree

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

torch.set_num_threads(8)

import utils.misc as misc
import skinning_models.models as models
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.skin_data import SkinData
from utils.util import save_skin_weights_to_rig, post_filter

def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--model', default='SkinningNetStacked', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=60, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--pretrained_weights', default=None, type=str, help='dataset path')
    parser.add_argument('--depth', default=1, type=int, help='network depth in transformer')
    parser.add_argument('--max_joints', default=70, type=int, help='max joints')
    parser.add_argument('--use_TAJA', action='store_true', default=True, help='whether to use TAJA')
    parser.add_argument('--save_folder', default="outputs", type=str, help='save folder')
    parser.add_argument('--save_skin_npy', action='store_true', default=False, help='save skinning weights as npy files')

    # for evaluation
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_data_path', default=None, type=str, help='eval dataset path')
    parser.add_argument('--pose_test', action='store_true', default=False, help='evaluate on diverse pose test set')
    parser.add_argument('--modelres_test', action='store_true', default=False, help='evaluate on modelresources test set')
    parser.add_argument('--xl_test', action='store_true', default=False, help='evaluate on articulation-xl test set')
    parser.add_argument('--filter_thre', default=0.15, type=float, help='filter threshold')
    
    # for generation
    parser.add_argument('--generate', action='store_true', default=False, help='Perform inference')
    parser.add_argument('--input_skel_folder', default=None, type=str, help='input skeleton folder')
    parser.add_argument('--mesh_folder', default=None, type=str, help='input mesh folder')
    parser.add_argument('--post_filter', action='store_true', default=False, help='whether to do post filtering')

    return parser

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    model.eval()

    prec_total = []
    rec_total = []
    l1_dist_total = []
    infer_all_time = []

    output_dir = args.save_folder
    os.makedirs(output_dir, exist_ok=True)
    eval_file = os.path.join(output_dir, 
        'evaluate_pose_test.txt' if args.pose_test else
        'evaluate_modelres_test.txt' if args.modelres_test else
        'evaluate_xl_test.txt' if args.xl_test else
        'evaluate_default.txt')
    if args.modelres_test:
        args.filter_thre = 0.35
    
    with open(eval_file, 'w') as f:
        def log_print(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=f)

        for data_iter_step, (sample_points, pc_w_norm, skeleton, valid_joints_mask, dist_graph, vertices, file_name, edges, gt_skin) in enumerate(data_loader):

            sample_points = sample_points.to(device, non_blocking=True)
            pc_w_norm = pc_w_norm.to(device, non_blocking=True)
            skeleton = skeleton.to(device, non_blocking=True)
            valid_joints_mask = valid_joints_mask.to(device, non_blocking=True)
            dist_graph = dist_graph.to(device, non_blocking=True)
            edges = edges.to(device, non_blocking=True)
            
            start_time = time.time()
            with torch.cuda.amp.autocast(enabled=False):
                generate_skin = model(
                sample_points,
                skeleton,
                pc_w_norm,
                dist_graph,
                valid_joints_mask
                )
            infer_time_pre_mesh = time.time() - start_time
            infer_all_time.append(infer_time_pre_mesh)
        
            generate_skin_np = generate_skin.cpu().numpy()  # (batch_size, ...)
            gt_skin_np = gt_skin.cpu().numpy()

            valid_joints_mask_np = valid_joints_mask.cpu().numpy()  # (batch_size, num_joints)

            batch_size = generate_skin_np.shape[0]
            for i in range(batch_size):
                tree = cKDTree(sample_points[i][:,:3].cpu().numpy())
                _, indices = tree.query(vertices[i].cpu().numpy())
                current_generate_skin = generate_skin_np[i][indices]  # (n_vertex, n_joints)
                current_gt_skin = gt_skin_np[i]              # (n_vertex, num_joints)
                current_valid_joints_mask = valid_joints_mask_np[i]  # (num_joints,)

                valid_joint_indices = np.where(current_valid_joints_mask == 1)[0]

                if len(valid_joint_indices) == 0:
                    continue

                generate_skin_masked = current_generate_skin[:, valid_joint_indices] 
                gt_skin_masked = current_gt_skin[:, valid_joint_indices]         

                if generate_skin_masked.size == 0:
                    continue
                
                generate_skin_masked[generate_skin_masked < 1e-3] = 0.0
               
                if args.post_filter:
                    generate_skin_masked = post_filter(generate_skin_masked, edges[i].cpu().numpy(), num_ring=1)
     
                generate_skin_masked[generate_skin_masked < np.max(generate_skin_masked, axis=1, keepdims=True) * args.filter_thre] = 0.0
                generate_skin_masked = generate_skin_masked / (generate_skin_masked.sum(axis=1, keepdims=True)+1e-10)
                    
                valid_rows = np.abs(np.sum(gt_skin_masked, axis=1) - 1) < 1e-2
                generate_skin_masked = generate_skin_masked[valid_rows]    
                gt_skin_masked = gt_skin_masked[valid_rows]

                if args.save_skin_npy:
                    test_folder = ('xl_test' if args.xl_test else 
                                'pose_test' if args.pose_test else
                                'modelres_test' if args.modelres_test else 'default')
                    os.makedirs(os.path.join(output_dir, test_folder), exist_ok=True)
                    npy_path = os.path.join(output_dir, test_folder, f"{file_name[i]}_skin.npy")
                    np.save(npy_path, generate_skin_masked)
                
                # metrics
                precision = np.sum(np.logical_and(generate_skin_masked > 0, gt_skin_masked > 0)) / (np.sum(generate_skin_masked > 0) + 1e-10)
                recall = np.sum(np.logical_and(generate_skin_masked > 0, gt_skin_masked > 0)) / (np.sum(gt_skin_masked > 0) + 1e-10)
                mean_l1_dist = np.sum(np.abs(generate_skin_masked - gt_skin_masked)) /len(generate_skin_masked)
                
                log_print('for', data_iter_step, ',', file_name[i], ': precision:', precision, 'recall:', recall, 'mean_l1_dist:', mean_l1_dist)
              
                prec_total.append(precision)
                rec_total.append(recall)
                l1_dist_total.append(mean_l1_dist)
            
        print("number of items: " + str(len(l1_dist_total)))
        final_precision = np.mean(prec_total) if prec_total else 0.0
        final_recall = np.mean(rec_total) if rec_total else 0.0
        final_avg_l1_dist = np.mean(l1_dist_total) if l1_dist_total else 0.0
        avg_infer_time = np.mean(infer_all_time)

        log_print('final_precision: ', final_precision,
            'final_recall: ', final_recall,
            'final_avg_l1_dist: ', final_avg_l1_dist,
            'avg_infer_time: ', avg_infer_time)

@torch.no_grad()
def generate(data_loader, model, device, args):
    model.eval()
 
    for data_iter_step, (sample_points, pc_w_norm, skeleton, valid_joints_mask, dist_graph, vertices, file_name, edges) in enumerate(data_loader):

        sample_points = sample_points.to(device, non_blocking=True)
        pc_w_norm = pc_w_norm.to(device, non_blocking=True)
        skeleton = skeleton.to(device, non_blocking=True)
        valid_joints_mask = valid_joints_mask.to(device, non_blocking=True)
        dist_graph = dist_graph.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)

        if skeleton[0].shape[0] > args.max_joints:
            continue

        with torch.cuda.amp.autocast(enabled=False):
            generate_skin = model(
            sample_points, 
            skeleton,
            pc_w_norm,
            dist_graph,
            valid_mask=valid_joints_mask,
            )
        
        generate_skin_np = generate_skin.cpu().numpy()  # (batch_size, ...)
        
        valid_joints_mask_np = valid_joints_mask.cpu().numpy()  # (batch_size, num_joints)

        batch_size = generate_skin_np.shape[0]
        for i in range(batch_size):
 
            tree = cKDTree(sample_points[i][:,:3].cpu().numpy())
            _, indices = tree.query(vertices[i].cpu().numpy())
            current_generate_skin = generate_skin_np[i][indices]  # (n_vertex, n_joints)
                
            current_valid_joints_mask = valid_joints_mask_np[i]  # (num_joints,)

            valid_joint_indices = np.where(current_valid_joints_mask == 1)[0]

            if len(valid_joint_indices) == 0:
                continue

            generate_skin_masked = current_generate_skin[:, valid_joint_indices] 
        
            if generate_skin_masked.size == 0:
                continue
            
            if args.post_filter:
                generate_skin_masked = post_filter(generate_skin_masked, edges[i].cpu().numpy(), num_ring=1)
            
            generate_skin_masked[generate_skin_masked < np.max(generate_skin_masked, axis=1, keepdims=True) * 0.35] = 0.0
            generate_skin_masked = generate_skin_masked / (generate_skin_masked.sum(axis=1, keepdims=True))
            
            output_dir = args.save_folder
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            pred_rig_path = os.path.join(args.input_skel_folder, f'{file_name[i]}.txt')
            print(file_name[i])

            # save rig files with skinning weights
            os.makedirs(os.path.join(output_dir, 'generate'), exist_ok=True)
            output_path = os.path.join(output_dir, f'generate/{file_name[i]}_skin.txt')
            print(output_path)
            
            save_skin_weights_to_rig(pred_rig_path, generate_skin_masked, output_path)
            np.save(os.path.join(output_dir, f"generate/{file_name[i]}_skin.npy"), generate_skin_masked)

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    if args.eval:
        dataset_val = SkinData(args, mode='eval', query_num=8192)
    elif args.generate:
        dataset_val = SkinData(args, mode='generate', query_num=8192)
    else:
        dataset_train = SkinData(args, mode='train', query_num=8192)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if not args.eval and not args.generate:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.eval or args.generate:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4 
        )
    
    model = models.__dict__[args.model](args)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)

    if args.pretrained_weights is not None:
        pkg = torch.load(args.pretrained_weights, map_location=torch.device("cpu"))
        model_without_ddp.load_state_dict(pkg["model"])
    
    if args.generate:
        generate(data_loader_val, model_without_ddp, device, args)
    elif args.eval:
        if not any([args.xl_test, args.pose_test, args.modelres_test]):
            raise ValueError("Please specify a test type: --xl_test, --pose_test, or --modelres_test")
        
        test_type = ('Articulation-XL2.0 Test' if args.xl_test else 
                    'Diverse-Pose Test' if args.pose_test else
                    'ModelsResource Test')
        print(f"Running evaluation: {test_type}")
        evaluate(data_loader_val, model_without_ddp, device, args)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)