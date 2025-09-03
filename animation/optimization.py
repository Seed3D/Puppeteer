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
import argparse
import json
import numpy as np
import logging
import glob
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from renderer import MeshRenderer3D
from model import RiggingModel
from utils.quat_utils import (
    compute_rest_local_positions, quat_inverse, quat_log, quat_multiply
)
from utils.loss_utils import (
    DepthModule, compute_reprojection_loss, geodesic_loss, root_motion_reg, 
    calculate_flow_loss, compute_depth_loss_normalized, joint_motion_coherence
)
from utils.data_loader import load_model_from_obj_and_rig, prepare_depth
from utils.save_utils import (
    save_args, visualize_joints_on_mesh, save_final_video,
    save_and_smooth_results, visualize_points_on_mesh, save_track_points
)
from utils.misc import warmup_then_decay
from third_partys.co_tracker.save_track import save_track

class AnimationOptimizer:
    """Main class for animation optimization with video guidance."""
    
    def __init__(self, args, device = 'cuda:0'):
        self.args = args
        self.device = device
        self.logger = self._setup_logger()
        
        # Training parameters
        self.reinit_patience_threshold = 20
        self.loss_divergence_factor = 2.0
        self.gradient_clip_norm = 1.0
        
        # Loss weights
        self.target_ratios = {
            'rgb': args.rgb_wt,        
            'flow': args.flow_wt,    
            'proj_joint': args.proj_joint_wt,   
            'proj_vert': args.proj_vert_wt,
            'depth': args.depth_wt, 
            'mask': args.mask_wt
        }
        self.loss_weights = {
            'rgb': 1.0,
            'flow': 1.0,
            'proj_joint': 1.0,
            'proj_vert': 1.0, 
            'depth': 1.0,
            'mask': 1.0
        }

    def _setup_logger(self):
        """Set up logging configuration."""
        logger = logging.getLogger("animation_optimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    def _add_file_handler(self, log_path):
        """Add file handler to logger."""
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _initialize_parameters(self, batch_size, num_joints):
        """Initialize optimization parameters."""
        
        # Fixed first frame quaternions (identity)
        fixed_quat_0 = torch.zeros((1, num_joints, 4), device=self.device)
        fixed_quat_0[..., 0] = 1.0
        
        # Initialize learnable quaternions for frames 1 to B-1
        learn_quats_init = torch.zeros((batch_size - 1, num_joints, 4), device=self.device)
        learn_quats_init[..., 0] = 1.0
        quats_to_optimize = learn_quats_init.clone().detach().requires_grad_(True)
        
        # Initialize global transformations
        fixed_global_quat_0 = torch.zeros((1, 4), device=self.device)
        fixed_global_quat_0[:, 0] = 1.0
        fixed_global_trans_0 = torch.zeros((1, 3), device=self.device)
        
        # Initialize learnable global transformations
        global_quats_init = torch.zeros((batch_size - 1, 4), device=self.device)
        global_quats_init[:, 0] = 1.0
        global_trans_init = torch.zeros((batch_size - 1, 3), device=self.device)
        
        global_quats = global_quats_init.clone().detach().requires_grad_(True)
        global_trans = global_trans_init.clone().detach().requires_grad_(True)
        
        return quats_to_optimize, global_quats, global_trans, fixed_quat_0, fixed_global_quat_0, fixed_global_trans_0
    
    def _setup_optimizer_and_scheduler(self, quats_to_optimize, global_quats, global_trans, n_iters):
        """Set up optimizer and learning rate scheduler."""
        
        base_lr = self.args.warm_lr
        max_lr = self.args.lr
        warmup_steps = 20
        
        min_lr = self.args.min_lr
        quat_lr = base_lr # *2
        
        optimizer = torch.optim.AdamW([
            {'params': quats_to_optimize, 'lr': quat_lr},
            {'params': global_quats, 'lr': quat_lr},
            {'params': global_trans, 'lr': base_lr}
        ])
        
        scheduler = warmup_then_decay(
            optimizer=optimizer,
            total_steps=n_iters,
            warmup_steps=warmup_steps,
            max_lr=max_lr,
            min_lr=min_lr,
            base_lr=base_lr
        )
        
        return optimizer, scheduler
    
    def _compute_smoothness_losses(self, quats_normed, all_global_quats_normed, all_global_trans, model):
        """Compute various smoothness losses."""
        
        # Rotation smoothness loss using geodesic distance
        theta = geodesic_loss(quats_normed[1:], quats_normed[:-1])
        rot_smoothness_loss = (theta ** 2).mean()
        
        # Second-order rotation smoothness (acceleration)
        omega = quat_log(quat_multiply(quat_inverse(quats_normed[:-1]), quats_normed[1:]))
        rot_acc = omega[1:] - omega[:-1]
        rot_acc_smoothness_loss = rot_acc.pow(2).mean()
        
        # Joint motion coherence loss (parent-child relative motion smoothness)
        joint_coherence_loss = joint_motion_coherence(quats_normed, model.parent_indices)
    
        # Root motion regularization
        root_pos_smooth_loss, root_quat_smooth_loss = root_motion_reg(
            all_global_quats_normed, all_global_trans
        )
        
        return rot_smoothness_loss, rot_acc_smoothness_loss, joint_coherence_loss, root_pos_smooth_loss + root_quat_smooth_loss
    
    def pre_calibrate_loss_weights(self, loss_components, target_ratios=None):
        """ calibrate loss weights """  
        loss_for_ratio = {name: loss.detach().clone() for name, loss in loss_components.items()}
        
        rgb_loss = loss_for_ratio['rgb'].item()
        
        for name, loss_val in loss_for_ratio.items():
            if name == 'rgb':
                continue
                
            if loss_val > 1e-8:
                scale_factor = rgb_loss / loss_val.item()
                target_ratio = target_ratios.get(name, 1.0)
                new_weight = self.loss_weights.get(name, 1.0) * scale_factor * target_ratio
                
                self.loss_weights[name] = new_weight

    def _compute_losses(
        self,
        model,
        renderer,
        images_batch,
        tracked_joints_2d,
        joint_vis_mask,
        track_verts_2d,
        vert_vis_mask,
        sampled_vertex_indices,
        track_indices,
        flow_dirs,
        depth_gt_raw,
        mask,
        out_dir,
        iteration
    ):
        """Compute all losses for the optimization."""
        
        batch_size = images_batch.shape[0]
        meshes = [model.get_mesh(t) for t in range(batch_size)]
        pred_images_all = renderer.render_batch(meshes)
        
        # 2D projection losses
        pred_joints_3d = model.joint_positions
        proj_joint_loss = compute_reprojection_loss(
            renderer, joint_vis_mask, pred_joints_3d,
            tracked_joints_2d, self.args.img_size
        )
        
        pred_points_3d = model.deformed_vertices[0]
        proj_vert_loss = compute_reprojection_loss(
            renderer, vert_vis_mask,
            pred_points_3d[:, sampled_vertex_indices],
            track_verts_2d[:, track_indices],
            self.args.img_size
        )
        
        # RGB loss
        pred_rgb = pred_images_all[..., :3]
        real_rgb = images_batch[..., :3]
        diff_rgb_masked = (pred_rgb - real_rgb) * mask.unsqueeze(-1)
        
        mse_rgb_num = (diff_rgb_masked ** 2).sum()
        mse_rgb_den = mask.sum() * 3
        rgb_loss = mse_rgb_num / mse_rgb_den.clamp_min(1e-8)
        
        # Mask loss
        silhouette_soft = renderer.render_silhouette_batch(meshes).squeeze()
        mask_loss = F.binary_cross_entropy(silhouette_soft, mask)
        
        # Depth losses
        fragments = renderer.get_rasterization_fragments(meshes)
        zbuf_depths = fragments.zbuf[..., 0]
        depth_loss = compute_depth_loss_normalized(depth_gt_raw, zbuf_depths, mask)
        
        # Flow losses
        flow_loss = calculate_flow_loss(flow_dirs, self.device, mask, renderer, model)
        
        loss_components = {
            'rgb': rgb_loss,
            'proj_joint': proj_joint_loss,
            'proj_vert': proj_vert_loss,
            'depth': depth_loss,
            'flow': flow_loss,
            'mask': mask_loss
        }
        
        return loss_components

    def optimization(
        self,
        images_batch,
        model,
        renderer,
        tracked_joints_2d,
        joint_vis_mask,
        track_verts_2d,
        vert_vis_mask,
        sampled_vertex_indices,
        track_indices,
        flow_dirs,
        n_iters,
        out_dir):
        """
        Optimize animation parameters with fixed first frame.
        """
        torch.autograd.set_detect_anomaly(True)
        
        batch_size, _, _, _ = images_batch.shape
        num_joints = model.joints_rest.shape[0]
        
        # Setup output directory and logging
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "optimization.log")
        self._add_file_handler(log_path)
        
        # Initialize parameters
        (quats_to_optimize, global_quats, global_trans, 
         fixed_quat_0, fixed_global_quat_0, fixed_global_trans_0) = self._initialize_parameters(batch_size, num_joints)
        
        # Setup rest positions and bind matrices
        rest_local_pos = compute_rest_local_positions(model.joints_rest, model.parent_indices)
        model.initialize_bind_matrices(rest_local_pos)

        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(
            quats_to_optimize, global_quats, global_trans, n_iters
        )
        
        # Initialize depth module and flow weights
        depth_module = DepthModule(
            encoder='vitl',
            device=self.device,
            input_size=images_batch.shape[1],
            fp32=False
        )
        
        # Prepare masks
        real_rgb = images_batch[..., :3]
        threshold = 0.95
        with torch.no_grad():
            background_mask = (real_rgb > threshold).all(dim=-1)
            mask = (~background_mask).float()
        
        depth_gt_raw = prepare_depth(
            flow_dirs.replace('flow', 'depth'), real_rgb, self.device, depth_module
        )
        
        # Optimization tracking
        best_loss = float('inf')
        patience = 0
        best_params = None
        
        pbar = tqdm(total=n_iters, desc="Optimizing animation")

        for iteration in range(n_iters):
            # Combine fixed and learnable parameters
            quats_all = torch.cat([fixed_quat_0, quats_to_optimize], dim=0)
            
            # Normalize quaternions
            reshaped = quats_all.reshape(-1, 4)
            norm = torch.norm(reshaped, dim=1, keepdim=True).clamp_min(1e-8)
            quats_normed = (reshaped / norm).reshape(batch_size, num_joints, 4)
            
            # Global transformations
            all_global_quats = torch.cat([fixed_global_quat_0, global_quats], dim=0)
            all_global_trans = torch.cat([fixed_global_trans_0, global_trans], dim=0)
            all_global_quats_normed = all_global_quats / torch.norm(
                all_global_quats, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            
            # Compute smoothness losses
            (rot_smoothness_loss, rot_acc_smoothness_loss, joint_coherence_loss,
            root_smooth_loss) = self._compute_smoothness_losses(
                quats_normed, all_global_quats_normed, all_global_trans, model
            )
            
            # animate model
            model.animate(quats_normed, all_global_quats_normed, all_global_trans)
            
            # Verify first frame hasn't changed
            verts0 = model.vertices[0]
            de0 = model.deformed_vertices[0][0]
            assert torch.allclose(de0, verts0, atol=1e-2), "First frame vertices have changed!"
            
            # Compute all losses
            loss_components = self._compute_losses(
                model, renderer, images_batch, tracked_joints_2d, joint_vis_mask,
                track_verts_2d, vert_vis_mask, sampled_vertex_indices, track_indices,
                flow_dirs, depth_gt_raw, mask, out_dir, iteration
            )
            
            total_smoothness_loss = rot_smoothness_loss + rot_acc_smoothness_loss * 10 
            
            if iteration == 0:
                self.pre_calibrate_loss_weights(loss_components, self.target_ratios)

            total_loss = (
                loss_components['rgb'] +
                self.loss_weights['mask'] * loss_components['mask'] +
                self.loss_weights['flow'] * loss_components['flow'] +
                self.loss_weights['proj_joint'] * loss_components['proj_joint'] +
                self.loss_weights['proj_vert'] * loss_components['proj_vert'] +
                self.loss_weights['depth'] * loss_components['depth'] +
                self.args.smooth_weight * total_smoothness_loss +
                self.args.coherence_weight * joint_coherence_loss +
                self.args.root_smooth_weight * root_smooth_loss
            )
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [quats_to_optimize, global_quats, global_trans],
                max_norm=self.gradient_clip_norm
            )
            optimizer.step()
            scheduler.step()
            
            # Update progress bar and logging
            loss_desc = (
                f"Loss: {total_loss.item():.4f}, "
                f"RGB: {loss_components['rgb'].item():.4f}, "
                f"Mask: {self.loss_weights['mask'] * loss_components['mask'].item():.4f}, "
                f"Flow: {self.loss_weights['flow'] * loss_components['flow'].item():.4f}, "
                f"Proj_joint: {self.loss_weights['proj_joint'] * loss_components['proj_joint'].item():.4f}, "
                f"Proj_vert: {self.loss_weights['proj_vert'] * loss_components['proj_vert'].item():.4f}, "
                f"Depth: {self.loss_weights['depth'] * loss_components['depth'].item():.4f}, "
                f"Smooth: {self.args.smooth_weight * total_smoothness_loss.item():.4f}, "
                f"Joint smooth: {self.args.coherence_weight * joint_coherence_loss.item():.4f}, "
                f"Root smooth: {self.args.root_smooth_weight * root_smooth_loss.item():.4f}"
            )
            pbar.set_description(loss_desc)
            
            if iteration % 5 == 0:
                self.logger.info(f"Iter {iteration}: {loss_desc}")
            
            # Adaptive reinitialization
            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = {
                    'quats': quats_to_optimize.clone().detach(),
                    'global_quats': global_quats.clone().detach(),
                    'global_trans': global_trans.clone().detach()
                }
                patience = 0
            elif (current_loss > best_loss * self.loss_divergence_factor or 
                  patience > self.reinit_patience_threshold * 2):
                # Reinitialize with best parameters
                quats_to_optimize = best_params['quats'].clone().requires_grad_(True)
                global_quats = best_params['global_quats'].clone().requires_grad_(True)
                global_trans = best_params['global_trans'].clone().requires_grad_(True)
                
                optimizer, scheduler = self._setup_optimizer_and_scheduler(
                    quats_to_optimize, global_quats, global_trans, n_iters
                )
                patience = 0
                self.logger.info(f'Adaptive reset at iteration {iteration} with best loss: {best_loss:.6f}')
            else:
                patience += 1
            
            pbar.update(1)
        
        pbar.close()
        
        # Prepare final results
        quats_final = torch.cat([fixed_quat_0, best_params['quats']], dim=0)
        
        # Final normalization
        reshaped = quats_final.reshape(-1, 4)
        norm = torch.norm(reshaped, dim=1, keepdim=True).clamp_min(1e-8)
        quats_final = (reshaped / norm).reshape(batch_size, num_joints, 4)
        
        global_quats_final = torch.cat([fixed_global_quat_0, best_params['global_quats']], dim=0)
        global_trans_final = torch.cat([fixed_global_trans_0, best_params['global_trans']], dim=0)
        global_quats_final = global_quats_final / torch.norm(
            global_quats_final, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        
        return quats_final, global_quats_final, global_trans_final

def load_and_prepare_data(args):
    """Load and prepare all necessary data for optimization."""
    
    # Define paths
    base_path = f'{args.input_path}/{args.seq_name}'
    mesh_path = f'{base_path}/objs/mesh.obj'
    rig_path = f'{base_path}/objs/rig.txt'
    img_path = f'{base_path}/imgs'
    flow_dirs = f'{base_path}/flow'
    
    # Load model
    model = load_model_from_obj_and_rig(mesh_path, rig_path, device=args.device)
    
    # Load images
    img_files = sorted(glob.glob(os.path.join(img_path, "*.png")))
    images = []
    for f in img_files:
        img = Image.open(f).convert("RGBA")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).to(args.device)
        images.append(t)
    
    images_batch = torch.stack(images, dim=0)
    
    return model, images_batch, flow_dirs, img_path

def setup_renderers(args):
    """Setup multiple renderers for different camera views."""
    
    available_views = [
        "front", "back", "left", "right",
        "front_left", "front_right", "back_left", "back_right"
    ]
    
    if args.main_renderer not in available_views:
        raise ValueError(f"Main renderer '{args.main_renderer}' not found in available cameras: {available_views}")
    
    main_cam_config = json.load(open(f"utils/cameras/{args.main_renderer}.json"))
    main_renderer = MeshRenderer3D(args.device, image_size=args.img_size, cam_params=main_cam_config)
    
    additional_views = [view.strip() for view in args.additional_renderers.split(',') if view.strip()]
    if len(additional_views) > 3:
        print(f"Warning: Only first 3 additional renderers will be used. Got: {additional_views}")
        additional_views = additional_views[:3]
    
    additional_renderers = {}
    for view_name in additional_views:
        if view_name in available_views and view_name != args.main_renderer:
            cam_config = json.load(open(f"utils/cameras/{view_name}.json"))
            renderer = MeshRenderer3D(args.device, image_size=args.img_size, cam_params=cam_config)
            additional_renderers[f"{view_name}_renderer"] = renderer
        elif view_name == args.main_renderer:
            print(f"Warning: '{view_name}' is already the main renderer, skipping...")
        elif view_name not in available_views:
            print(f"Warning: Camera view '{view_name}' not found, skipping...")
    
    return main_renderer, additional_renderers

def get_parser():
    """Create argument parser with all configuration options."""
    
    parser = argparse.ArgumentParser(description="3D Rigging Optimization")
    
    # Training parameters
    training_group = parser.add_argument_group('Training')
    training_group.add_argument("--iter", type=int, default=500, help="Number of training iterations")
    training_group.add_argument("--img_size", type=int, default=512, help="Image resolution")
    training_group.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    training_group.add_argument("--img_fps", type=int, default=15, help="Image frame rate")
    training_group.add_argument('--main_renderer', type=str, default='front', help='Main renderer camera view (default: front)')
    training_group.add_argument('--additional_renderers', type=str, default="back, right, left", help='Additional renderer views (max 3), comma-separated (e.g., "back,left,right"). ')
    
    # Learning rates
    lr_group = parser.add_argument_group('Learning Rates')
    lr_group.add_argument("--lr", type=float, default=2e-3, help="Base learning rate")
    lr_group.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    lr_group.add_argument("--warm_lr", type=float, default=1e-5, help="Warmup learning rate")
    
    # Loss weights
    loss_group = parser.add_argument_group('Loss Weights')
    loss_group.add_argument("--smooth_weight", type=float, default=0.2)
    loss_group.add_argument("--root_smooth_weight", type=float, default=1.0)
    loss_group.add_argument("--coherence_weight", type=float, default=10)
    loss_group.add_argument("--rgb_wt", type=float, default=1.0, help="RGB loss target ratio (relative importance)")
    loss_group.add_argument("--mask_wt", type=float, default=1.0, help="Mask loss target ratio") 
    loss_group.add_argument("--proj_joint_wt", type=float, default=1.5, help="Joint projection loss target ratio")
    loss_group.add_argument("--proj_vert_wt", type=float, default=3.0, help="Point projection loss target ratio")
    loss_group.add_argument("--depth_wt", type=float, default=0.8, help="Depth loss target ratio")
    loss_group.add_argument("--flow_wt", type=float, default=0.8, help="Flow loss target ratio")

    # Data and output
    data_group = parser.add_argument_group('Data and Output')
    data_group.add_argument("--input_path", type=str, default="inputs")
    data_group.add_argument("--save_path", type=str, default="results")
    data_group.add_argument("--save_name", type=str, default="results")
    data_group.add_argument("--seq_name", type=str, default=None)
    
    # Flags
    flag_group = parser.add_argument_group('Flags')
    flag_group.add_argument('--gauss_filter', action='store_true', default=False)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Setup output directory
    out_dir = f'{args.save_path}/{args.seq_name}/{args.save_name}'
    save_args(args, out_dir)
    
    # Initialize optimizer
    ani_optimizer = AnimationOptimizer(args, device=args.device)
    
    # Setup renderers
    renderer, additional_renderers = setup_renderers(args)

    # Load and prepare data
    model, images_batch, flow_dirs, img_path = load_and_prepare_data(args)
    
    # Setup tracking
    joint_vis_mask = visualize_joints_on_mesh(model, renderer, args.seq_name, out_dir=out_dir)
    joint_vis_mask = torch.from_numpy(joint_vis_mask).float().to(args.device)
    
    joint_project_2d = renderer.project_points(model.joints_rest)
    
    # Setup track paths
    track_2d_path = img_path.replace('imgs', 'track_2d_joints')
    os.makedirs(track_2d_path, exist_ok=True)
    
    # Load or generate tracks
    if not os.listdir(track_2d_path):
        print("Generating joint tracks")
        tracked_joints_2d = save_track(args.seq_name, joint_project_2d, img_path, track_2d_path, out_dir)
    else:
        print("Loading existing joint tracks")
        tracked_joints_2d = np.load(f'{track_2d_path}/pred_tracks.npy')
    
    # Setup point tracking
    vert_vis_mask = visualize_points_on_mesh(model, renderer, args.seq_name, out_dir=out_dir)
    vert_vis_mask = torch.from_numpy(vert_vis_mask).float().to(args.device)
    
    track_verts_2d, track_indices, sampled_vertex_indices = save_track_points(
        vert_vis_mask, renderer, model, img_path, out_dir, args
    )
    vert_vis_mask = vert_vis_mask[sampled_vertex_indices]
    
    # Run optimization
    print(f"Starting optimization")
    final_quats, root_quats, root_pos = ani_optimizer.optimization(
        images_batch=images_batch,
        model=model,
        renderer=renderer,
        tracked_joints_2d=tracked_joints_2d,
        joint_vis_mask=joint_vis_mask,
        track_verts_2d=track_verts_2d,
        vert_vis_mask=vert_vis_mask,
        sampled_vertex_indices=sampled_vertex_indices,
        track_indices=track_indices,
        flow_dirs=flow_dirs,
        n_iters=args.iter,
        out_dir=out_dir
    )
    
    # Save results
    save_and_smooth_results(
        args, model, renderer, final_quats, root_quats, root_pos,
        out_dir, additional_renderers, fps=10
    )
    
    print("Optimization completed successfully")
    save_final_video(args)
    

if __name__ == "__main__":
    main()