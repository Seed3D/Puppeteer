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

from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesAtlas
from pytorch3d.structures import Meshes

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import subprocess
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from third_partys.co_tracker.save_track import save_track

def render_single_mesh(renderer, mesh_path, out_path="render_result.png", atlas_size=8):
    """
    Test render a single mesh and save the result.
    """
    device = renderer.device 

    verts, faces, aux = load_obj(
        mesh_path,
        device=device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=atlas_size,
        texture_wrap="repeat"
    )
    atlas = aux.texture_atlas            # (F, atlas_size, atlas_size, 3)

    vmin, vmax = verts.min(0).values, verts.max(0).values
    center = (vmax + vmin) / 2.
    scale  = (vmax - vmin).max()
    verts  = (verts - center) / scale

    mesh_norm = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesAtlas(atlas=[atlas])
    )
    with torch.no_grad():
        rendered = renderer.render(mesh_norm)  # shape=[1, H, W, 4]
    
    rendered_img = renderer.tensor_to_image(rendered)
    
    pil_img = Image.fromarray(rendered_img)
    pil_img.save(out_path)
    print(f"Saved render to {out_path}")

def apply_gaussian_smoothing(data, sigma = 1.0, preserve_first_frame = True, eps = 1e-8):
    """
    Apply Gaussian smoothing along the time axis with quaternion normalization.
    """
    smoothed = gaussian_filter1d(data, sigma=sigma, axis=0)
    
    # Preserve first frame if requested
    if preserve_first_frame and data.shape[0] > 0:
        smoothed[0] = data[0]

    if data.shape[-1] == 4:
        norms = np.linalg.norm(smoothed, axis=-1, keepdims=True)
        smoothed = smoothed / np.maximum(norms, eps)
    
    return smoothed

def render_single_view_sequence(quats, root_quats, root_pos, renderer, model, output_dir, view_name, fps = 25):
    """
    Render animation sequence from a single viewpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    T = quats.shape[0]

    model.animate(quats, root_quats, root_pos)

    for i in tqdm(range(T), desc=f"Rendering {view_name}"):
        mesh = model.get_mesh(i)
        rendered = renderer.render(mesh)

        img_array = renderer.tensor_to_image(rendered)
        img = Image.fromarray(img_array)
        
        frame_path = output_dir / f"{view_name}_frame_{i:04d}.png"
        img.save(frame_path)

    # Create video
    video_path = output_dir / f"{view_name}_output_video.mp4"
    cmd = f"ffmpeg -y -framerate {fps} -i {output_dir}/{view_name}_frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
    subprocess.call(cmd, shell=True)

def save_and_smooth_results(args, model, renderer, final_quats, root_quats, root_pos, out_dir, additional_renderers = None, load_pt = False, sigma = 1.0, fps = 25):
    """
    Save and smooth animation results with multi-view rendering.
    """
    device = final_quats.device
    T = final_quats.shape[0]
    # Save Raw Results
    if not load_pt:
        raw_dir = os.path.join(out_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        torch.save(final_quats, os.path.join(raw_dir, "local_quats.pt"))
        torch.save(root_quats, os.path.join(raw_dir, "root_quats.pt"))
        torch.save(root_pos, os.path.join(raw_dir, "root_pos.pt"))
        if hasattr(model, 'rest_local_positions'):
            torch.save(model.rest_local_positions, os.path.join(raw_dir, "rest_local_positions.pt"))
        
        print(f"Saved raw motion to {raw_dir}")
        
    quats_np = final_quats.cpu().numpy()
    root_quats_np = root_quats.cpu().numpy()
    root_pos_np = root_pos.cpu().numpy()

    # Apply Gaussian smoothing if enabled
    if args.gauss_filter:
        print(f"Applying Gaussian smoothing (sigma={sigma})")
        
        smooth_quats_np = apply_gaussian_smoothing(
            quats_np, sigma=sigma, preserve_first_frame=True
        )
        smooth_root_quats_np = apply_gaussian_smoothing(
            root_quats_np, sigma=sigma, preserve_first_frame=True
        )
        smooth_root_pos_np = apply_gaussian_smoothing(
            root_pos_np, sigma=sigma, preserve_first_frame=True
        )
        smooth_dir = os.path.join(out_dir, "smoothed")
        os.makedirs(smooth_dir, exist_ok=True)
        save_dir = smooth_dir
      
    else:
        smooth_quats_np = quats_np
        smooth_root_quats_np = root_quats_np
        smooth_root_pos_np = root_pos_np
        save_dir = raw_dir
    
    smooth_quats = torch.tensor(smooth_quats_np, dtype=torch.float32, device=device)
    smooth_root_quats = torch.tensor(smooth_root_quats_np, dtype=torch.float32, device=device)
    smooth_root_pos = torch.tensor(smooth_root_pos_np, dtype=torch.float32, device=device)
    
    # Render Sequences
    if not load_pt and args.gauss_filter:
        smooth_dir_path = Path(smooth_dir)
        torch.save(smooth_quats, smooth_dir_path / "local_quats.pt")
        torch.save(smooth_root_quats, smooth_dir_path / "root_quats.pt")
        torch.save(smooth_root_pos, smooth_dir_path / "root_pos.pt")
        print(f"Saved smoothed motion to {smooth_dir}")
    
    # Render main view
    print(f"Rendering {args.main_renderer} view ({T} frames)")
    render_single_view_sequence(
        smooth_quats, smooth_root_quats, smooth_root_pos,
        renderer, model, save_dir, args.main_renderer, fps
    )
    
    # Render additional views if provided
    if additional_renderers:
        for renderer_key, view_renderer in additional_renderers.items():
            view_name = renderer_key.replace("_renderer", "")
            render_single_view_sequence(
                smooth_quats, smooth_root_quats, smooth_root_pos,
                view_renderer, model, save_dir, view_name, fps
            )

def save_args(args, output_dir, filename="config.json"):
    args_dict = vars(args)
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

def visualize_joints_on_mesh(model, renderer, seq_name, out_dir):
    """
    Render mesh with joint visualizations and return visibility mask.
    """
    joints_2d = renderer.project_points(model.joints_rest)
   
    mesh = model.get_mesh()
    image_with_joints, vis_mask = renderer.render_with_points(mesh, model.joints_rest)
    image_np = image_with_joints[0].cpu().numpy()
    if image_np.shape[2] == 4:
        image_rgb = image_np[..., :3]
    else:
        image_rgb = image_np
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    img = Image.fromarray(image_rgb)
    output_path = f"{out_dir}/mesh_with_joints_{seq_name}_visible.png"
    img.save(output_path)
    return vis_mask

def visualize_points_on_mesh(model, renderer, seq_name, out_dir):
    """
    Render mesh with point visualizations and return visibility mask.
    """
    points_2d = renderer.project_points(model.vertices[0])

    mesh = model.get_mesh()
    image_with_points, vis_mask = renderer.render_with_points(mesh, model.vertices[0], for_vertices=True)
    image_np = image_with_points[0].cpu().numpy()
    if image_np.shape[2] == 4:
        image_rgb = image_np[..., :3]
    else:
        image_rgb = image_np
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    img = Image.fromarray(image_rgb)
    output_path = f"{out_dir}/mesh_with_verts_{seq_name}_visible.png"
    img.save(output_path)
    return vis_mask

def save_track_points(point_vis_mask, renderer, model, img_path, out_dir, args):
    """
    Save and track selected points on the mesh with intelligent sampling.
    """

    vertex_project_2d = renderer.project_points(model.vertices[0])
    visible_indices = torch.where(point_vis_mask)[0]
    
    track_2d_point_path = img_path.replace('imgs', 'track_2d_verts')
    os.makedirs(track_2d_point_path, exist_ok=True)
    
    num_visible = len(visible_indices)
    MAX_VISIBLE_POINTS = 15000
    MAX_SAMPLE_POINTS = 4000
    
    # Determine tracking strategy
    tracking_mode = "full" if num_visible <= MAX_VISIBLE_POINTS else "sampled"
    
    if not os.listdir(track_2d_point_path):
        # Generate new tracking data
        if tracking_mode == "full":
            print(f"Saving tracks for all visible vertices (count: {num_visible})")
            
            # Track all visible points
            visible_vertex_project_2d = vertex_project_2d[visible_indices]
            track_2d_point = save_track(
                args.seq_name, visible_vertex_project_2d, img_path, 
                track_2d_point_path, out_dir, for_point=True
            )
            
            np.save(f'{track_2d_point_path}/visible_indices.npy', 
                    visible_indices.cpu().numpy())
            
            # Sample subset for final use
            num_sample = min(MAX_SAMPLE_POINTS, num_visible)
            sampled_local_indices = torch.randperm(num_visible)[:num_sample]
            sampled_vertex_indices = visible_indices[sampled_local_indices]
            np.save(f'{track_2d_point_path}/sampled_indices.npy', 
                    sampled_vertex_indices.cpu().numpy())
            
        else:
            print(f"Too many visible vertices ({num_visible} > {MAX_VISIBLE_POINTS}), "
                    f"tracking only {MAX_SAMPLE_POINTS} sampled vertices")
            
            # Sample points directly from visible set
            num_sample = min(MAX_SAMPLE_POINTS, num_visible)
            sampled_local_indices = torch.randperm(num_visible)[:num_sample]
            sampled_vertex_indices = visible_indices[sampled_local_indices]
            
            # Track only sampled points
            sampled_vertex_project_2d = vertex_project_2d[sampled_vertex_indices]
            track_2d_point = save_track(
                args.seq_name, sampled_vertex_project_2d, img_path, 
                track_2d_point_path, out_dir, for_point=True
            )
            
            np.save(f'{track_2d_point_path}/visible_indices.npy', 
                    visible_indices.cpu().numpy())
            np.save(f'{track_2d_point_path}/sampled_indices.npy', 
                    sampled_vertex_indices.cpu().numpy())
    
    else:
        # Load existing tracking data
        print("Loading existing vertex tracks")
        track_2d_point = np.load(f'{track_2d_point_path}/pred_tracks.npy')
        
        visible_indices = np.load(f'{track_2d_point_path}/visible_indices.npy')
        visible_indices = torch.from_numpy(visible_indices).long().to(args.device)
        
        sampled_vertex_indices = np.load(f'{track_2d_point_path}/sampled_indices.npy')
        sampled_vertex_indices = torch.from_numpy(sampled_vertex_indices).long().to(args.device)
    
    track_2d_point = torch.from_numpy(track_2d_point).float().to(args.device)
    
    # Create index mapping for tracking data
    if tracking_mode == "full":
        # Map from original vertex indices to positions in tracking data
        vertex_to_track_idx = {idx.item(): i for i, idx in enumerate(visible_indices)}
        
        track_indices = torch.tensor(
            [vertex_to_track_idx[idx.item()] for idx in sampled_vertex_indices], 
            device=args.device, dtype=torch.long
        )
    else:
        # Direct mapping for sampled-only tracking
        track_indices = torch.arange(len(sampled_vertex_indices), 
                                    device=args.device, dtype=torch.long)
    
    return track_2d_point, track_indices, sampled_vertex_indices

def save_final_video(args):

    additional_views = [view.strip() for view in args.additional_renderers.split(',') if view.strip()]
    if len(additional_views) > 3:
        additional_views = additional_views[:3]
    additional_views = [view for view in additional_views if view != args.main_renderer]
    
    save_dir = 'raw' if not args.gauss_filter else 'smoothed'
    import subprocess
    cmd = (
        f'ffmpeg '
        f'-i {args.input_path}/{args.seq_name}/input.mp4 '
        f'-i {args.save_path}/{args.seq_name}/{args.save_name}/{save_dir}/{args.main_renderer}_output_video.mp4 '
        '-filter_complex "'
        '[0:v][1:v]hstack=inputs=2[stacked]; '
        '[stacked]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text=\'gt\':x=(w/4-text_w/2):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black:boxborderw=10, '
        f'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text=\'ours\':x=(3*w/4-text_w/2):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black:boxborderw=10" '
        f'-c:a copy {args.save_path}/{args.seq_name}/{args.save_name}/concat_output.mp4'
    )

    subprocess.call(cmd, shell=True)
    cmd = (
        f'ffmpeg '
        f'-i {args.input_path}/{args.seq_name}/input.mp4 '
        f'-i {args.save_path}/{args.seq_name}/{args.save_name}/{save_dir}/{args.main_renderer}_output_video.mp4 '
        f'-i {args.save_path}/{args.seq_name}/{args.save_name}/{save_dir}/{additional_views[0]}_output_video.mp4 '
        f'-i {args.save_path}/{args.seq_name}/{args.save_name}/{save_dir}/{additional_views[1]}_output_video.mp4 '
        f'-i {args.save_path}/{args.seq_name}/{args.save_name}/{save_dir}/{additional_views[2]}_output_video.mp4 '
        '-filter_complex "'
        '[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[stacked]; '
        '[stacked]drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text=\'gt\':x=(w/10-text_w/2):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black:boxborderw=10, '
        f'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text=\'{args.main_renderer}\':x=(3*w/10-text_w/2):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black:boxborderw=10, '
        f'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text=\'{additional_views[0]}\':x=(5*w/10-text_w/2):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black:boxborderw=10, '
        f'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text=\'{additional_views[1]}\':x=(7*w/10-text_w/2):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black:boxborderw=10, '
        f'drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text=\'{additional_views[2]}\':x=(9*w/10-text_w/2):y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black:boxborderw=10" '
        f'-c:a copy {args.save_path}/{args.seq_name}/{args.save_name}/concat_output_4view.mp4'
    )
    subprocess.call(cmd, shell=True)

def load_motion_data(motion_dir, device="cuda:0"):
    """
    Load saved motion data.
    """
    local_quats = torch.load(os.path.join(motion_dir, "local_quats.pt"), map_location=device)
    root_quats = torch.load(os.path.join(motion_dir, "root_quats.pt"), map_location=device)
    root_pos = torch.load(os.path.join(motion_dir, "root_pos.pt"), map_location=device)
    
    # Load rest positions if available (for reference)
    rest_pos_path = os.path.join(motion_dir, "rest_local_positions.pt")
    if os.path.exists(rest_pos_path):
        rest_positions = torch.load(rest_pos_path, map_location=device)
    else:
        rest_positions = None
        print("Warning: rest_local_positions.pt not found, model should have them initialized")
    
    return local_quats, root_quats, root_pos, rest_positions