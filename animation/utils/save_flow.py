

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module processes PNG frame sequences to generate optical flow using PTLFlow,
with support for visualization and video generation.
"""

import argparse
import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm

from third_partys.ptlflow.ptlflow.utils import flow_utils
from third_partys.ptlflow.ptlflow.utils.io_adapter import IOAdapter
import third_partys.ptlflow.ptlflow as ptlflow

class OpticalFlowProcessor:
    """Handles optical flow computation and visualization."""
    
    def __init__(
        self,
        model_name: str = 'dpflow',
        checkpoint: str = 'sintel',
        device: Optional[str] = None,
        resize_to: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize optical flow processor.
        
        Args:
            model_name: Name of the flow model to use
            checkpoint: Checkpoint/dataset name for the model
            device: Device to run on (auto-detect if None)
            resize_to: Optional (width, height) to resize frames
        """
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.resize_to = resize_to
        
        # Initialize model
        self.model = ptlflow.get_model(model_name, ckpt_path=checkpoint).to(self.device).eval()
        print(f"Loaded {model_name} model on {self.device}")
        
        self.io_adapter = None
    
    def load_frame_sequence(self, frames_dir: Union[str, Path]) -> Tuple[List[np.ndarray], List[Path]]:
        """
        Load PNG frame sequence from directory.
        """
        frames_dir = Path(frames_dir)
        
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        # Find PNG files and sort naturally
        png_files = list(frames_dir.glob('*.png'))
        if len(png_files) < 2:
            raise ValueError(f"Need at least 2 PNG frames, found {len(png_files)} in {frames_dir}")
        
        # Natural sorting for proper frame order
        png_files.sort(key=lambda x: self._natural_sort_key(x.name))
        
        frames = []
        for png_path in tqdm(png_files, desc="Loading frames"):
            # Load image in color
            img_bgr = cv.imread(str(png_path), cv.IMREAD_COLOR)
            
            if self.resize_to:
                img_bgr = cv.resize(img_bgr, self.resize_to, cv.INTER_LINEAR)
        
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            frames.append(img_rgb)
        
        return frames, png_files
    
    def _natural_sort_key(self, filename: str) -> List[Union[int, str]]:
        """Natural sorting key for filenames with numbers."""
        import re
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', filename)]
    
    def compute_optical_flow_sequence(
        self,
        frames: List[np.ndarray],
        flow_vis_dir: Union[str, Path],
        flow_save_dir: Optional[Union[str, Path]] = None,
        save_visualizations: bool = True
    ) -> List[torch.Tensor]:
        """
        Compute optical flow for entire frame sequence.
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for optical flow")
        
        flow_vis_dir = Path(flow_vis_dir)
        flow_save_dir = Path(flow_save_dir) if flow_save_dir else flow_vis_dir
        
        H, W = frames[0].shape[:2]
        
        # Initialize IO adapter
        if self.io_adapter is None:
            self.io_adapter = IOAdapter(self.model, (H, W))
        
        flows = []
        for i in tqdm(range(len(frames) - 1), desc="Computing optical flow"):
            # Prepare frame pair
            frame_pair = [frames[i], frames[i + 1]]
            raw_inputs = self.io_adapter.prepare_inputs(frame_pair)
            
            imgs = raw_inputs['images'][0]  # (2, 3, H, W)
            
            pair_tensor = torch.stack((imgs[0:1], imgs[1:2]), dim=1).squeeze(0)  # (1, 2, 3, H, W)
            pair_tensor = pair_tensor.to(self.device, non_blocking=True).contiguous()
        
            with torch.no_grad():
                flow_result = self.model({'images': pair_tensor.unsqueeze(0)})
                flow = flow_result['flows'][0]  # (1, 2, H, W)
            
            flows.append(flow)
            
            if save_visualizations:
                self._save_flow_outputs(flow, i, flow_vis_dir, flow_save_dir)
             
        return flows
    
    def _save_flow_outputs(
        self,
        flow_tensor: torch.Tensor,
        frame_idx: int,
        viz_dir: Path,
        flow_dir: Path
    ) -> None:
        """Save flow outputs in both .flo and visualization formats."""
        # Save raw flow (.flo format)
        flow_hw2 = flow_tensor[0]  # (2, H, W)
        flow_np = flow_hw2.permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        
        flow_path = flow_dir / f'flow_{frame_idx:04d}.flo'
        flow_utils.flow_write(flow_path, flow_np)
        
        # Save visualization
        flow_rgb = flow_utils.flow_to_rgb(flow_tensor)[0]  # Remove batch dimension
        
        if flow_rgb.dim() == 4:  # (Npred, 3, H, W)
            flow_rgb = flow_rgb[0]
        
        flow_rgb_np = (flow_rgb * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        viz_bgr = cv.cvtColor(flow_rgb_np, cv.COLOR_RGB2BGR)
        
        viz_path = viz_dir / f'flow_viz_{frame_idx:04d}.png'
        cv.imwrite(str(viz_path), viz_bgr)

def create_flow_video(
    image_dir: Union[str, Path],
    output_filename: str = 'flow.mp4',
    fps: int = 10,
    pattern: str = 'flow_viz_*.png',
    cleanup_temp: bool = True
) -> bool:
    """
    Create MP4 video from flow visualization images.
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
    
    image_files = sorted(image_dir.glob(pattern))
    if not image_files:
        print(f"No images found matching pattern '{pattern}' in {image_dir}")
    
    temp_dir = image_dir / 'temp_sequence'
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Copy files with sequential naming
        for i, img_file in enumerate(image_files):
            temp_name = temp_dir / f'frame_{i:05d}.png'
            shutil.copy2(img_file, temp_name)
        
        # Create video using ffmpeg
        output_path = image_dir / output_filename
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(temp_dir / 'frame_%05d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        
        subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return True
    except Exception as e:
        print(f"Video creation failed: {e}")
        return False
    finally:
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir)

def main(
    frames_dir: Union[str, Path],
    flow_vis_dir: Union[str, Path] = 'flow_out',
    flow_save_dir: Optional[Union[str, Path]] = None,
    resize_to: Optional[Tuple[int, int]] = None,
    model_name: str = 'dpflow',
    checkpoint: str = 'sintel'
) -> bool:

    # Initialize processor
    processor = OpticalFlowProcessor(
        model_name=model_name,
        checkpoint=checkpoint,
        resize_to=resize_to
    )
    
    # Load frames
    frames, png_paths = processor.load_frame_sequence(frames_dir)
    
    # Compute optical flow
    flows = processor.compute_optical_flow_sequence(
        frames=frames,
        flow_vis_dir=flow_vis_dir,
        flow_save_dir=flow_save_dir,
        save_visualizations=True
    )
    
    # Create video
    create_flow_video(flow_vis_dir)

def get_parser():
    parser = argparse.ArgumentParser(description="Optical flow inference on frame sequences")
    
    parser.add_argument('--input_path', type=str, help="base input path")
    parser.add_argument('--seq_name', type=str, help="sequence name")
    parser.add_argument('--model_name', type=str, default='dpflow', help="Optical flow model to use")
    parser.add_argument('--checkpoint', type=str, default='sintel', help="Model checkpoint/dataset name")
    parser.add_argument('--resize_width', type=int, default=None, help="Resize frame width (must specify both width and height)")
    parser.add_argument('--resize_height', type=int, default=None, help="Resize frame height (must specify both width and height)")
    parser.add_argument('--fps', type=int, default=10, help="Frame rate for output video")

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Path
    frames_dir = f'{args.input_path}/{args.seq_name}/imgs'
    flow_vis_dir = frames_dir.replace("imgs", "flow_vis")
    flow_save_dir = frames_dir.replace("imgs", "flow")

    os.makedirs(flow_vis_dir, exist_ok=True)
    os.makedirs(flow_save_dir, exist_ok=True)
    
    # Prepare resize parameter
    resize_to = None
    if args.resize_width and args.resize_height:
        resize_to = (args.resize_width, args.resize_height)
    
    # Process optical flow
    success = main(
        frames_dir=frames_dir,
        flow_vis_dir=flow_vis_dir,
        flow_save_dir=flow_save_dir,
        resize_to=resize_to,
        model_name=args.model_name,
        checkpoint=args.checkpoint
    )
    
    print("Optical flow processing completed successfully")
