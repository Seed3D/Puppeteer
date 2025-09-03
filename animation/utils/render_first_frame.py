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
import json
import argparse
from pathlib import Path

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from PIL import Image

from renderer import MeshRenderer3D
from utils.save_utils import render_single_mesh


def render_mesh_all_cameras(mesh_path, cameras_dir, output_dir="renders", image_size=512, device="cuda:0"):
    """
    Render mesh from all camera viewpoints in the cameras directory.
    
    Args:
        mesh_path: Path to OBJ mesh file
        cameras_dir: Directory containing camera JSON config files
        output_dir: Output directory for rendered images
        image_size: Output image size
        device: Device to use
    """
    cameras_dir = Path(cameras_dir)
    output_dir = Path(output_dir)

    # Find all JSON camera config files
    json_files = list(cameras_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON camera files found in {cameras_dir}")
        return
    
    print(f"Found {len(json_files)} camera configurations")
    
    # Render from each camera viewpoint
    for json_file in json_files:
        # Load camera config
        with open(json_file, 'r') as f:
            cam_params = json.load(f)
        
        # Setup renderer for this camera
        renderer = MeshRenderer3D(device=device, image_size=image_size, cam_params=cam_params)
        
        camera_name = json_file.stem
        output_path = output_dir / f"render_{camera_name}.png"
        
        render_single_mesh(renderer, mesh_path, str(output_path))
    
    print(f"All renders saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Render a mesh to an image")
    parser.add_argument('--input_path', type=str, help="base input path")
    parser.add_argument('--seq_name', type=str, help="sequence name")
    parser.add_argument("--cameras_dir", default="utils/cameras", help="Camera config JSON file")
    parser.add_argument("-s", "--size", type=int, default=512, help="Image size")
    parser.add_argument("-d", "--device", default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    mesh_path = f'{args.input_path}/{args.seq_name}/objs/mesh.obj'
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found: {mesh_path}")
    output_dir = f'{args.input_path}/{args.seq_name}/first_frames/'
    os.makedirs(output_dir, exist_ok=True)

    render_mesh_all_cameras(
        mesh_path=mesh_path,
        cameras_dir=args.cameras_dir,
        output_dir=output_dir,
        image_size=args.size,
        device=args.device
    )

if __name__ == "__main__":
    main()