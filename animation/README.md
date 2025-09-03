# 3D Animation with Video Guidance
This repository provides a complete pipeline for generating 3D object animations with video guidance. The system includes data processing and optimization algorithms for rigging-based animation.

## Overview
The pipeline takes a rigged 3D model and a reference video, then optimizes the object's motion to match the video guidance while maintaining realistic skeletal constraints.

## Prerequisites

### Model Downloads
Download the required pre-trained models:

- [Video-Depth-Anything](https://huggingface.co/depth-anything/Video-Depth-Anything-Large) - For depth estimation
- [CoTracker3](https://huggingface.co/facebook/cotracker3) - For point tracking

```
python download.py
```

### Input Data Structure

Organize your input data as follows:
```
inputs/
└── {seq_name}/
    ├── objs/
    │   ├── mesh.obj          # 3D mesh geometry
    │   ├── rig.txt           # Rigging definition
    │   ├── material.mtl      # Material properties (optional)
    │   └── texture.png       # Texture maps (optional)
    ├── first_frames/         # Rendered initial frames
    ├── imgs/                 # Extracted video frames
    ├── flow/                 # Optical flow data
    ├── flow_vis/             # Visualized optical flow
    ├── depth/                # Esitmated depth data
    ├── track/                # tracked joints/vertices
    └── input.mp4             # Source video
```

## Data Processing

Given a 3D model with rigging under `inputs/{seq_name}/objs` (`mesh.obj, rig.txt`, optional `.mtl` and texture `.png`), we first render the object from a specified viewpoint. This image is used as the input (first frame) to the video generation model (e.g., [Jimeng AI](https://jimeng.jianying.com/ai-tool/home?type=video)).

```
python utils/render_first_frame.py --input_path inputs --seq_name {seq_name}
```
Replace `{seq_name}` with your sequence name. The first-frame images are saved to `inputs/{seq_name}/first_frames`. This generates reference images from 4 different viewpoints (you can add more). Choose the viewpoint that best shows the object's joints and key parts for optimal animation results. Save the generated videos to `inputs/{seq_name}/input.mp4`.

Then we extract the frames from the video by running:

```
cd inputs/{seq_name}; mkdir imgs
ffmpeg -i input.mp4 -vf fps=10 imgs/frame_%04d.png
cd ../../
```

Estimate optical flows by running:

```
python utils/save_flow.py --input_path inputs --seq_name {seq_name}
```
The flow `.flo` files are saved to `inputs/{seq_name}/flow`, the flow visualization are saved to `inputs/{seq_name}/flow_vis`. Depth and tracking information are saved during optimization.

## Optimization

To optimize the animation, you can run

```
bash demo.sh
```

The results are saved to `results/{seq_name}/{save_name}`


## TODO

- [ ] Add multi-view supervisions.