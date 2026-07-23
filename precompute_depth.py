#!/usr/bin/env python3
"""
precompute_depth.py — Run Video-Depth-Anything standalone so the heavy model
is freed before Puppeteer's optimization loop starts (fixes OOM on <=12GB
GPUs). Writes <seq>/depth/depth_gt_raw.pt in exactly the format
utils/data_loader.prepare_depth caches.

Usage (from animation/ dir, venv active):
  PYTHONPATH=. python ../precompute_depth.py --input_path ../examples --seq_name deer --img_size 512
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from utils.loss_utils import DepthModule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--seq_name", required=True)
    p.add_argument("--img_size", type=int, default=512)
    args = p.parse_args()

    base = os.path.join(args.input_path, args.seq_name)
    img_paths = sorted(glob.glob(os.path.join(base, "imgs", "frame_*.png")))
    assert img_paths, f"no frames under {base}/imgs"

    frames = []
    for pth in img_paths:
        img = Image.open(pth).convert("RGB").resize((args.img_size, args.img_size))
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
    batch = torch.from_numpy(np.stack(frames))  # [T,H,W,3] in 0..1

    depth = DepthModule(encoder="vitl", device="cuda")
    with torch.no_grad():
        depth_gt_raw = depth.get_depth_maps(batch)
    os.makedirs(os.path.join(base, "depth"), exist_ok=True)
    out = os.path.join(base, "depth", "depth_gt_raw.pt")
    torch.save(depth_gt_raw.cpu(), out)
    print("wrote", out, tuple(depth_gt_raw.shape))


if __name__ == "__main__":
    main()
