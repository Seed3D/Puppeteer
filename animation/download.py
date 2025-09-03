from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="facebook/cotracker3",
    filename="scaled_offline.pth",
    local_dir="third_partys/co_tracker/ckpt"
)

file_path = hf_hub_download(
    repo_id="depth-anything/Video-Depth-Anything-Large",
    filename="video_depth_anything_vitl.pth",
    local_dir="third_partys/Video_Depth_Anything/ckpt"
)
