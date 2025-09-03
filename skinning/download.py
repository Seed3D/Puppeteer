from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="mikaelaangel/partfield-ckpt",
    filename="model_objaverse.ckpt",
    local_dir="third_partys/PartField/ckpt"
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth",
    local_dir="skinning"
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skinning_ckpts/puppeteer_skin_w_diverse_pose_depth2.pth",
    local_dir="skinning"
)

file_path = hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skinning_ckpts/puppeteer_skin_wo_diverse_pose_depth1.pth",
    local_dir="skinning"
)