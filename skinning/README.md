# Skinning Weight Prediction
This folder provides the skinning weight prediction implementation and scripts to evaluate the paper’s metrics on three test sets. You can also run inference on your own 3D objects.

## Weights Download
First download [checkpoints of PartField](https://huggingface.co/mikaelaangel/partfield-ckpt) and our [released weights](https://huggingface.co/Seed3D/Puppeteer) for skinning weight prediction:

```
ln -s ../../skeleton/third_party/Michelangelo third_partys/Michelangelo
python download.py
```

## Evaluation

To reproduce our evaluations, run the following command on `Articulation-XL2.0-test`, `ModelResource-test` and `Diverse-pose-test`. The test sets are available [here](https://drive.google.com/drive/folders/1zIAcg1sAJtVemMKybZEMPnUzKXDST_dX?usp=sharing), we preprocess the released NPZ files and save them as h5 files (check `utils/save_h5.py` for how we save them). The inference process requires 4.2 GB of VRAM.

```
bash eval.sh
```

We save the skinning weights as `.npy` files by passing `--save_skin_npy`. 

## Demo

Given meshes and skeletons, we can predict skinning weights by running:

```
bash demo.sh
```

For inputs, place meshes `.obj` files in the directory specified by `--mesh_folder`, and place rig `.txt` files in `--input_skel_folder`. Each mesh and rig pair must share same filenames. The rig files should follow the RigNet format containing:

```
joints [joint_name] [x] [y] [z]
root [root_joint_name]
hier [parent_joint_name] [child_joint_name]
```

If you are using GLB files, refer to `skeleton/data_utils/read_rig_mesh_from_glb.py` for reading the mesh and rig. After predict skinning weights, we will save the final rig files by adding skinning lines:

```
skin [vertex_index] [joints_name1] [skinning_weight1] [joints_name2] [skinning_weight2] ...
```

⚠️ Note that meshes with complex topology may require more data processing time.

## Visualization
The skinning visualizations shown in the paper can be reproduced using `utils/visualize.py`. This script generates two types of visualizations: (1) objects with skinning weights represented as colors, and (2) objects with L1 error maps that highlight differences between predicted and ground truth skinning weights.