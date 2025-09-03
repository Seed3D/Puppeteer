#!/bin/bash

echo "Starting Puppeteer rigging pipeline..."

mkdir -p results

# skeleton
echo "Running skeleton generation..."
cd skeleton
python demo.py \
    --input_dir ../examples \
    --pretrained_weights skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth \
    --output_dir ../results/ --save_name skel_results --input_pc_num 8192 \
    --save_render --apply_marching_cubes --joint_token --seq_shuffle

echo "Skeleton generation completed."

# If you found the results not satisfactory, try the model trained with bone-based tokenization:
# echo "Running skeleton with bone tokenization..."
# python demo.py \
#     --input_dir ../examples \
#     --pretrained_weights skeleton_ckpts/puppeteer_skeleton_w_diverse_pose_bone_token.pth \
#     --output_dir ../results/ --save_name skel_results --input_pc_num 8192 \
#     --save_render --apply_marching_cubes --hier_order --seq_shuffle


# Copy generated rig files to skeletons for following skinning
echo "Copying generated rig files..."
mkdir -p ../results/skeletons/
cd ../results/skel_results/
for file in *_pred.txt; do
    if [ -f "$file" ]; then
        new_name=$(echo "$file" | sed 's/_pred\.txt$/.txt/')
        cp "$file" "../skeletons/$new_name"
    fi
done
cd ../../skeleton
echo "Rig files copied to results/skeletons/"


# skinning
# Note that meshes with complex topology may require more data processing time.
echo "Running skinning..."
cd ../skinning
CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port=10009 \
    main.py \
    --num_workers 1 --batch_size 1 --generate --save_skin_npy \
    --pretrained_weights skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth \
    --input_skel_folder ../results/skeletons \
    --mesh_folder ../examples \
    --post_filter --depth 1 --save_folder ../results/skin_results

echo "Skinning completed."


echo "Copying generated skin files..."
mkdir -p ../results/final_rigging/
cd ../results/skin_results/generate/
for file in *_skin.txt; do
    if [ -f "$file" ]; then
        new_name=$(echo "$file" | sed 's/_skin\.txt$/.txt/')
        cp "$file" "../../final_rigging/$new_name"
    fi
done
cd ../../../
echo "Final rig files copied to results/final_rigging/"