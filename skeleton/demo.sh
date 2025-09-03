CUDA_VISIBLE_DEVICES=0 python demo.py --input_dir ./examples \
            --pretrained_weights skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth \
            --save_name infer_results_demo --input_pc_num 8192 \
            --save_render --apply_marching_cubes --joint_token --seq_shuffle

# If you found the results not satisfactory, try the model trained with bone-based tokenization:

# CUDA_VISIBLE_DEVICES=0 python demo.py --input_dir ./examples \
#             --pretrained_weights skeleton_ckpts/puppeteer_skeleton_w_diverse_pose_bone_token.pth \
#             --save_name infer_results_demo_bone_token --input_pc_num 8192 \
#             --save_render --apply_marching_cubes --hier_order --seq_shuffle


# If you want to run the demo using MagicArticulate weights, run:

# CUDA_VISIBLE_DEVICES=0 python demo.py --input_dir ./examples \
#             --pretrained_weights skeleton_ckpts/checkpoint_trainonv2_hier.pth \
#             --save_name infer_results_demo_magicarti --input_pc_num 8192 \
#             --save_render --apply_marching_cubes --hier_order