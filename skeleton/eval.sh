CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset_path articulation_xlv2_test.npz \
            --pretrained_weights skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth \
            --save_name infer_results_xl --input_pc_num 8192 \
            --save_render --joint_token --seq_shuffle

# when evaluate on xl2.0-test, it needs time as we have 2000 data for inference.
# change dataset_path and save_name when evaluating on other test sets.


# If you also want to evaluate MagicArticulate, run:

# CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset_path articulation_xlv2_test.npz \
#             --pretrained_weights skeleton_ckpts/checkpoint_trainonv2_hier.pth \
#             --save_name infer_results_xl_magicarti --input_pc_num 8192 \
#             --save_render --hier_order