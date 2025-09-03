
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=10011 main.py \
         --num_workers 1 --batch_size 1 --depth 1 --eval \
         --xl_test --save_skin_npy --save_folder outputs \
         --eval_data_path articulation_xl2_test.h5 \
         --pretrained_weights skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth

# remember to change eval_data_path and pass [xl_test, pose_test, modelres_test] when evaluating.