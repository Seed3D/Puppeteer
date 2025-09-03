CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=10009 main.py \
         --num_workers 1 --batch_size 1 --generate --save_skin_npy \
         --pretrained_weights skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth \
         --input_skel_folder skel_folder \
         --mesh_folder mesh_folder \
         --post_filter --depth 1 --save_folder outputs

### We recommend enabling `--post_filter` to smooth skinning weights by averaging the weights of neighboring vertices.
### If results are unsatisfactory, try increasing `--depth` from 1 to 2 and updating the checkpoint path.
