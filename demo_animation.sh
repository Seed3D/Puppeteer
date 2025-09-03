#!/bin/bash

echo "Running animation..."

# copy rig and mesh for animation
for txt_file in results/final_rigging/*.txt; do
    if [ -f "$txt_file" ]; then
        seq_name=$(basename "$txt_file" .txt)
        
        mkdir -p "examples/$seq_name/objs/"
        
        cp "$txt_file" "examples/$seq_name/objs/rig.txt"
        echo "Copied $txt_file -> examples/$seq_name/objs/rig.txt"
        
        obj_file="examples/$seq_name.obj"
        if [ -f "$obj_file" ]; then
            cp "$obj_file" "examples/$seq_name/objs/mesh.obj"
            echo "Copied $obj_file -> examples/$seq_name/objs/mesh.obj"
        else
            echo "Warning: $obj_file not found"
        fi
        
        # extract frames
        video_file="examples/$seq_name/input.mp4"
        if [ -f "$video_file" ]; then
            echo "Found video file: $video_file"
            cd "examples/$seq_name"
            mkdir -p imgs
            ffmpeg -i input.mp4 -vf fps=10 imgs/frame_%04d.png -y
            echo "Extracted frames from $video_file to imgs/"
            cd ../../
        else
            echo "No video file found: $video_file"
        fi
    fi
done

cd animation

# save flow
echo "Processing sequences with save_flow.py..."
for seq_dir in ../examples/*/; do
    if [ -d "$seq_dir" ]; then
        seq_name=$(basename "$seq_dir")
        echo "Processing sequence: $seq_name"
        python utils/save_flow.py --input_path ../examples --seq_name "$seq_name"
    fi
done

# animation
echo "Running optimization for each sequence..."
mkdir -p ../results/animation

python optimization.py --save_path ../results/animation --iter 200 --input_path ../examples --img_size 960 \
        --seq_name 'cat' --save_name 'cat_demo'

echo "Animation completed."

cd ..
echo "Puppeteer pipeline completed successfully!"