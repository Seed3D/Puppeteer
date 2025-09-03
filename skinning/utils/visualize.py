# Modified from https://github.com/PeizhuoLi/neural-blend-shapes/blob/main/blender_scripts/vertex_color.py
import bpy
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def generate_color_table(num_colors, base_color_path):
    base_colors = np.load(base_color_path)
    idx = list(range(base_colors.shape[0]))
    random.seed(5)
    random.shuffle(idx)
    pt = 0
    res = []
    for i in range(num_colors):
        res.append(base_colors[idx[pt]])
        pt += 1
        pt %= base_colors.shape[0]
    return np.array(res)

def weight2color(weight, colors):
    res = np.matmul(weight, colors)
    res = np.clip(res, 0, 1)
    return res

def export_obj_with_vertex_colors(obj_object, vertex_colors, filepath):
    mesh = obj_object.data

    with open(filepath, 'w') as f:
        f.write("# Exported OBJ with vertex colors\n")
        for v, color in zip(mesh.vertices, vertex_colors):
            f.write("v {} {} {} {} {} {}\n".format(
                v.co.x, v.co.y, v.co.z,
                color[0], color[1], color[2]
            ))
        
        f.write("g {}\n".format(obj_object.name))
        for poly in mesh.polygons:
            vertices = [v + 1 for v in poly.vertices]
            face = "f " + " ".join([str(v) for v in vertices]) + "\n"
            f.write(face)

def compute_error_map(weight_method, weight_gt):
    error = np.abs(weight_method - weight_gt)  # Shape: (n_vertex, n_color)
    error_map = np.max(error, axis=1)  # Taking the maximum error across all colors for each vertex
    # error_map = np.mean(error, axis=1) 
    # Normalize error_map to [0,1]
    error_map = np.clip(error_map, 0, 1)
    return error_map

def error_map_to_color(error_map, colormap='Reds'):
    cmap = plt.get_cmap(colormap)
    colors = cmap(error_map)[:, :3]  # Discard alpha channel
    return colors

def generate_color_bar(filepath, colormap='Reds'):
    fig, ax = plt.subplots(figsize=(8, 1.5))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=0, vmax=1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='horizontal')
    cb1.ax.tick_params(labelsize=26) 
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def main():
    base_color_path = "utils/tableau_color.npy"

    ### xlv2_test
    obj_path = "xlv2_test_mesh" # save from npz files, check skeleton/data_utils/convert_npz_to_mesh_rig.py
    weight_path1 = "gt_onxlv2_npy"
    weight_path2 = "ours_onxlv2_npy"
    # weight_path3 = "rignet_onxlv2_npy"
    # weight_path4 = "gvb_onxlv2_npy"
    save_path = "obj_w_skincolor_errormap_onxlv2_test"
    
    color_bar_path = "error_color_bar.png" # Path to save the color bar image

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    generate_color_bar(color_bar_path, colormap='Reds')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for obj_file in os.listdir(obj_path):
        obj_file_path = os.path.join(obj_path, obj_file)
        wt_gt_file = os.path.join(weight_path1, obj_file.replace(".obj", "_skin.npy"))
        wt_ours_file = os.path.join(weight_path2, obj_file.replace(".obj", "_skin.npy"))
        # wt_rignet_file = os.path.join(weight_path3, obj_file.replace(".obj", "_skin.npy"))
        # wt_gvb_file = os.path.join(weight_path4, obj_file.replace(".obj", "_skin.npy"))
        if not os.path.exists(wt_gt_file) or not os.path.exists(wt_ours_file): # or not os.path.exists(wt_rignet_file) or not os.path.exists(wt_gvb_file):
            continue

        # 导入OBJ文件
        bpy.ops.wm.obj_import(filepath=obj_file_path)
        imported_objects = bpy.context.selected_objects
        if not imported_objects:
            print("Failed to import OBJ file:", obj_file_path)
            continue
        obj = imported_objects[0]

        weight_gt = np.load(wt_gt_file)
        weight_ours = np.load(wt_ours_file)
        # weight_rignet = np.load(wt_rignet_file)
        # weight_gvb = np.load(wt_gvb_file)
        if weight_gt.shape != weight_ours.shape: # or weight_gt.shape != weight_rignet.shape or weight_gt.shape != weight_gvb.shape:
            print("Weight shape mismatch among files")
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            continue

        if len(weight_gt.shape) != 2:
            print("Weight file should be 2D array")
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            continue

        n_vertices = len(obj.data.vertices)
        if weight_gt.shape[0] != n_vertices:
            print(f"Vertex count mismatch: OBJ has {n_vertices}, but weight file has {weight_gt.shape[0]}")
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            continue

        # generate colors
        n_color = weight_gt.shape[1]
        colors = generate_color_table(n_color, base_color_path)
        vertex_colors_gt = weight2color(weight_gt, colors)
        vertex_colors_ours = weight2color(weight_ours, colors)
        # vertex_colors_rignet = weight2color(weight_rignet, colors)
        # vertex_colors_gvb = weight2color(weight_gvb, colors)

        # Save obj with vertex colors
        output_path_gt = os.path.join(save_path, obj_file.replace(".obj", "_gt.obj"))
        output_path_ours = os.path.join(save_path, obj_file.replace(".obj", "_ours.obj"))
        # output_path_rignet = os.path.join(save_path, obj_file.replace(".obj", "_rignet.obj"))
        # output_path_gvb = os.path.join(save_path, obj_file.replace(".obj", "_gvb.obj"))
        export_obj_with_vertex_colors(obj, vertex_colors_gt, output_path_gt)
        export_obj_with_vertex_colors(obj, vertex_colors_ours, output_path_ours)
        # export_obj_with_vertex_colors(obj, vertex_colors_rignet, output_path_rignet)
        # export_obj_with_vertex_colors(obj, vertex_colors_gvb, output_path_gvb)

        # Save object with error color map
        # 1. Ours vs GT
        error_ours = compute_error_map(weight_ours, weight_gt)
        colors_error_ours = error_map_to_color(error_ours, colormap='Reds')
        output_path_error_ours = os.path.join(save_path, obj_file.replace(".obj", "_error_ours.obj"))
        export_obj_with_vertex_colors(obj, colors_error_ours, output_path_error_ours)

        # 2. Rignet vs GT
        # error_rignet = compute_error_map(weight_rignet, weight_gt)
        # colors_error_rignet = error_map_to_color(error_rignet, colormap='Reds')
        # output_path_error_rignet = os.path.join(save_path, obj_file.replace(".obj", "_error_rignet.obj"))
        # export_obj_with_vertex_colors(obj, colors_error_rignet, output_path_error_rignet)

        # 3. GVB vs GT
        # error_gvb = compute_error_map(weight_gvb, weight_gt)
        # colors_error_gvb = error_map_to_color(error_gvb, colormap='Reds')
        # output_path_error_gvb = os.path.join(save_path, obj_file.replace(".obj", "_error_gvb.obj"))
        # export_obj_with_vertex_colors(obj, colors_error_gvb, output_path_error_gvb)

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    print("Done")

if __name__ == "__main__":
    main()
