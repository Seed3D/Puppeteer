# Modified from https://github.com/zhan-xu/RigNet

import numpy as np

##### for quantitative calculation
def chamfer_dist(pt1, pt2):
    pt1 = pt1[np.newaxis, :, :]
    pt2 = pt2[:, np.newaxis, :]
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
    min_left = np.mean(np.min(dist, axis=0))
    min_right = np.mean(np.min(dist, axis=1))
    return (min_left + min_right) / 2

def oneway_chamfer(pt_src, pt_dst):
    pt1 = pt_src[np.newaxis, :, :]
    pt2 = pt_dst[:, np.newaxis, :]
    dist = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=2))
    avg_dist = np.mean(np.min(dist, axis=0))
    return avg_dist

def joint2bone_chamfer_dist(joints1, bones1, joints2, bones2):
    bone_sample_1 = sample_skel(joints1, bones1)
    bone_sample_2 = sample_skel(joints2, bones2)
    dist1 = oneway_chamfer(joints1, bone_sample_2)
    dist2 = oneway_chamfer(joints2, bone_sample_1)
    return (dist1 + dist2) / 2

def bone2bone_chamfer_dist(joints1, bones1, joints2, bones2):
    bone_sample_1 = sample_skel(joints1, bones1)
    bone_sample_2 = sample_skel(joints2, bones2)
    return chamfer_dist(bone_sample_1, bone_sample_2)

def sample_bone(p_pos, ch_pos):
    ray = ch_pos - p_pos

    bone_length = np.linalg.norm(p_pos - ch_pos)
    num_step = np.round(bone_length / 0.005).astype(int)
    i_step = np.arange(0, num_step + 1)
    unit_step = ray / (num_step + 1e-30)
    unit_step = np.repeat(unit_step[np.newaxis, :], num_step + 1, axis=0)
    res = p_pos + unit_step * i_step[:, np.newaxis]
    return res

def sample_skel(joints, bones):
    bone_sample = []
    for parent_idx, child_idx in bones:
        p_pos = joints[parent_idx]
        ch_pos = joints[child_idx]
        res = sample_bone(p_pos, ch_pos)
        bone_sample.append(res)
    
    if bone_sample:
        bone_sample = np.concatenate(bone_sample, axis=0)
    else:
        bone_sample = np.empty((0, 3))
    
    return bone_sample