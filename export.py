from collections import defaultdict
from typing import Union, Tuple

import argparse
import bpy # type: ignore
import numpy as np
import trimesh

from mathutils import Vector # type: ignore

def export_fbx(
    path: str,
    vertices: np.ndarray,
    faces: Union[np.ndarray, None],
    bones: np.ndarray,
    parents: list[Union[int, None]],
    names: list[str],
    vertex_group: np.ndarray,
    dfs_order: list[int],
    group_per_vertex: int=4,
):
    # clean bpy
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)
    # make mesh
    mesh = bpy.data.meshes.new('mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # make object from mesh
    object = bpy.data.objects.new('character', mesh)

    # make collection
    collection = bpy.data.collections.new('CA_collection')
    bpy.context.scene.collection.children.link(collection)

    # add object to scene collection
    collection.objects.link(object)

    # deselect mesh
    # mesh.select_set(False)
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.data.armatures.get('Armature')
    edit_bones = armature.edit_bones
    bone_root = edit_bones.get('Bone')

    J = len(names)
    def extrude_bone(
        edit_bones,
        name: str,
        parent_name: str,
        head: Tuple[float, float, float],
        tail: Tuple[float, float, float],
        is_root: bool=False,
    ):
        if is_root:
            bone = bone_root
        else:
            bone = edit_bones.new(name)
        bone.head = Vector((head[0], head[1], head[2]))
        bone.tail = Vector((tail[0], tail[1], tail[2]))
        bone.name = name
        if parent_name is not None:
            parent_bone = edit_bones.get(parent_name)
            bone.parent = parent_bone
        else:
            bone.parent = None
        bone.use_connect = True
    
    for k in range(J):
        i = dfs_order[k]
        edit_bones = armature.edit_bones
        if parents[i] is None: # root
            extrude_bone(edit_bones, names[i], None, bones[i, :3], bones[i, 3:], is_root=True)
        else:
            pname = 'Root' if parents[i] is None else names[parents[i]]
            extrude_bone(edit_bones, names[i], pname, bones[i, :3], bones[i, 3:])

    # must set to object mode to enable parent_set
    bpy.ops.object.mode_set(mode='OBJECT')
    objects = bpy.data.objects
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob = objects['character']
    arm = bpy.data.objects['Armature']
    ob.select_set(True)
    arm.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    vis = []
    for x in ob.vertex_groups:
        vis.append(x.name)
    #sparsify
    nGroupPerVertex = group_per_vertex
    argsorted = np.argsort(-vertex_group, axis=1)
    vertex_group_reweight = vertex_group[np.arange(vertex_group.shape[0])[..., None],argsorted] 
    vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :nGroupPerVertex].sum(axis=1)[...,None]

    for v, w in enumerate(vertex_group):
        for ii in range(nGroupPerVertex):
            i = argsorted[v,ii]
            if i >= len(names):
                continue
            n = names[i]
            if n not in vis:
                continue
            ob.vertex_groups[n].add([v], vertex_group_reweight[v, ii], 'REPLACE')

    bpy.ops.export_scene.fbx(filepath=path, check_existing=False, add_leaf_bones=False)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True, help="obj path of processed mesh")
    parser.add_argument("--res", type=str, required=True, help="path of prediction result")
    parser.add_argument("--output", type=str, required=False, default="res.fbx")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    mesh_path = args.mesh
    res_path = args.res
    output_path = args.output

    # change to face -y axis
    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0,-1.0],
        [0.0, 1.0, 0.0],
    ])

    # load original model
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = (rot @ mesh.vertices.T).T
    faces = mesh.faces

    N = vertices.shape[0]

    # handle RigNet output
    f_info = open(res_path, 'r')
    joint_pos = {}
    joint_hier = {}
    joint_skin = []
    id_mapping = {}
    name_mapping = {}
    parent_mapping = {}
    root_name = None
    root_pos = None
    tot = 0
    for line in f_info:
        word = line.split()
        if word[0] == 'joints':
            joint_pos[word[1]] = rot @ np.array([float(word[2]),  float(word[3]), float(word[4])])
            id_mapping[word[1]] = tot
            name_mapping[tot] = word[1]
            tot += 1
        if word[0] == 'root':
            root_pos = joint_pos[word[1]]
            root_name = word[1]
        if word[0] == 'hier':
            if word[1] not in joint_hier.keys():
                joint_hier[word[1]] = [word[2]]
            else:
                joint_hier[word[1]].append(word[2])
        if word[0] == 'skin':
            skin_item = word[1:]
            joint_skin.append(skin_item)
    f_info.close()
    J = len(joint_pos)
    bones = np.zeros((J, 6))

    parents = []
    names = []

    for name in joint_hier:
        for son in joint_hier[name]:
            parent_mapping[son] = name

    son = defaultdict(list)
    for i in range(J):
        name = name_mapping[i]
        names.append(name)
        parents.append(None if name==root_name else id_mapping[parent_mapping[name]])
        if name != root_name:
            son[id_mapping[parent_mapping[name]]].append(i)

    # extrude tails for blender
    for i in range(J):
        name = name_mapping[i]
        head = joint_pos[name]
        tail = head + np.array([0., 0., 0.1])
        if len(joint_hier.get(name, [])) == 1:
            tail = joint_pos[joint_hier[name][0]]
        elif name != root_name:
            pname = name_mapping[parents[i]]
            direction = joint_pos[name] - joint_pos[pname]
            tail = head + direction * 0.5
        bones[i, :3] = head
        bones[i, 3:] = tail

    # assign skin weight
    skin = np.zeros((N, J))
    for skin_item in joint_skin:
        u = int(skin_item[0])
        for j in range(1, len(skin_item), 2):
            id = id_mapping[skin_item[j]]
            w = skin_item[j + 1]
            skin[u, id] = w
    
    dfs_order = []
    Q = [id_mapping[root_name]]
    while Q:
        u = Q.pop()
        dfs_order.append(u)
        Q.extend(son[u])
    export_fbx(
        path=output_path,
        vertices=vertices,
        faces=faces,
        bones=bones,
        parents=parents,
        names=names,
        vertex_group=skin,
        dfs_order=dfs_order,
    )