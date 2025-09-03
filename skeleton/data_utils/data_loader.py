#  Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import glob
import numpy as np
import trimesh

class DataLoader:
    def __init__(self):
        self.joint_name_to_idx = {}

    def load_rig_data(self, rig_path):
        joints = []
        joints_names = []
        bones = []

        with open(rig_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == 'joints':
                    joint_name = parts[1]
                    joint_pos = [float(parts[2]), float(parts[3]), float(parts[4])]
                    self.joint_name_to_idx[joint_name] = len(joints)
                    joints.append(joint_pos)
                    joints_names.append(joint_name)
                elif parts[0] == 'root':
                    self.root_name = parts[1]
                elif parts[0] == 'hier':
                    parent_joint = self.joint_name_to_idx[parts[1]]
                    child_joint = self.joint_name_to_idx[parts[2]]
                    bones.append([parent_joint, child_joint])

        self.joints = np.array(joints)
        self.bones = np.array(bones)
        self.joints_names = joints_names
        self.root_idx = None
        if self.root_name is not None:
            self.root_idx = self.joint_name_to_idx[self.root_name]

    def load_mesh(self, mesh_path):
        mesh = trimesh.load(mesh_path, process=False)
        mesh.visual.vertex_colors[:, 3] = 100  # set transparency
        self.mesh = mesh
        
        # Compute the centroid normal of the mesh
        v = self.mesh.vertices
        xmin, ymin, zmin = v.min(axis=0)
        xmax, ymax, zmax = v.max(axis=0)
        self.bbox_center = np.array([(xmax + xmin)/2, (ymax + ymin)/2, (zmax + zmin)/2])
        self.bbox_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        self.bbox_scale = max(xmax - xmin, ymax - ymin, zmax - zmin)

        normal = mesh.center_mass - self.bbox_center
        normal = normal / (np.linalg.norm(normal)+1e-5)

        # Choose axis order based on normal direction
        if abs(normal[1]) > abs(normal[2]):  # if Y component is dominant
            self.axis_order = [0, 1, 2]  # swapping Y and Z
        else:
            self.axis_order =[0, 2, 1]  # keep default order

        self.mesh.vertices = self.mesh.vertices[:, self.axis_order]
        self.joints = self.joints[:, self.axis_order]
        self.normalize_coordinates()

    def normalize_coordinates(self):
        
        # Compute scale and offset
        scale = 1.0 / (self.bbox_scale+1e-5)
        offset = -self.bbox_center

        self.mesh.vertices = (self.mesh.vertices + offset) * scale
        self.joints = (self.joints + offset) * scale
        
        # Calculate appropriate radii based on the mean size
        self.joint_radius = 0.01 
        self.bone_radius = 0.005

    def query_mesh_rig(self):
        
        input_dict = {"shape": self.mesh}

        # Create joints as spheres
        joint_meshes = []
        for i, joint in enumerate(self.joints):
            
            sphere = trimesh.creation.icosphere(
                radius=self.joint_radius, subdivisions=2
            )
            sphere.apply_translation(joint)
            if i == self.root_idx:
                # root green
                sphere.visual.vertex_colors = [0, 255, 0, 255]
            else:
                sphere.visual.vertex_colors = [0, 0, 255, 255]
            
            joint_meshes.append(sphere)
        input_dict["joint_meshes"] = trimesh.util.concatenate(joint_meshes)

        # Create bones as cylinders
        bone_meshes = []
        for bone in self.bones:
            start, end = self.joints[bone[0]], self.joints[bone[1]]
            cylinder = trimesh.creation.cylinder(radius=self.bone_radius, segment=np.array([[0, 0, 0], end - start]))
            cylinder.apply_translation(start)
            cylinder.visual.vertex_colors = [255, 0, 0, 255]  #[0, 0, 255, 255]  # blue
            bone_meshes.append(cylinder)
        input_dict["bone_meshes"] = trimesh.util.concatenate(bone_meshes)

        return input_dict