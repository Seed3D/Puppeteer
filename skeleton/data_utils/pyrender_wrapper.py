# Modified from https://github.com/lab4d-org/lab4d

import os
import numpy as np
import cv2
import pyrender
import trimesh
from pyrender import (
    IntrinsicsCamera,
    Mesh,
    Node,
    Scene,
    OffscreenRenderer,
    MetallicRoughnessMaterial,
    RenderFlags
)

os.environ["PYOPENGL_PLATFORM"] = "egl"

def look_at(eye, center, up):
    """Create a look-at (view) matrix."""
    f = np.array(center, dtype=np.float32) - np.array(eye, dtype=np.float32)
    f /= np.linalg.norm(f)

    u = np.array(up, dtype=np.float32)
    u /= np.linalg.norm(u)

    s = np.cross(f, u)
    u = np.cross(s, f)

    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -np.matmul(m[:3, :3], np.array(eye, dtype=np.float32))

    return m

class PyRenderWrapper:
    def __init__(self, image_size=(1024, 1024)) -> None:
        # renderer
        self.image_size = image_size
        render_size = max(image_size)
        self.r = OffscreenRenderer(render_size, render_size)
        self.intrinsics = IntrinsicsCamera(
            render_size, render_size, render_size / 2, render_size / 2
        )
        # light
        self.light_pose = np.eye(4)
        self.set_light_topdown()
        self.direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        self.material = MetallicRoughnessMaterial(
            roughnessFactor=0.75, metallicFactor=0.75, alphaMode="BLEND"
        )
        self.init_camera()

    def init_camera(self):
        self.flip_pose = np.eye(4)
        self.set_camera(np.eye(4))

    def set_camera(self, scene_to_cam):
        # object to camera transforms
        self.scene_to_cam = self.flip_pose @ scene_to_cam
        
    def set_light_topdown(self, gl=False):
        # top down light, slightly closer to the camera
        if gl:
            rot = cv2.Rodrigues(np.asarray([-np.pi / 2, 0, 0]))[0]
        else:
            rot = cv2.Rodrigues(np.asarray([np.pi / 2, 0, 0]))[0]
        self.light_pose[:3, :3] = rot

    def align_light_to_camera(self):
        self.light_pose = np.linalg.inv(self.scene_to_cam)

    def set_intrinsics(self, intrinsics):
        """
        Args:
            intrinsics: (4,) fx,fy,px,py
        """
        self.intrinsics = IntrinsicsCamera(
            intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        )

    def get_cam_to_scene(self):
        cam_to_scene = np.eye(4)
        cam_to_scene[:3, :3] = self.scene_to_cam[:3, :3].T
        cam_to_scene[:3, 3] = -self.scene_to_cam[:3, :3].T @ self.scene_to_cam[:3, 3]
        return cam_to_scene
    
    def set_camera_view(self, angle, bbox_center, distance=2.0):
        # Calculate camera position based on angle and distance from bounding box center
        camera_position = bbox_center + distance * np.array([np.sin(angle), 0, np.cos(angle)], dtype=np.float32)
        look_at_matrix = look_at(camera_position, bbox_center, [0, 1, 0])
        self.scene_to_cam = look_at_matrix @ self.flip_pose
    
    def render(self, input_dict):
        # Create separate scenes for transparent objects (mesh) and solid objects (joints and bones)
        scene_transparent = Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]) * 0.1)
        scene_solid = Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]) * 0.1)

        mesh_pyrender = Mesh.from_trimesh(input_dict["shape"], smooth=False)
        mesh_pyrender.primitives[0].material = self.material
        scene_transparent.add(mesh_pyrender, pose=np.eye(4), name="shape")

        if "joint_meshes" in input_dict:
            joints_pyrender = Mesh.from_trimesh(input_dict["joint_meshes"], smooth=False)
            joints_pyrender.primitives[0].material = self.material
            scene_solid.add(joints_pyrender, pose=np.eye(4), name="joints")

        if "bone_meshes" in input_dict:
            bones_pyrender = Mesh.from_trimesh(input_dict["bone_meshes"], smooth=False)
            bones_pyrender.primitives[0].material = self.material
            scene_solid.add(bones_pyrender, pose=np.eye(4), name="bones")

        # Camera for both scenes
        scene_transparent.add(self.intrinsics, pose=self.get_cam_to_scene())
        scene_solid.add(self.intrinsics, pose=self.get_cam_to_scene())

        # Light for both scenes
        scene_transparent.add(self.direc_l, pose=self.light_pose)
        scene_solid.add(self.direc_l, pose=self.light_pose)

        # Render transparent scene first
        color_transparent, depth_transparent = self.r.render(scene_transparent)

        # Render solid scene on top
        color_solid, depth_solid = self.r.render(scene_solid)

        # Combine the two scenes
        color_combined = np.where(depth_solid[..., np.newaxis] == 0, color_transparent, color_solid)

        return color_combined, depth_solid
    def delete(self):
        self.r.delete()
