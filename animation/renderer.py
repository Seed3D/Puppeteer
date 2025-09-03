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

import numpy as np
import torch
import cv2

from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch, Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, BlendParams, SoftSilhouetteShader
)
from utils.loss_utils import compute_visibility_mask_igl

def create_camera_from_blender_params(cam_params, device):
    """
    Convert Blender camera parameters to PyTorch3D camera
    
    Args:
        cam_params (dict): Camera parameters from Blender JSON
        device: Device to create camera on
    
    Returns:
        FoVPerspectiveCameras: Converted camera
    """
    # Extract matrix world and convert to rotation and translation
    matrix_world = torch.tensor(cam_params['matrix_world'], dtype=torch.float32)
    
    # Extract field of view (use x_fov, assuming symmetric FOV)
    fov = cam_params['x_fov'] * 180 / np.pi  # Convert radians to degrees

    rotation_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    
    # Apply transformations
    adjusted_matrix = rotation_matrix @ matrix_world
    world2cam_matrix_tensor = torch.linalg.inv(adjusted_matrix)
    
    aligned_matrix = torch.tensor([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)
    world2cam_matrix = aligned_matrix @ world2cam_matrix_tensor.to(device)
    cam2world_matrix = torch.linalg.inv(world2cam_matrix)
    
    # Extract rotation and translation
    R = cam2world_matrix[:3, :3]
    T = torch.tensor([
        world2cam_matrix[0, 3], 
        world2cam_matrix[1, 3], 
        world2cam_matrix[2, 3]
    ], device=device, dtype=torch.float32)
    
    return FoVPerspectiveCameras(
        device=device, 
        fov=fov, 
        R=R[None], 
        T=T[None],
        znear=0.1,
        zfar=100.0
    )
    
class MeshRenderer3D:
    """
    PyTorch3D mesh renderer with support for various rendering modes.
    
    Features:
    - Standard mesh rendering with Phong shading
    - Silhouette rendering
    - Multi-frame batch rendering  
    - Point projection with visibility computation
    """
    def __init__(self, device, image_size=1024, cam_params=None, light_params=None, raster_params=None):
        self.device = device
        # Initialize camera
        self.camera = self._setup_camera(cam_params)
        
        # Initialize light
        self.light = self._setup_light(light_params)
        
        # Initialize rasterization settings
        self.raster_settings = self._setup_raster_settings(raster_params, image_size)
        self.camera.image_size = self.raster_settings.image_size

        # Initialize renderers
        self._setup_renderers()

    def _setup_camera(self, cam_params):
        """Setup camera based on parameters."""
        if cam_params is None:
            # Default camera
            R, T = look_at_view_transform(3.0, 30, 20, at=[[0.0, 1.0, 0.0]])
            return FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        # Check if Blender parameters
        if "matrix_world" in cam_params and "x_fov" in cam_params:
            return create_camera_from_blender_params(cam_params, self.device)
        else:
            raise ValueError("Need to provide blender parameters.")

    def _setup_light(self, light_params):
        """Setup light source."""
        if light_params is None:
            return PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        
        location = [[
            light_params.get('light_x', 0.0),
            light_params.get('light_y', 0.0),
            light_params.get('light_z', 3.0)
        ]]
        return PointLights(device=self.device, location=location)

    def _setup_raster_settings(self, raster_params, default_size):
        """Setup rasterization settings."""
        if raster_params is None:
            raster_params = {
                "image_size": [default_size, default_size],
                "blur_radius": 0.0,
                "faces_per_pixel": 1,
                "bin_size": 0,
                "cull_backfaces": False
            }
        
        return RasterizationSettings(**raster_params)

    def _setup_renderers(self) -> None:
        """Initialize main and silhouette renderers."""
        rasterizer = MeshRasterizer(
            cameras=self.camera,
            raster_settings=self.raster_settings
        )
        
        # Main renderer with Phong shading
        self.renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.camera,
                lights=self.light
            )
        )
        
        # Silhouette renderer
        blend_params = BlendParams(
            sigma=1e-4,
            gamma=1e-4,  
            background_color=(0.0, 0.0, 0.0)
        )
        
        self.silhouette_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

    def render(self, meshes):
        """
        Render meshes with Phong shading.
        
        Args:
            meshes: Single mesh or list of meshes
            
        Returns:
            Rendered images tensor of shape (1, H, W, C)
        """
        scene_mesh = self._prepare_scene_mesh(meshes)
        return self.renderer(scene_mesh)

    def render_batch(self, mesh_list):
        """
        Render multiple frames as a batch.
        
        Args:
            mesh_list: List of mesh lists (one per frame)
            
        Returns:
            Batch of rendered images of shape (B, H, W, C)
        """
        assert isinstance(mesh_list, list)

        batch_meshes = []
        for frame_meshes in mesh_list:
            scene_mesh = self._prepare_scene_mesh(frame_meshes)
            batch_meshes.append(scene_mesh)
        
        batch_mesh = join_meshes_as_batch(batch_meshes)
        return self.renderer(batch_mesh)

    def get_rasterization_fragments(self, mesh_list):
        """
        Get rasterization fragments for batch of meshes.
        
        Args:
            mesh_list: List of mesh lists (one per frame)
            
        Returns:
            Rasterization fragments
        """
        assert isinstance(mesh_list, list)
        
        batch_meshes = []
        for frame_meshes in mesh_list:
            scene_mesh = self._prepare_scene_mesh(frame_meshes)
            batch_meshes.append(scene_mesh)
        
        batch_mesh = join_meshes_as_batch(batch_meshes)
        return self.renderer.rasterizer(batch_mesh)

    def render_silhouette_batch(self, mesh_list):
        """
        Render silhouette masks for multiple frames.
        
        Args:
            mesh_list: List of mesh lists (one per frame)
            
        Returns:
            Batch of silhouette masks of shape (B, H, W, 1)
        """
        assert isinstance(mesh_list, list)
        
        batch_meshes = []
        for frame_meshes in mesh_list:
            scene_mesh = self._prepare_scene_mesh(frame_meshes)
            batch_meshes.append(scene_mesh)
        
        batch_mesh = join_meshes_as_batch(batch_meshes)
        silhouette = self.silhouette_renderer(batch_mesh)
        return silhouette[..., 3:]  # Return alpha channel

    def tensor_to_image(self, tensor):
        """
        Convert rendered tensor to numpy image array.
        
        Args:
            tensor: Rendered tensor of shape (B, H, W, C)
            
        Returns:
            Numpy array of shape (H, W, 3) with values in [0, 255]
        """
        return (tensor[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

    def project_points(self, points_3d):
        """
        Project 3D joints/vertices to 2D image plane
        
        Args:
            points_3d: shape (N, 3) or (B, N, 3) tensor of 3D points
            
        Returns:
            points_2d: shape (N, 2) or (B, N, 2) tensor of 2D projected points
        """
        if not torch.is_tensor(points_3d):
            points_3d = torch.tensor(points_3d, device=self.device, dtype=torch.float32)
        
        
        if len(points_3d.shape) == 2:
            points_3d = points_3d.unsqueeze(0)  # (1, N, 3)
        
        # project points
        projected = self.camera.transform_points_screen(points_3d, image_size=self.raster_settings.image_size)
        
        if projected.shape[0] == 1:
            projected_points = projected.squeeze(0)[:, :2]
        else:
            projected_points = projected[:, :, :2]
        return projected_points
    
    def render_with_points(self, meshes, points_3d, point_radius=3, for_vertices=False):
        """
        render the mesh and visualize the joints/vertices on the image
        
        Args:
            meshes: mesh or list of meshes to be rendered
            points_3d: shape (N, 3) tensor of 3D joints/vertices
            point_radius: radius of the drawn points
            for_vertices: if True, compute visibility for vertices, else for joints
            
        Returns:
            Image with joints/vertices drawn, visibility mask
        """
        rendered_image = self.render(meshes)
        
        # project 3D points to 2D
        points_2d = self.project_points(points_3d)
        
        image_np = rendered_image[0, ..., :3].cpu().numpy()
        image_with_points = image_np.copy()
        height, width = image_np.shape[:2]
        
        ray_origins = self.camera.get_camera_center()  # (B, 3)
        ray_origins = np.tile(ray_origins.detach().cpu().numpy(), (points_3d.shape[0], 1))

        verts = meshes.verts_packed().detach().cpu().numpy()
        faces = meshes.faces_packed().detach().cpu().numpy()
        
        ray_dirs = points_3d.detach().cpu().numpy() - ray_origins # calculate ray directions
        distances = np.linalg.norm(ray_dirs, axis=1)  # distances from camera to points
        ray_dirs = (ray_dirs.T / distances).T        # normalize to unit vectors
        
        vis_mask = compute_visibility_mask_igl(ray_origins, ray_dirs, distances, verts, faces, distance_tolerance=1e-6, for_vertices=for_vertices)
        
        # draw points
        visible_color=(1, 0, 0) # visible points are red
        invisible_color=(0, 0, 1) # invisible points are blue
        for i, point in enumerate(points_2d):
            x, y = int(point[0].item()), int(point[1].item())
        
            if 0 <= x < width and 0 <= y < height:
                point_color = visible_color if vis_mask[i] else invisible_color
                cv2.circle(image_with_points, (x, y), point_radius, point_color, -1)
        
        result = torch.from_numpy(image_with_points).to(self.device)
        result = result.unsqueeze(0)
        
        if rendered_image.shape[-1] == 4:
            alpha = rendered_image[..., 3:]
            result = torch.cat([result, alpha], dim=-1)
        
        return result, vis_mask

    def _prepare_scene_mesh(self, meshes):
        """Convert meshes to a single scene mesh."""
        if isinstance(meshes, Meshes):
            return meshes
        elif isinstance(meshes, list):
            return join_meshes_as_scene(meshes)
        else:
            raise ValueError("meshes must be Meshes object or list of Meshes")


        