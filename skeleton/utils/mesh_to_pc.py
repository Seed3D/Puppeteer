# Modified from https://github.com/buaacyw/MeshAnything
import mesh2sdf.core
import numpy as np
import skimage.measure
import trimesh
import time
from typing import List, Tuple

class MeshProcessor:
    """A class to handle mesh normalization, watertight conversion and point cloud sampling."""
    
    @staticmethod
    def normalize_mesh_vertices(vertices: np.ndarray, scaling_factor: float = 0.95) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Normalize mesh vertices to be centered at origin and scaled appropriately.
        """
        min_bounds = vertices.min(axis=0)
        max_bounds = vertices.max(axis=0)
        
        center = (min_bounds + max_bounds) * 0.5
        max_dimension = (max_bounds - min_bounds).max()
        scale = 2.0 * scaling_factor / max_dimension
        
        normalized_vertices = (vertices - center) * scale
        return normalized_vertices, center, scale

    @staticmethod
    def convert_to_watertight(mesh: trimesh.Trimesh, octree_depth: int = 7) -> trimesh.Trimesh:
        """
        Convert to watertight using mesh2sdf and marching cubes.
        """
        grid_size = 2 ** octree_depth
        iso_level = 2 / grid_size
        
        # Normalize vertices for SDF computation
        normalized_vertices, original_center, original_scale = MeshProcessor.normalize_mesh_vertices(mesh.vertices)
        
        # Compute signed distance field
        sdf = mesh2sdf.core.compute(normalized_vertices, mesh.faces, size=grid_size)
        
        # Run marching cubes algorithm
        vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), iso_level)
        
        # Transform vertices back to original coordinate system
        vertices = vertices / grid_size * 2 - 1  # Map to [-1, 1] range
        vertices = vertices / original_scale + original_center
        
        # Create new watertight mesh
        watertight_mesh = trimesh.Trimesh(vertices, faces, normals=normals)
        return watertight_mesh

    @staticmethod
    def convert_meshes_to_point_clouds(
        meshes: List[trimesh.Trimesh], 
        points_per_mesh: int = 8192, 
        apply_marching_cubes: bool = False, 
        octree_depth: int = 7
    ) -> List[np.ndarray]:
        """
        Process a list of meshes into point clouds with normals.
        """
        point_clouds_with_normals = []
        processed_meshes = []
        
        for mesh in meshes:
            # Optionally convert to watertight mesh
            if apply_marching_cubes:
                start_time = time.time()
                mesh = MeshProcessor.convert_to_watertight(mesh, octree_depth=octree_depth)
                processing_time = time.time() - start_time
                print(f"Marching cubes complete! Time: {processing_time:.2f}s")
            
            # Store processed mesh
            processed_meshes.append(mesh)
            
            # Sample points and get corresponding face normals
            points, face_indices = mesh.sample(points_per_mesh, return_index=True)
            point_normals = mesh.face_normals[face_indices]
            
            # Combine points and normals
            points_with_normals = np.concatenate([points, point_normals], axis=-1, dtype=np.float16)
            point_clouds_with_normals.append(points_with_normals)

        return point_clouds_with_normals