import trimesh
import numpy as np
from trimesh.util import concatenate
from trimesh import Scene

RESOLUTION = 16
SIZE       = RESOLUTION - 2

def voxelize(path):

    mesh = trimesh.load_mesh(path, process=True, validate=True)

    if isinstance(mesh, Scene):

        geometries = list(mesh.geometry.values())
        mesh = concatenate(geometries)

    x_size = mesh.bounds[1][0] - mesh.bounds[0][0]
    y_size = mesh.bounds[1][1] - mesh.bounds[0][1]
    z_size = mesh.bounds[1][2] - mesh.bounds[0][2]

    max_size = max([x_size, y_size, z_size])

    mesh.bounds, x_size, y_size, z_size, max_size

    scale = SIZE / max_size

    mesh.apply_scale(scale)

    voxels = mesh.voxelized(pitch=1.0)
    voxels = voxels.fill().matrix

    grid = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))

    coordinates = np.argwhere(voxels == True)

    for c in coordinates:
        grid[c[0], c[2], c[1]] = 1

    return grid
