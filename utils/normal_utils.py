import torch

def orientation_to_normal(rotation):
    """
    Converts unit quaternions to normal vectors.
    rotation: [N, 4] unit quaternions (w, x, y, z) or (r, x, y, z)
    Returns:
    normals: [N, 3] extracted normal vectors (third column of the rotation matrix)
    """
    r = rotation[:, 0]
    x = rotation[:, 1]
    y = rotation[:, 2]
    z = rotation[:, 3]
    
    # Third column of the rotation matrix constructed from quaternion
    # R[:, 0, 2] = 2 * (x*z + r*y)
    # R[:, 1, 2] = 2 * (y*z - r*x)
    # R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    nx = 2 * (x * z + r * y)
    ny = 2 * (y * z - r * x)
    nz = 1 - 2 * (x * x + y * y)
    
    normals = torch.stack([nx, ny, nz], dim=1)
    return normals
