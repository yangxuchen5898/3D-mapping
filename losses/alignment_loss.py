import torch

def compute_alignment_loss(xyz, normals, knn_idx, sigma_d, beta1, beta2):
    """
    Computes the surface-aligned regularization inspired by SuGaR.
    
    xyz: [N, 3] primitive centers
    normals: [N, 3] primitive normals
    knn_idx: [N, k] pre-computed neighbor indices
    sigma_d: scalar, controls distance weighting
    beta1: weight for L_plane
    beta2: weight for L_normal
    """
    N, k = knn_idx.shape
    
    # [N, k, 3]
    neighbor_xyz = xyz[knn_idx]
    neighbor_normals = normals[knn_idx]
    
    # [N, 1, 3]
    center_xyz = xyz.unsqueeze(1)
    center_normals = normals.unsqueeze(1)
    
    # Compute weights
    # ||x_j - x_i||^2
    dist_sq = torch.sum((neighbor_xyz - center_xyz) ** 2, dim=-1)
    weight = torch.exp(-dist_sq / (sigma_d ** 2)) # [N, k]
    
    # 1. Point to local plane alignment L_plane
    # n_i^T (x_j - x_i)
    plane_dist = torch.sum(center_normals * (neighbor_xyz - center_xyz), dim=-1)
    l_plane = torch.sum(weight * (plane_dist ** 2)) / N
    
    # 2. Local normal consistency L_normal
    # | n_i^T n_j |
    normal_dot = torch.sum(center_normals * neighbor_normals, dim=-1)
    l_normal = torch.sum(weight * (1.0 - torch.abs(normal_dot))) / N
    
    align_loss = beta1 * l_plane + beta2 * l_normal
    
    return align_loss, l_plane, l_normal
