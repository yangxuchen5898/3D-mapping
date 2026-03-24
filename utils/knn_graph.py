import torch

def build_knn_graph(xyz, k, chunk_size=4096):
    """
    Builds a 3D KNN graph for the primitives using a chunk-based approach to prevent OOM.
    xyz: [N, 3] primitive centers. Should be detached so gradients don't flow through neighbor selection.
    k: number of nearest neighbors
    chunk_size: number of query points to process at once
    
    Returns:
    knn_idx: [N, min(k, N-1)] indices of nearest neighbors excluding the point itself.
    """
    N = xyz.shape[0]
    
    # Safety feature for extremely small point clouds (early training stages or small scenes)
    actual_k = min(k, N - 1)
    if actual_k <= 0:
        return torch.empty((N, 0), dtype=torch.long, device=xyz.device)

    knn_idx_list = []
    
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            xyz_chunk = xyz[i:end_idx]  # [C, 3]
            
            # Compute distance between this chunk and all points => [C, N]
            dist_chunk = torch.cdist(xyz_chunk, xyz)
            
            # topk to find the actual_k + 1 nearest (first is usually self)
            _, idx_chunk = torch.topk(dist_chunk, k=actual_k + 1, dim=1, largest=False, sorted=True)
            
            # Exclude self. Because distance to self is exactly 0, it should be the first column
            # idx_chunk is [C, actual_k+1], we take from 1: leaving [C, actual_k]
            idx_chunk = idx_chunk[:, 1:]
            
            knn_idx_list.append(idx_chunk)
            
        knn_idx = torch.cat(knn_idx_list, dim=0)

    return knn_idx
