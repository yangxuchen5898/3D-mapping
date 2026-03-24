import torch

def build_knn_graph(xyz, k):
    """
    Builds a 3D KNN graph for the primitives.
    xyz: [N, 3] primitive centers. Should be detached so gradients don't flow through neighbor selection.
    k: number of nearest neighbors
    
    Returns:
    knn_idx: [N, k] indices of nearest neighbors excluding the point itself.
    """
    with torch.no_grad():
        # compute pairwise distance squared
        dist = torch.cdist(xyz, xyz)
        # topk to find the k+1 nearest (first is self)
        # we want smallest, so we set largest=False
        _, idx = torch.topk(dist, k=k+1, dim=1, largest=False, sorted=True)
        # exclude self (which is at index 0 because dist to self is 0)
        knn_idx = idx[:, 1:]
    return knn_idx
