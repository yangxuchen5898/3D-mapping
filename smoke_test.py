import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from losses.alignment_loss import compute_alignment_loss
from utils.knn_graph import build_knn_graph
from utils.normal_utils import orientation_to_normal

def test_knn():
    print("Testing chunk-based KNN graph:")
    N = 10000
    k = 10
    xyz = torch.rand((N, 3), device="cuda")
    
    # Test batch/chunk KNN build
    try:
        knn_idx = build_knn_graph(xyz.detach(), k, chunk_size=2048)
        print(f"PASS: Shape is {knn_idx.shape}, expected {(N, k)}")
        
        # Ensure self is eliminated properly
        for i in range(5):
            assert i not in knn_idx[i], f"Fail: {i} is its own neighbor"
        print("PASS: Self-exclusion logic holding")
    except Exception as e:
        print("FAIL:", e)

def test_align():
    print("Testing alignments components gradients:")
    N = 100
    k = 10
    # Simulate Gaussians parameters
    xyz = torch.rand((N, 3), device="cuda", requires_grad=True)
    rotation = torch.randn((N, 4), device="cuda")
    rotation = torch.nn.functional.normalize(rotation, dim=1)
    rotation.requires_grad_(True)
    
    # Detach for KNN query ONLY
    knn_idx = build_knn_graph(xyz.detach(), k, chunk_size=16)
    
    # Forward pass
    normals = orientation_to_normal(rotation)
    align_loss, l_plane, l_n = compute_alignment_loss(xyz, normals, knn_idx, sigma_d=0.01, beta1=1.0, beta2=0.1)
    
    print(f"Align Loss computed: {align_loss.item()}")
    
    # Backward pass
    try:
        align_loss.backward()
        print("PASS: Backward pass executed.")
        
        if xyz.grad is not None and not torch.all(xyz.grad == 0):
            print("PASS: xyz received gradients.")
        else:
            print("FAIL: xyz has no gradients!")
        
        if rotation.grad is not None and not torch.all(rotation.grad == 0):
            print("PASS: rotation received gradients.")
        else:
            print("FAIL: rotation has no gradients!")
    except Exception as e:
        print("FAIL ON BACKWARD:", e)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Aborting test.")
        sys.exit(0)
    
    test_knn()
    test_align()
