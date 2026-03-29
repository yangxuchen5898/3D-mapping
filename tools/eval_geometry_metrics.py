import os
import torch
import numpy as np
import json
import trimesh
from scipy.spatial import cKDTree

def evaluate_geometry_metrics(model_path, gaussians, dtu_gt_dir=None):
    print(f"Starting geometry evaluation for {model_path}...")
    metrics_dir = os.path.join(model_path, "metrics")
    geom_dir = os.path.join(model_path, "geometry")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(geom_dir, exist_ok=True)

    # 1. "Extract" the simplest geometry (just cleaning and exporting Point Cloud)
    # We filter out highly transparent points. A real mesh extraction requires TSDF/Poisson.
    print("Extracting Point Cloud for evaluation...")
    with torch.no_grad():
        xyz = gaussians.get_xyz.cpu().numpy()
        opacities = gaussians.get_opacity.cpu().numpy().squeeze()
        # Keep points with opacity > 0.5 (a basic threshold)
        mask = opacities > 0.5
        filtered_xyz = xyz[mask]

    pred_pc_path = os.path.join(geom_dir, "reconstructed_pointcloud.ply")
    pred_mesh = trimesh.points.PointCloud(filtered_xyz)
    pred_mesh.export(pred_pc_path)

    # 2. Check GT Dependency
    gt_pc_path = None
    if dtu_gt_dir:
        # Default typical name for DTU evaluation GT mesh/pc
        candidate = os.path.join(dtu_gt_dir, "stl105_total.ply")
        if os.path.exists(candidate):
            gt_pc_path = candidate
    
    if gt_pc_path is None or not os.path.exists(gt_pc_path):
        error_msg = (
            "GT point cloud not found! Geometry evaluation cannot be executed.\n"
            "Please ensure the official DTU GT is provided.\n"
            f"Expected location: '{dtu_gt_dir}/stl105_total.ply' (or similar)."
        )
        print(error_msg)
        with open(os.path.join(metrics_dir, "geometry_metrics_error.txt"), 'w') as f:
            f.write(error_msg)
        return

    # 3. Calculate Chamfer Distance and F-Score
    print(f"Comparing against GT from: {gt_pc_path}")
    gt_mesh = trimesh.load(gt_pc_path)
    if isinstance(gt_mesh, trimesh.Trimesh):
        gt_points = gt_mesh.vertices
    else:
        gt_points = gt_mesh.vertices if hasattr(gt_mesh, 'vertices') else np.array(gt_mesh)

    gt_kdtree = cKDTree(gt_points)
    pred_kdtree = cKDTree(filtered_xyz)

    # Distances from pred to GT (Accuracy)
    dist_pred_to_gt, _ = gt_kdtree.query(filtered_xyz)
    # Distances from GT to pred (Completeness)
    dist_gt_to_pred, _ = pred_kdtree.query(gt_points)

    mean_acc = np.mean(dist_pred_to_gt)
    mean_comp = np.mean(dist_gt_to_pred)
    chamfer_dist = (mean_acc + mean_comp) / 2.0

    # F-score thresholds (usually 1mm and 2mm for DTU, adjusting based on native scale)
    thresholds = [1.0, 2.0]
    f_scores = {}
    for th in thresholds:
        precision = np.mean(dist_pred_to_gt < th)
        recall = np.mean(dist_gt_to_pred < th)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f_scores[f"F-score_{th}"] = f1

    # Format Output
    metrics_dict = {
        "Chamfer_Distance": float(chamfer_dist),
        "Mean_Accuracy": float(mean_acc),
        "Mean_Completeness": float(mean_comp),
        "F_Scores": {k: float(v) for k, v in f_scores.items()}
    }

    # Save Results
    with open(os.path.join(metrics_dir, "geometry_metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    with open(os.path.join(metrics_dir, "geometry_metrics.txt"), 'w') as f:
        f.write(f"Chamfer Distance: {chamfer_dist:.6f}\n")
        f.write(f"Accuracy: {mean_acc:.6f}\n")
        f.write(f"Completeness: {mean_comp:.6f}\n")
        for k, v in f_scores.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"\nGeometry Evaluation Completed. Results saved to {metrics_dir}")
    print(f"Chamfer Dist: {chamfer_dist:.6f}, F-score(1.0): {f_scores['F-score_1.0']:.4f}\n")
    return metrics_dict
