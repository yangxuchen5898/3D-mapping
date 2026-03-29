import os
import torch
import numpy as np
import json
import trimesh
from scipy.spatial import cKDTree
import glob
import re

from utils.normal_utils import orientation_to_normal


def _find_gt_point_cloud(dtu_gt_dir):
    """Find GT point cloud robustly for DTU-like layouts."""
    if not dtu_gt_dir:
        return None, []

    checked_locations = []
    raw_path = os.path.normpath(dtu_gt_dir)
    abs_path = os.path.abspath(raw_path)
    checked_locations.append(abs_path)

    def _collect_ply_recursive(folder):
        return sorted(
            glob.glob(os.path.join(folder, "**", "*.ply"), recursive=True)
            + glob.glob(os.path.join(folder, "**", "*.PLY"), recursive=True)
        )

    # 1) Direct file input
    if os.path.isfile(abs_path) and abs_path.lower().endswith(".ply"):
        return abs_path, checked_locations

    # 2) Existing folder input (recursive)
    if os.path.isdir(abs_path):
        cands = _collect_ply_recursive(abs_path)
        if cands:
            return cands[0], checked_locations

        # Common case: user passes .../scanXX/GT but GT dir absent/empty.
        parent = os.path.dirname(abs_path)
        if os.path.isdir(parent):
            checked_locations.append(parent)
            preferred = [
                os.path.join(parent, "points.ply"),
                os.path.join(parent, "points3D.ply"),
                os.path.join(parent, "sparse", "0", "points3D.ply"),
            ]
            for p in preferred:
                if os.path.isfile(p):
                    return p, checked_locations
            parent_recursive = _collect_ply_recursive(parent)
            if parent_recursive:
                return parent_recursive[0], checked_locations

    # 3) Non-existing path: infer scan id and fallback search in DTU root.
    scan_match = re.search(r"scan\d+", abs_path.replace("\\", "/"), flags=re.IGNORECASE)
    if scan_match:
        scan_name = scan_match.group(0)
        scan_lower = scan_name.lower()
        # Assume DTU root is parent of scan folder in the input path.
        scan_prefix = abs_path[: scan_match.start()]
        dtu_root = os.path.normpath(scan_prefix)
        scan_dir = os.path.join(dtu_root, scan_name)
        checked_locations.extend([dtu_root, scan_dir])

        preferred = [
            os.path.join(scan_dir, "points.ply"),
            os.path.join(scan_dir, "points3D.ply"),
            os.path.join(scan_dir, "sparse", "0", "points3D.ply"),
        ]
        for p in preferred:
            if os.path.isfile(p):
                return p, checked_locations

        if os.path.isdir(scan_dir):
            cands = _collect_ply_recursive(scan_dir)
            if cands:
                # Prefer files containing scan name, then any file.
                for c in cands:
                    if scan_lower in c.lower():
                        return c, checked_locations
                return cands[0], checked_locations

        if os.path.isdir(dtu_root):
            all_cands = _collect_ply_recursive(dtu_root)
            for c in all_cands:
                if scan_lower in c.lower():
                    return c, checked_locations

    return None, checked_locations

def evaluate_geometry_metrics(model_path, gaussians, dtu_gt_dir=None):
    print(f"Starting geometry evaluation for {model_path}...")
    metrics_dir = os.path.join(model_path, "metrics")
    geom_dir = os.path.join(model_path, "geometry")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(geom_dir, exist_ok=True)

    print("Extracting Point Cloud for evaluation...")
    with torch.no_grad():
        xyz = gaussians.get_xyz.cpu().numpy()
        opacities = gaussians.get_opacity.cpu().numpy().reshape(-1)
        normals = orientation_to_normal(gaussians.get_rotation).cpu().numpy()
        
        mask = opacities > 0.5
        if np.any(mask):
            filtered_xyz = xyz[mask]
            filtered_normals = normals[mask]
        else:
            # Early iterations can have all opacities <= 0.5. Fallback to top-K opacity points.
            k = min(20000, xyz.shape[0])
            topk_idx = np.argpartition(opacities, -k)[-k:] if k > 0 else np.array([], dtype=np.int64)
            filtered_xyz = xyz[topk_idx] if k > 0 else np.empty((0, 3), dtype=xyz.dtype)
            filtered_normals = normals[topk_idx] if k > 0 else np.empty((0, 3), dtype=normals.dtype)

    if filtered_xyz.shape[0] == 0:
        error_msg = (
            "Predicted point cloud is empty after opacity filtering/fallback. "
            "Geometry evaluation skipped. Try more iterations."
        )
        print(error_msg)
        with open(os.path.join(metrics_dir, "geometry_metrics_error.txt"), 'w', encoding='utf-8') as f:
            f.write(error_msg)
        return None

    pred_pc_path = os.path.join(geom_dir, "reconstructed_pointcloud.ply")
    pred_mesh = trimesh.points.PointCloud(filtered_xyz)
    pred_mesh.export(pred_pc_path)

    gt_pc_path, checked_locations = _find_gt_point_cloud(dtu_gt_dir)
                
    if gt_pc_path is None or not os.path.exists(gt_pc_path):
        checked_text = "\n".join([f"- {p}" for p in checked_locations]) if checked_locations else "- (none)"
        error_msg = (
            "GT point cloud not found! Geometry evaluation cannot be executed.\n"
            "Please ensure the official DTU GT is provided.\n"
            f"Expected a '.ply' file at or inside: '{dtu_gt_dir}'.\n"
            "Checked locations:\n"
            f"{checked_text}"
        )
        print(error_msg)
        with open(os.path.join(metrics_dir, "geometry_metrics_error.txt"), 'w', encoding='utf-8') as f:
            f.write(error_msg)
        return None

    print(f"Comparing against GT from: {gt_pc_path}")
    gt_mesh = trimesh.load(gt_pc_path)
    
    gt_normals = None
    if isinstance(gt_mesh, trimesh.Trimesh):
        gt_points = gt_mesh.vertices
        if hasattr(gt_mesh, 'vertex_normals') and len(gt_mesh.vertex_normals) > 0:
            gt_normals = gt_mesh.vertex_normals
    else:
        gt_points = gt_mesh.vertices if hasattr(gt_mesh, 'vertices') else np.array(gt_mesh)
        if hasattr(gt_mesh, 'vertex_normals') and len(gt_mesh.vertex_normals) > 0:
            gt_normals = np.array(gt_mesh.vertex_normals)
    
    if gt_normals is None or len(gt_normals) == 0:
        try:
            import open3d as o3d
            print("GT has no normals. Estimating normals via open3d...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(gt_points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)
            gt_normals = np.asarray(pcd.normals)
        except ImportError:
            print("Warning: open3d not installed and GT missing normals. Normal Consistency metric will be skipped.")
            gt_normals = None

    if gt_points is None or len(gt_points) == 0:
        error_msg = f"GT point cloud is empty: {gt_pc_path}"
        print(error_msg)
        with open(os.path.join(metrics_dir, "geometry_metrics_error.txt"), 'w', encoding='utf-8') as f:
            f.write(error_msg)
        return None

    gt_kdtree = cKDTree(gt_points)
    pred_kdtree = cKDTree(filtered_xyz)

    dist_pred_to_gt, idx_pred_to_gt = gt_kdtree.query(filtered_xyz)
    dist_gt_to_pred, idx_gt_to_pred = pred_kdtree.query(gt_points)

    mean_acc = np.mean(dist_pred_to_gt)
    mean_comp = np.mean(dist_gt_to_pred)
    chamfer_dist = (mean_acc + mean_comp) / 2.0

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

    normal_consistency = -1.0
    if gt_normals is not None and len(gt_normals) > 0:
        normal_acc = np.abs(np.sum(filtered_normals * gt_normals[idx_pred_to_gt], axis=1)).mean()
        normal_comp = np.abs(np.sum(gt_normals * filtered_normals[idx_gt_to_pred], axis=1)).mean()
        normal_consistency = (normal_acc + normal_comp) / 2.0

    metrics_dict = {
        "Chamfer_Distance": float(chamfer_dist),
        "Mean_Accuracy": float(mean_acc),
        "Mean_Completeness": float(mean_comp),
        "Normal_Consistency": float(normal_consistency),
        "F_Scores": {k: float(v) for k, v in f_scores.items()}
    }

    with open(os.path.join(metrics_dir, "geometry_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)

    with open(os.path.join(metrics_dir, "geometry_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Chamfer Distance: {chamfer_dist:.6f}\n")
        f.write(f"Accuracy: {mean_acc:.6f}\n")
        f.write(f"Completeness: {mean_comp:.6f}\n")
        for k, v in f_scores.items():
            f.write(f"{k}: {v:.6f}\n")
        if normal_consistency != -1.0:
            f.write(f"Normal Consistency: {normal_consistency:.6f}\n")
        else:
            f.write("Normal Consistency: N/A\n")

    print(f"\nGeometry Evaluation Completed. Results saved to {metrics_dir}")
    print(f"Chamfer Dist: {chamfer_dist:.6f}, Normal Consistency: {normal_consistency if normal_consistency != -1.0 else 'N/A'}, F-score(1.0): {f_scores['F-score_1.0']:.4f}\n")
    return metrics_dict
