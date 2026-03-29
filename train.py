#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import os
import torch
import sys
import uuid
import yaml
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from random import randint

# Impoart internal backbone natively (now local to 3D-mapping)
from scene import Scene, GaussianModel
from gaussian_renderer import render, network_gui
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.image_utils import psnr, render_net_image
from arguments import ModelParams, PipelineParams, OptimizationParams

# Try import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Import 3D-mapping new modules
from losses.alignment_loss import compute_alignment_loss
from utils.knn_graph import build_knn_graph
from utils.normal_utils import orientation_to_normal
from utils.schedules import schedule_lambda2

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def training(dataset, opt, pipe, cfg, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, eval_render, eval_geometry, dtu_gt_dir, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_align_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Extract our new alignment configurations
    align_cfg = cfg.get('align', {})
    enable_align = align_cfg.get('enable', False) and cfg.get('mode', 'baseline') == 'ours'
    knn_k = align_cfg.get('knn_k', 10)
    knn_chunk_size = align_cfg.get('knn_chunk_size', 4096)
    sigma_d = align_cfg.get('sigma_d', 0.01)
    beta1 = align_cfg.get('beta1', 1.0)
    beta2 = align_cfg.get('beta2', 0.1)
    lambda2_max = align_cfg.get('lambda2_max', 0.5)
    warmup_iters = align_cfg.get('warmup_iters', 3000)
    ramp_iters = align_cfg.get('ramp_iters', 4000)

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 1. 2DGS Backbone Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        # 2. 2DGS Photometric Loss
        Ll1 = l1_loss(image, gt_image)
        photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 3. 2DGS Geometry Regularizations
        lambda_normal_reg = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist_reg = opt.lambda_dist if iteration > 3000 else 0.0

        if "rend_dist" in render_pkg:
            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            geo_normal_loss = lambda_normal_reg * (normal_error).mean()
            geo_dist_loss = lambda_dist_reg * (rend_dist).mean()
            geo_loss = geo_normal_loss + geo_dist_loss
        else:
            geo_loss = torch.tensor(0.0, device="cuda")

        # 4. Our Alignment Loss
        align_loss = torch.tensor(0.0, device="cuda")
        lambda2 = 0.0

        if enable_align:
            lambda2 = schedule_lambda2(iteration, warmup_iters, ramp_iters, lambda2_max)
            
            if lambda2 > 0:
                xyz = gaussians.get_xyz
                rotation = gaussians.get_rotation
                
                # knn graph should be detached to prevent gradient flow to neighbor search
                knn_idx = build_knn_graph(xyz.detach(), knn_k, chunk_size=knn_chunk_size)
                normals = orientation_to_normal(rotation)
                
                align_loss, l_plane, l_normal_align = compute_alignment_loss(
                    xyz, normals, knn_idx, sigma_d, beta1, beta2
                )

        # 5. Total Loss
        total_loss = photo_loss + geo_loss + lambda2 * align_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Logging tracking
            ema_loss_for_log = 0.4 * photo_loss.item() + 0.6 * ema_loss_for_log
            if "rend_dist" in render_pkg:
                ema_dist_for_log = 0.4 * geo_dist_loss.item() + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * geo_normal_loss.item() + 0.6 * ema_normal_for_log
            
            if enable_align and lambda2 > 0:
                ema_align_for_log = 0.4 * align_loss.item() + 0.6 * ema_align_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Tot loss": f"{total_loss.item():.{5}f}",
                    "Photo": f"{ema_loss_for_log:.{5}f}",
                    "Align": f"{ema_align_for_log:.{5}f}",
                    "Lam2": f"{lambda2:.{4}f}",
                    "Pts": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

            # Tensorboard
            if tb_writer is not None:
                tb_writer.add_scalar('train/total_loss', total_loss.item(), iteration)
                tb_writer.add_scalar('train/photo_loss', photo_loss.item(), iteration)
                if "rend_dist" in render_pkg:
                    tb_writer.add_scalar('train/geo_dist_loss', geo_dist_loss.item(), iteration)
                    tb_writer.add_scalar('train/geo_normal_loss', geo_normal_loss.item(), iteration)
                if enable_align:
                    tb_writer.add_scalar('train/align_loss', align_loss.item(), iteration)
                    tb_writer.add_scalar('train/lambda2', lambda2, iteration)

            # Densification / Checkpointing / Reports
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # Post-training Evaluation
    if eval_render:
        try:
            from tools.eval_render_metrics import evaluate_render_metrics
            evaluate_render_metrics(args.model_path, scene, pipe, gaussians, background)
        except Exception as e:
            print(f"Render Evaluation failed: {e}")

    if eval_geometry:
        try:
            from tools.eval_geometry_metrics import evaluate_geometry_metrics
            evaluate_geometry_metrics(args.model_path, gaussians, dtu_gt_dir)
        except Exception as e:
            print(f"Geometry Evaluation failed: {e}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    return tb_writer

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")

    # Evaluation Arguments
    parser.add_argument("--eval_render", action="store_true", help="Evaluate PSNR/SSIM/LPIPS after training")
    parser.add_argument("--eval_geometry", action="store_true", help="Evaluate Chamfer/F-score after training")
    parser.add_argument("--dtu_gt_dir", type=str, default=None, help="Path to DTU GT dir for geometry eval")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    cfg = load_config(args.config)
    print("Project Mode:", cfg.get('mode', 'baseline'))
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), cfg, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.eval_render, args.eval_geometry, args.dtu_gt_dir, args)
    print("\nTraining complete.")
