import os
import torch
import torchvision
from PIL import Image
import numpy as np
import json
import csv
from tqdm import tqdm

from utils.image_utils import psnr
from utils.loss_utils import ssim

def evaluate_render_metrics(model_path, scene, pipe, gaussians, background):
    print(f"Starting render evaluation for {model_path}...")
    metrics_dir = os.path.join(model_path, "metrics")
    renders_dir = os.path.join(model_path, "renders", "test")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(renders_dir, exist_ok=True)

    try:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    except ImportError:
        print("Warning: lpips not installed. LPIPS metric will be skipped. Run 'pip install lpips' to enable it.")
        loss_fn_vgg = None

    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("No test cameras found. Skipping render evaluation.")
        return

    from gaussian_renderer import render
    
    psnr_list, ssim_list, lpips_list = [], [], []
    per_view_data = []

    with torch.no_grad():
        for idx, view in enumerate(tqdm(test_cameras, desc="Evaluating Setup")):
            render_pkg = render(view, gaussians, pipe, background)
            pred_img = render_pkg["render"].clamp(0.0, 1.0)
            gt_img = view.original_image.cuda().clamp(0.0, 1.0)
            
            # Save predict image
            pred_save = (pred_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_save).save(os.path.join(renders_dir, f"{view.image_name}_pred.png"))
            
            # Calculate metrics
            curr_psnr = psnr(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).item()
            curr_ssim = ssim(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).item()
            curr_lpips = 0.0
            
            if loss_fn_vgg is not None:
                # LPIPS expects inputs in [-1, 1]
                pred_img_lpips = pred_img * 2.0 - 1.0
                gt_img_lpips = gt_img * 2.0 - 1.0
                curr_lpips = loss_fn_vgg(pred_img_lpips.unsqueeze(0), gt_img_lpips.unsqueeze(0)).item()

            psnr_list.append(curr_psnr)
            ssim_list.append(curr_ssim)
            if loss_fn_vgg is not None: lpips_list.append(curr_lpips)

            per_view_data.append({
                "image_name": view.image_name,
                "PSNR": curr_psnr,
                "SSIM": curr_ssim,
                "LPIPS": curr_lpips if loss_fn_vgg is not None else -1
            })

    # Aggregate
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else -1.0

    metrics_dict = {
        "mean_PSNR": mean_psnr,
        "mean_SSIM": mean_ssim,
        "mean_LPIPS": mean_lpips
    }

    # Save to JSON
    with open(os.path.join(metrics_dir, "metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    # Save to TXT
    with open(os.path.join(metrics_dir, "metrics.txt"), 'w') as f:
        f.write(f"PSNR: {mean_psnr:.4f}\n")
        f.write(f"SSIM: {mean_ssim:.4f}\n")
        if mean_lpips != -1.0:
            f.write(f"LPIPS: {mean_lpips:.4f}\n")
            
    # Save to CSV
    csv_file = os.path.join(metrics_dir, "per_view_metrics.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "PSNR", "SSIM", "LPIPS"])
        writer.writeheader()
        for row in per_view_data:
            writer.writerow(row)

    print(f"\nRender Evaluation Completed. Results saved to {metrics_dir}")
    print(f"Mean PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}\n")
    return metrics_dict
