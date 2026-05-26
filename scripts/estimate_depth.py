#!/usr/bin/env python3
"""
SpatialCortex — Monocular depth estimation for COLMAP-registered frames.

Runs Depth Anything V2 (metric-indoor) on each registered frame, then
calibrates the metric scale per-frame using COLMAP sparse 3D points as
ground-truth depth anchors.

Without calibration Depth Anything V2's metric model is already in metres,
but a per-frame scale correction accounts for any systematic drift in the
predicted depth.

Output:
    <output>/frame_00042_depth.npy   ← float32 (H, W), metres, same resolution
                                        as the source image
    <output>/scale_log.json          ← per-frame scale factors for debugging

Usage:
    conda activate gsam2
    python scripts/estimate_depth.py \\
        --images  data/colmap/images \\
        --colmap  data/colmap/sparse/0 \\
        --output  data/depth

Dependencies (already in gsam2 env):
    pip install transformers torch torchvision pillow tqdm numpy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ── COLMAP parsers ────────────────────────────────────────────────────────────

def read_cameras(cameras_txt: Path) -> dict:
    """Returns {camera_id: {'fx','fy','cx','cy','w','h'}}"""
    cameras = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cid   = int(parts[0])
            model = parts[1]
            w, h  = int(parts[2]), int(parts[3])
            if model == "PINHOLE":
                fx, fy, cx, cy = map(float, parts[4:8])
            else:
                # fallback: treat first two params as focal
                fx = fy = float(parts[4])
                cx, cy  = w / 2.0, h / 2.0
            cameras[cid] = dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
    return cameras


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def read_images_with_poses(images_txt: Path) -> dict:
    """Returns {image_name: {'R': (3,3), 't': (3,), 'camera_id': int,
                              'obs': [(u,v,pid), ...]}}"""
    result = {}
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]

    i = 0
    while i < len(lines) - 1:
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz      = map(float, parts[5:8])
        cam_id          = int(parts[8])
        name            = parts[9]

        R = quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])

        obs_parts = lines[i + 1].split()
        obs = []
        for j in range(0, len(obs_parts) - 2, 3):
            u, v = float(obs_parts[j]), float(obs_parts[j+1])
            pid  = int(obs_parts[j+2])
            if pid != -1:
                obs.append((u, v, pid))

        result[name] = dict(R=R, t=t, camera_id=cam_id, obs=obs)
        i += 2

    return result


def read_points3d(points_txt: Path) -> dict:
    """Returns {point_id: np.array([x, y, z])}"""
    pts = {}
    with open(points_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pid   = int(parts[0])
            pts[pid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    return pts


# ── Depth model ───────────────────────────────────────────────────────────────

def load_depth_model(device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    print(f"Loading {model_id} …")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model     = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def predict_depth(processor, model, img_pil: Image.Image,
                  target_h: int, target_w: int, device: str) -> np.ndarray:
    """Returns depth in metres, resized to (target_h, target_w)."""
    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # predicted_depth: (1, H', W')
    depth = outputs.predicted_depth          # metres
    depth = F.interpolate(
        depth.unsqueeze(1),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]                                  # (H, W)

    return depth.cpu().numpy().astype(np.float32)


# ── Scale calibration using COLMAP sparse points ──────────────────────────────

def colmap_scale(depth_pred: np.ndarray,
                 frame_info: dict,
                 points3d: dict,
                 cam: dict) -> float:
    """
    Compute per-frame scale = median(colmap_depth / predicted_depth) over
    all visible COLMAP 3D points in this frame.

    If fewer than 5 anchor points are found, returns 1.0 (no correction).
    """
    R, t = frame_info["R"], frame_info["t"]
    H, W = depth_pred.shape
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    ratios = []
    for (u_obs, v_obs, pid) in frame_info["obs"]:
        if pid not in points3d:
            continue

        # COLMAP depth of this 3D point in camera space
        X_world = points3d[pid]
        X_cam   = R @ X_world + t
        d_colmap = X_cam[2]
        if d_colmap <= 0:
            continue

        # Predicted depth at the observed pixel location
        ui, vi = int(round(u_obs)), int(round(v_obs))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        d_pred = float(depth_pred[vi, ui])
        if d_pred <= 0.01:
            continue

        ratios.append(d_colmap / d_pred)

    if len(ratios) < 5:
        return 1.0

    return float(np.median(ratios))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-frame metric depth, calibrated with COLMAP."
    )
    parser.add_argument("--images",  required=True, help="COLMAP images directory")
    parser.add_argument("--colmap",  required=True, help="sparse/0/ directory")
    parser.add_argument("--output",  required=True, help="Output directory for .npy depth maps")
    parser.add_argument("--device",  default="cuda")
    args = parser.parse_args()

    sparse_dir = Path(args.colmap)
    img_dir    = Path(args.images)
    out_dir    = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading COLMAP data …")
    cameras  = read_cameras(sparse_dir / "cameras.txt")
    poses    = read_images_with_poses(sparse_dir / "images.txt")
    points3d = read_points3d(sparse_dir / "points3D.txt")
    print(f"  {len(poses)} poses, {len(points3d):,} 3D points")

    processor, model = load_depth_model(args.device)

    scale_log = {}
    img_files  = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.name in poses
    )
    print(f"Estimating depth for {len(img_files)} frames …")

    for img_path in tqdm(img_files):
        img_pil = Image.open(img_path).convert("RGB")
        W, H    = img_pil.size

        depth = predict_depth(processor, model, img_pil, H, W, args.device)

        # Scale-calibrate with COLMAP sparse points
        frame_info = poses[img_path.name]
        cam        = cameras[frame_info["camera_id"]]
        scale      = colmap_scale(depth, frame_info, points3d, cam)
        depth      = depth * scale

        np.save(out_dir / f"{img_path.stem}_depth.npy", depth)
        scale_log[img_path.name] = round(scale, 5)

    (out_dir / "scale_log.json").write_text(json.dumps(scale_log, indent=2))

    scales = list(scale_log.values())
    print(f"\nDone.  Depth maps → {out_dir}/")
    print(f"Scale factors — median: {np.median(scales):.3f}  "
          f"range: [{min(scales):.3f}, {max(scales):.3f}]")


if __name__ == "__main__":
    main()
