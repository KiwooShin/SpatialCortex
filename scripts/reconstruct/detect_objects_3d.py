#!/usr/bin/env python3
"""
SpatialCortex — end-to-end 3D object detection pipeline.

For each COLMAP-registered frame, in a single pass:
  1. Grounded-SAM 2  — open-vocabulary 2D detection + instance segmentation
  2. Depth Anything V2  — metric indoor depth, scale-calibrated with COLMAP
  3. Depth unprojection  — mask pixels → world-space 3D AABB
  4. Cuboid projection  — 3D box corners → image, drawn with transparent fill

Output:
    data/detections_3d.json      ← [{object_id, label, bbox_3d, …}]
    data/visualizations/*.jpg    ← frames annotated with projected 3D cuboids

Usage:
    conda activate gsam2
    python scripts/detect_objects_3d.py \\
        --colmap  data/colmap \\
        --output  data/detections_3d.json \\
        --vis-dir data/visualizations

    # Optional: save intermediate masks and depth maps for debugging
        --save-masks  data/detections/masks \\
        --save-depth  data/depth

Setup (one-time):
    conda create -n gsam2 python=3.10 -y && conda activate gsam2
    pip install torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128
    git clone https://github.com/IDEA-Research/Grounded-SAM-2 ~/Grounded-SAM-2
    cd ~/Grounded-SAM-2 && pip install -e .
    pip install transformers accelerate pillow tqdm numpy opencv-python

    # Grounding DINO is loaded via HuggingFace (no CUDA compilation needed)
    # SAM 2 weights
    wget -P ~/Grounded-SAM-2/checkpoints/ \\
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ── Kitchen object prompt ─────────────────────────────────────────────────────

KITCHEN_PROMPT = (
    "dish . toaster . sink . faucet . stove . cooker . mug . bowl . cup . "
    "bottle . knife . pot . pan . plate . glass . spoon . fork . kettle . "
    "microwave . refrigerator . cabinet . cutting board"
)
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.25

# ── Per-class colour palette (RGB) ────────────────────────────────────────────

PALETTE_RGB = [
    (255, 100,  80), ( 80, 200, 120), ( 80, 150, 255), (255, 195,  60),
    (190,  90, 255), ( 60, 220, 215), (255, 130, 190), (160, 255, 100),
    (255, 160,  60), (120, 185, 255), (255, 230, 100), (100, 220, 180),
]

def class_color_bgr(label: str) -> tuple:
    r, g, b = PALETTE_RGB[hash(label) % len(PALETTE_RGB)]
    return (b, g, r)


# ── COLMAP parsers ────────────────────────────────────────────────────────────

def read_cameras(cameras_txt: Path) -> dict:
    cameras = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cid, model = int(parts[0]), parts[1]
            w, h = int(parts[2]), int(parts[3])
            fx, fy = (float(parts[4]), float(parts[5])) if model == "PINHOLE" \
                     else (float(parts[4]), float(parts[4]))
            cx = float(parts[6]) if model == "PINHOLE" else w / 2.0
            cy = float(parts[7]) if model == "PINHOLE" else h / 2.0
            cameras[cid] = dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
    return cameras


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def read_poses(images_txt: Path) -> dict:
    """Returns {frame_name: {R, t, camera_id, obs: [(u,v,pid)]}}"""
    result = {}
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines) - 1:
        parts   = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz      = map(float, parts[5:8])
        cam_id, name    = int(parts[8]), parts[9]
        obs_parts = lines[i + 1].split()
        obs = [
            (float(obs_parts[j]), float(obs_parts[j+1]), int(obs_parts[j+2]))
            for j in range(0, len(obs_parts)-2, 3)
            if int(obs_parts[j+2]) != -1
        ]
        result[name] = dict(R=quat_to_rot(qw,qx,qy,qz), t=np.array([tx,ty,tz]),
                            camera_id=cam_id, obs=obs)
        i += 2
    return result


def read_points3d(points_txt: Path) -> dict:
    pts = {}
    with open(points_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            pts[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return pts


def make_K(cam: dict) -> np.ndarray:
    return np.array([[cam["fx"], 0, cam["cx"]],
                     [0, cam["fy"], cam["cy"]],
                     [0,         0,          1]])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_gdino(device: str):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id  = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def load_sam2(sam2_config: str, sam2_checkpoint: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(sam2_config, sam2_checkpoint, device=device))
    return predictor


def load_depth_model(device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id  = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model     = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


# ── Per-frame: detection ──────────────────────────────────────────────────────

def detect(gdino_processor, gdino_model, sam2_pred,
           img_pil: Image.Image, device: str,
           box_thresh: float, text_thresh: float,
           prompt: str) -> list:
    """Returns list of {label, confidence, bbox_2d, mask: np.uint8 (H,W)}"""
    W, H = img_pil.size

    # Grounding DINO via HuggingFace transformers (no CUDA compilation needed)
    inputs = gdino_processor(images=img_pil, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[(H, W)],
    )[0]

    if len(results["boxes"]) == 0:
        return []

    boxes_xyxy = results["boxes"].cpu().numpy()      # already in pixel xyxy
    labels     = results["labels"]                   # list of str
    scores     = results["scores"].cpu().numpy()

    # SAM 2 segmentation from GDINO boxes
    sam2_pred.set_image(np.array(img_pil))
    masks, sam_scores, _ = sam2_pred.predict(
        point_coords=None, point_labels=None,
        box=boxes_xyxy, multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks[:, 0]   # (N, H, W)

    detections = []
    for label, score, box, mask in zip(labels, scores, boxes_xyxy, masks):
        detections.append({
            "label":      label or "object",
            "confidence": round(float(score), 4),
            "bbox_2d":    [round(float(v), 1) for v in box],
            "mask":       (mask > 0.5).astype(np.uint8) * 255,
        })
    return detections


# ── Per-frame: depth ──────────────────────────────────────────────────────────

def estimate_depth(processor, depth_model,
                   img_pil: Image.Image, H: int, W: int,
                   device: str) -> np.ndarray:
    inputs = {k: v.to(device) for k, v in
              processor(images=img_pil, return_tensors="pt").items()}
    with torch.no_grad():
        raw = depth_model(**inputs).predicted_depth   # (1, H', W')
    depth = F.interpolate(raw.unsqueeze(1), size=(H, W),
                          mode="bilinear", align_corners=False)[0, 0]
    return depth.cpu().numpy().astype(np.float32)


def calibrate_scale(depth: np.ndarray, pose: dict,
                    points3d: dict, H: int, W: int) -> float:
    """Median ratio of COLMAP depth to predicted depth over visible sparse points."""
    R, t = pose["R"], pose["t"]
    ratios = []
    for u_obs, v_obs, pid in pose["obs"]:
        if pid not in points3d:
            continue
        Xc = R @ points3d[pid] + t
        if Xc[2] <= 0:
            continue
        ui, vi = int(round(u_obs)), int(round(v_obs))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        d_pred = float(depth[vi, ui])
        if d_pred > 0.01:
            ratios.append(Xc[2] / d_pred)
    return float(np.median(ratios)) if len(ratios) >= 5 else 1.0


# ── Per-detection: 3D lifting ─────────────────────────────────────────────────

def unproject_mask(mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
                   n_samples: int = 500) -> np.ndarray | None:
    """
    Unproject mask pixels → camera-space 3D points.

    Returns (N, 3) in camera space (X right, Y down, Z depth).
    Camera space is used for both the OBB and visualization so that
    the projected cuboid aligns with the image axes regardless of how
    the camera is oriented in world space.
    """
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.erode(mask, kernel, iterations=1)

    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    if len(xs) > n_samples:
        idx = np.random.choice(len(xs), n_samples, replace=False)
        xs, ys = xs[idx], ys[idx]

    depths = depth[ys, xs]
    valid  = depths > 0.05
    if valid.sum() < 5:
        return None
    xs, ys, depths = xs[valid], ys[valid], depths[valid]

    lo, hi = np.percentile(depths, 10), np.percentile(depths, 90)
    keep   = (depths >= lo) & (depths <= hi)
    if keep.sum() < 3:
        return None
    xs, ys, depths = xs[keep], ys[keep], depths[keep]

    return np.stack([
        (xs - K[0,2]) / K[0,0] * depths,
        (ys - K[1,2]) / K[1,1] * depths,
        depths,
    ], axis=1)                          # (N, 3) camera space


def cam_to_world(pts_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R.T @ (pts_cam - t).T).T


def obb_corners_cam(pts_cam: np.ndarray) -> np.ndarray:
    """
    Oriented bounding box (OBB) via PCA in camera space.

    PCA finds the three axes of greatest variance in the point cloud,
    giving a tighter, correctly-oriented box compared to an axis-aligned box.
    The result is 8 corners in camera space, in the same index order used
    by FACES / EDGES so draw_cuboid works unchanged.
    """
    centroid   = pts_cam.mean(axis=0)
    centered   = pts_cam - centroid
    _, _, Vt   = np.linalg.svd(centered, full_matrices=False)   # Vt: (3,3)

    # Project points onto PCA axes, find extents
    proj       = centered @ Vt.T                                 # (N, 3)
    mn, mx     = proj.min(0), proj.max(0)
    center_pca = (mn + mx) / 2
    hs         = (mx - mn) / 2                                   # half-sizes

    # 8 corners in PCA frame (same order as FACES/EDGES)
    sx, sy, sz = hs
    corners_pca = np.array([
        [-sx,-sy,-sz], [+sx,-sy,-sz], [+sx,+sy,-sz], [-sx,+sy,-sz],
        [-sx,-sy,+sz], [+sx,-sy,+sz], [+sx,+sy,+sz], [-sx,+sy,+sz],
    ]) + center_pca                                              # (8, 3)

    # Rotate back to camera space
    return corners_pca @ Vt + centroid                          # (8, 3)


def project_corners_cam(corners_cam: np.ndarray, K: np.ndarray,
                        W: int, H: int) -> np.ndarray | None:
    """
    Project camera-space OBB corners → image pixels.
    No R/t needed: corners are already in camera space.
    """
    pts = []
    for Xc in corners_cam:
        if Xc[2] <= 0.01:
            return None
        pts.append([K[0,0] * Xc[0] / Xc[2] + K[0,2],
                    K[1,1] * Xc[1] / Xc[2] + K[1,2]])
    pts = np.array(pts, dtype=np.float32)
    sx  = pts[:,0].max() - pts[:,0].min()
    sy  = pts[:,1].max() - pts[:,1].min()
    if sx < 2 or sy < 2 or sx > W * 3 or sy > H * 3:
        return None
    return pts.astype(np.int32)


def aabb_world(pts_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> dict:
    """World-space AABB stored in JSON for cross-frame merging / navigation."""
    pts_world = cam_to_world(pts_cam, R, t)
    mn, mx    = pts_world.min(0), pts_world.max(0)
    return {"center": ((mn+mx)/2).round(4).tolist(),
            "size":   (mx-mn).round(4).tolist()}


FACES = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def draw_cuboid(img: np.ndarray, corners: np.ndarray,
                color: tuple, label: str, alpha: float = 0.18):
    overlay = img.copy()
    for face in FACES:
        cv2.fillPoly(overlay, [corners[face].reshape(-1,1,2)], color)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    for i, j in EDGES:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, 2, cv2.LINE_AA)
    top = int(np.argmin(corners[:,1]))
    lx, ly = int(corners[top,0]), int(corners[top,1])-10
    cv2.putText(img, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(img, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,         1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: Grounded-SAM 2 + depth estimation + 3D cuboid visualization."
    )
    parser.add_argument("--colmap",  default="data/colmap",
                        help="COLMAP output directory (contains images/ and sparse/0/)")
    parser.add_argument("--output",  default="data/detections_3d.json")
    parser.add_argument("--vis-dir", default="data/visualizations",
                        help="Output directory for annotated frames ('' to disable)")
    parser.add_argument("--save-masks", default="",
                        help="If set, save binary mask PNGs here for debugging")
    parser.add_argument("--save-depth", default="",
                        help="If set, save *_depth.npy files here for debugging")
    parser.add_argument("--sam2-checkpoint",
                        default="~/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config",
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--prompt",        default=KITCHEN_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold",type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--device",        default="cuda")
    args = parser.parse_args()

    colmap_dir = Path(args.colmap)
    sparse_dir = colmap_dir / "sparse" / "0"
    img_dir    = colmap_dir / "images"
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis_dir    = Path(args.vis_dir)  if args.vis_dir    else None
    masks_dir  = Path(args.save_masks) if args.save_masks else None
    depth_dir  = Path(args.save_depth) if args.save_depth else None
    for d in [vis_dir, masks_dir, depth_dir]:
        if d: d.mkdir(parents=True, exist_ok=True)

    # ── COLMAP data ────────────────────────────────────────────────────────
    print("Reading COLMAP data …")
    cameras  = read_cameras(sparse_dir / "cameras.txt")
    poses    = read_poses(sparse_dir / "images.txt")
    points3d = read_points3d(sparse_dir / "points3D.txt")
    print(f"  {len(poses)} registered frames, {len(points3d):,} sparse points")

    # ── Load models ────────────────────────────────────────────────────────
    print("\nLoading Grounding DINO (HuggingFace) …")
    gdino_processor, gdino_model = load_gdino(args.device)
    print("Loading SAM 2 …")
    sam2_pred = load_sam2(
        args.sam2_config,
        os.path.expanduser(args.sam2_checkpoint),
        args.device,
    )
    print("Loading Depth Anything V2 …")
    depth_processor, depth_model = load_depth_model(args.device)

    # ── Per-frame pipeline ─────────────────────────────────────────────────
    img_files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg",".jpeg",".png") and f.name in poses
    )
    print(f"\nProcessing {len(img_files)} frames …\n")

    all_objects, scale_log = [], {}

    for img_path in tqdm(img_files):
        img_pil = Image.open(img_path).convert("RGB")
        W, H    = img_pil.size
        pose    = poses[img_path.name]
        cam     = cameras[pose["camera_id"]]
        K       = make_K(cam)
        R, t    = pose["R"], pose["t"]

        # 1. Grounded-SAM 2: 2D detection + segmentation
        detections = detect(
            gdino_processor, gdino_model, sam2_pred, img_pil,
            args.device, args.box_threshold, args.text_threshold, args.prompt,
        )
        if not detections:
            continue

        # 2. Depth Anything V2 + COLMAP scale calibration
        depth = estimate_depth(depth_processor, depth_model, img_pil, H, W, args.device)
        scale = calibrate_scale(depth, pose, points3d, H, W)
        depth = depth * scale
        scale_log[img_path.name] = round(scale, 5)

        if depth_dir:
            np.save(depth_dir / f"{img_path.stem}_depth.npy", depth)

        # 3. Lift each detection to 3D + draw on image
        img_bgr = cv2.imread(str(img_path)) if vis_dir else None

        for i, det in enumerate(detections):
            label = det["label"]
            mask  = det["mask"]

            if masks_dir:
                safe = label.replace(" ","_")
                Image.fromarray(mask).save(masks_dir / f"{img_path.stem}_{safe}_{i}.png")

            # Unproject to camera space
            pts_cam = unproject_mask(mask, depth, K)
            if pts_cam is None:
                continue

            # World-space AABB for JSON (cross-frame merging, navigation)
            bbox_3d = aabb_world(pts_cam, R, t)

            obj = {
                "object_id":    f"{label.replace(' ','_')}_{img_path.stem}_{i}",
                "label":        label,
                "confidence":   det["confidence"],
                "source_frame": img_path.name,
                "bbox_2d":      det["bbox_2d"],
                "bbox_3d":      bbox_3d,
            }
            all_objects.append(obj)

            # Camera-space OBB → project to image for visualization
            # OBB aligns with the object's actual orientation, not world axes,
            # so the projected cuboid matches what you see in the image.
            if img_bgr is not None:
                corners_cam = obb_corners_cam(pts_cam)
                corners_2d  = project_corners_cam(corners_cam, K, W, H)
                if corners_2d is not None:
                    draw_cuboid(img_bgr, corners_2d, class_color_bgr(label),
                                f"{label} {det['confidence']:.2f}")

        if img_bgr is not None and vis_dir:
            cv2.imwrite(str(vis_dir / img_path.name), img_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── Save outputs ───────────────────────────────────────────────────────
    out_path.write_text(json.dumps(all_objects, indent=2))

    frames_with = len({o["source_frame"] for o in all_objects})
    print(f"\nDone.")
    print(f"  {len(all_objects)} 3D objects across {frames_with} frames → {out_path}")
    if vis_dir:
        print(f"  Visualizations → {vis_dir}/")
    if scale_log:
        scales = list(scale_log.values())
        print(f"  Depth scale — median {np.median(scales):.3f}  "
              f"range [{min(scales):.3f}, {max(scales):.3f}]")


if __name__ == "__main__":
    main()
