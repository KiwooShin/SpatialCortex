#!/usr/bin/env python3
"""
SpatialCortex — Lift 2D detections to 3D and render projected cuboids.

For each COLMAP-registered frame:
  1. Loads 2D detections (from run_gsam2.py) and depth map (from estimate_depth.py).
  2. Samples pixels within each SAM 2 mask, unprojects them to world space using
     the COLMAP camera pose and the calibrated depth map.
  3. Computes an axis-aligned 3D bounding box (AABB) in world space.
  4. Projects the 8 corners of each AABB back into the image and draws a
     translucent 3D cuboid with per-class colour and a text label.

Output:
    data/detections_3d.json          ← [{object_id, label, bbox_3d, …}, …]
    data/visualizations/
    └── frame_00042.jpg              ← original frame + projected 3D cuboids

Usage:
    conda activate gsam2
    python scripts/lift_to_3d.py \\
        --detections data/detections \\
        --depth      data/depth \\
        --colmap     data/colmap/sparse/0 \\
        --images     data/colmap/images \\
        --output     data/detections_3d.json \\
        --vis-dir    data/visualizations

    # Skip saving visualisations:
        --no-vis
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ── Colour palette (RGB) — one per detected class ─────────────────────────────

PALETTE_RGB = [
    (255, 100,  80),   # red
    ( 80, 200, 120),   # green
    ( 80, 150, 255),   # blue
    (255, 195,  60),   # amber
    (190,  90, 255),   # purple
    ( 60, 220, 215),   # cyan
    (255, 130, 190),   # pink
    (160, 255, 100),   # lime
    (255, 160,  60),   # orange
    (120, 185, 255),   # sky
    (255, 230, 100),   # yellow
    (100, 220, 180),   # teal
]


def class_color(label: str) -> tuple:
    """Deterministic BGR colour for a given label string."""
    r, g, b = PALETTE_RGB[hash(label) % len(PALETTE_RGB)]
    return (b, g, r)   # OpenCV uses BGR


# ── COLMAP parsers ────────────────────────────────────────────────────────────

def read_cameras(cameras_txt: Path) -> dict:
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
                fx = fy = float(parts[4])
                cx, cy = w / 2.0, h / 2.0
            cameras[cid] = dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
    return cameras


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def read_images_with_poses(images_txt: Path) -> dict:
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
        result[name]    = dict(
            R=quat_to_rot(qw, qx, qy, qz),
            t=np.array([tx, ty, tz]),
            camera_id=cam_id,
        )
        i += 2
    return result


# ── Unproject: 2D mask → world-space 3D point cloud ──────────────────────────

def unproject_mask(mask: np.ndarray, depth: np.ndarray,
                   R: np.ndarray, t: np.ndarray, K: np.ndarray,
                   n_samples: int = 500) -> np.ndarray | None:
    """
    Sample up to n_samples pixels inside mask, unproject using depth, return
    world-space points (N, 3).  Returns None if not enough valid points.
    """
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None

    # Random subsample for speed
    if len(xs) > n_samples:
        idx = np.random.choice(len(xs), n_samples, replace=False)
        xs, ys = xs[idx], ys[idx]

    depths = depth[ys, xs]
    valid  = depths > 0.05              # discard zero / near-zero depth
    if valid.sum() < 5:
        return None
    xs, ys, depths = xs[valid], ys[valid], depths[valid]

    # Remove depth outliers (top/bottom 10 %)
    lo, hi = np.percentile(depths, 10), np.percentile(depths, 90)
    keep   = (depths >= lo) & (depths <= hi)
    if keep.sum() < 3:
        return None
    xs, ys, depths = xs[keep], ys[keep], depths[keep]

    # Back-project to camera space
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X_cam = np.stack([
        (xs - cx) / fx * depths,
        (ys - cy) / fy * depths,
        depths,
    ], axis=1)                          # (N, 3)

    # Camera → world  (R, t define world → camera)
    X_world = (R.T @ (X_cam - t).T).T  # (N, 3)
    return X_world


def aabb_from_points(pts: np.ndarray) -> dict:
    """Axis-aligned bounding box from a point cloud."""
    mn  = pts.min(axis=0)
    mx  = pts.max(axis=0)
    center = ((mn + mx) / 2).tolist()
    size   = (mx - mn).tolist()
    return {"center": [round(v, 4) for v in center],
            "size":   [round(v, 4) for v in size]}


# ── Project: world-space AABB corners → 2D image pixels ──────────────────────

def box_corners(center, size) -> np.ndarray:
    """Return the 8 corners of an axis-aligned box. Shape (8, 3)."""
    cx, cy, cz = center
    sx, sy, sz = [s / 2 for s in size]
    return np.array([
        [cx-sx, cy-sy, cz-sz],   # 0 bottom-front-left
        [cx+sx, cy-sy, cz-sz],   # 1 bottom-front-right
        [cx+sx, cy+sy, cz-sz],   # 2 bottom-back-right
        [cx-sx, cy+sy, cz-sz],   # 3 bottom-back-left
        [cx-sx, cy-sy, cz+sz],   # 4 top-front-left
        [cx+sx, cy-sy, cz+sz],   # 5 top-front-right
        [cx+sx, cy+sy, cz+sz],   # 6 top-back-right
        [cx-sx, cy+sy, cz+sz],   # 7 top-back-left
    ])


def project_corners(corners_world: np.ndarray,
                    R: np.ndarray, t: np.ndarray, K: np.ndarray,
                    img_w: int, img_h: int) -> np.ndarray | None:
    """
    Project 8 world-space corners into image.
    Returns (8, 2) int32 pixel coords, or None if any corner is behind camera.
    """
    pts_2d = []
    for Xw in corners_world:
        Xc = R @ Xw + t
        if Xc[2] <= 0.01:
            return None             # corner behind camera — skip this box
        x = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
        y = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
        pts_2d.append([x, y])

    pts = np.array(pts_2d, dtype=np.float32)

    # Sanity check: projected box should have reasonable screen extent
    span_x = pts[:, 0].max() - pts[:, 0].min()
    span_y = pts[:, 1].max() - pts[:, 1].min()
    if span_x < 2 or span_y < 2 or span_x > img_w * 3 or span_y > img_h * 3:
        return None

    return pts.astype(np.int32)


# ── Visualisation: draw 3D cuboid onto frame ──────────────────────────────────

FACES = [
    [0, 1, 2, 3],   # bottom
    [4, 5, 6, 7],   # top
    [0, 1, 5, 4],   # front
    [2, 3, 7, 6],   # back
    [0, 3, 7, 4],   # left
    [1, 2, 6, 5],   # right
]

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom ring
    (4, 5), (5, 6), (6, 7), (7, 4),   # top ring
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def draw_cuboid(img: np.ndarray, corners: np.ndarray,
                color_bgr: tuple, label: str,
                face_alpha: float = 0.18) -> np.ndarray:
    """
    Draw a projected 3D cuboid onto img (in-place) with transparent face fill,
    solid edges, and a text label above the topmost projected corner.
    """
    overlay = img.copy()

    # Filled faces — drawn onto overlay, then blended
    for face in FACES:
        pts = corners[face].reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color_bgr)

    cv2.addWeighted(overlay, face_alpha, img, 1.0 - face_alpha, 0, img)

    # Solid edges on top of blend
    for i, j in EDGES:
        cv2.line(img,
                 (int(corners[i, 0]), int(corners[i, 1])),
                 (int(corners[j, 0]), int(corners[j, 1])),
                 color_bgr, 2, cv2.LINE_AA)

    # Label: white outline + coloured text
    top_idx  = int(np.argmin(corners[:, 1]))
    lx, ly   = int(corners[top_idx, 0]), int(corners[top_idx, 1]) - 10
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.65
    cv2.putText(img, label, (lx, ly), font, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly), font, scale, color_bgr,       1, cv2.LINE_AA)

    return img


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_frame(frame_name: str,
                  detections: list,
                  depth: np.ndarray,
                  pose: dict,
                  K: np.ndarray,
                  img_path: Path,
                  vis_dir: Path | None) -> list:
    R, t = pose["R"], pose["t"]
    H, W = depth.shape
    results = []

    img = cv2.imread(str(img_path)) if vis_dir else None

    for i, det in enumerate(detections):
        label     = det["label"]
        mask_path = Path(det["mask_path"])
        if not mask_path.exists():
            mask_path = img_path.parent.parent / det["mask_path"]
        if not mask_path.exists():
            continue

        mask = np.array(__import__("PIL").Image.open(mask_path).convert("L"))

        # Erode mask by 3 px to avoid background bleed at object boundaries
        kernel = np.ones((3, 3), np.uint8)
        mask   = cv2.erode(mask, kernel, iterations=1)

        pts_world = unproject_mask(mask, depth, R, t, K)
        if pts_world is None:
            continue

        bbox_3d = aabb_from_points(pts_world)

        obj_id = f"{label.replace(' ','_')}_{frame_name}_{i}"
        results.append({
            "object_id":    obj_id,
            "label":        label,
            "confidence":   det["confidence"],
            "source_frame": frame_name,
            "bbox_2d":      det["bbox_2d"],
            "mask_path":    det["mask_path"],
            "bbox_3d":      bbox_3d,
        })

        # Draw projected cuboid onto the frame
        if img is not None:
            size = bbox_3d["size"]
            if all(s > 0.01 for s in size):   # skip degenerate boxes
                corners_w = box_corners(bbox_3d["center"], size)
                corners_2d = project_corners(corners_w, R, t, K, W, H)
                if corners_2d is not None:
                    color = class_color(label)
                    draw_cuboid(img, corners_2d, color,
                                f"{label} {det['confidence']:.2f}")

    if img is not None and vis_dir is not None:
        out_path = vis_dir / frame_name
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Lift 2D detections to 3D AABBs and render projected cuboids."
    )
    parser.add_argument("--detections", required=True,
                        help="Directory from run_gsam2.py (contains all_detections.json)")
    parser.add_argument("--depth",      required=True,
                        help="Directory of *_depth.npy files from estimate_depth.py")
    parser.add_argument("--colmap",     required=True,
                        help="sparse/0/ directory")
    parser.add_argument("--images",     required=True,
                        help="COLMAP images directory")
    parser.add_argument("--output",     default="data/detections_3d.json")
    parser.add_argument("--vis-dir",    default="data/visualizations",
                        help="Output directory for annotated frames")
    parser.add_argument("--no-vis",     action="store_true",
                        help="Skip generating visualisation frames")
    args = parser.parse_args()

    det_dir    = Path(args.detections)
    depth_dir  = Path(args.depth)
    sparse_dir = Path(args.colmap)
    img_dir    = Path(args.images)
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis_dir = None
    if not args.no_vis:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    print("Reading COLMAP data …")
    cameras = read_cameras(sparse_dir / "cameras.txt")
    poses   = read_images_with_poses(sparse_dir / "images.txt")

    # Build a single K matrix (all frames share one camera in this dataset)
    cam = cameras[next(iter(cameras))]
    K   = np.array([
        [cam["fx"],      0, cam["cx"]],
        [     0, cam["fy"], cam["cy"]],
        [     0,      0,       1    ],
    ])
    print(f"  K = fx={cam['fx']:.1f}  fy={cam['fy']:.1f}  "
          f"cx={cam['cx']:.1f}  cy={cam['cy']:.1f}")

    all_dets_path = det_dir / "all_detections.json"
    all_dets      = json.loads(all_dets_path.read_text())
    print(f"  Loaded detections for {len(all_dets)} frames")

    all_objects = []

    for frame_name, detections in tqdm(all_dets.items()):
        if not detections:
            continue
        if frame_name not in poses:
            continue

        depth_path = depth_dir / f"{Path(frame_name).stem}_depth.npy"
        if not depth_path.exists():
            continue

        img_path = img_dir / frame_name
        if not img_path.exists():
            continue

        depth = np.load(depth_path)
        pose  = poses[frame_name]
        cam_k = cameras[pose["camera_id"]]
        K_frame = np.array([
            [cam_k["fx"],         0, cam_k["cx"]],
            [          0, cam_k["fy"], cam_k["cy"]],
            [          0,         0,          1  ],
        ])

        objs = process_frame(
            frame_name, detections, depth, pose, K_frame,
            img_path, vis_dir,
        )
        all_objects.extend(objs)

    out_path.write_text(json.dumps(all_objects, indent=2))
    print(f"\nDone.  {len(all_objects)} 3D objects → {out_path}")
    if vis_dir:
        n_vis = len(list(vis_dir.iterdir()))
        print(f"       {n_vis} visualisation frames → {vis_dir}/")


if __name__ == "__main__":
    main()
