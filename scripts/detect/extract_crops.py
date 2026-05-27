"""
extract_crops.py — extract best-view RGB crop for each fused object in scene_obbs.csv.

For each object:
  1. Find all raw snippet detections of the same class within dist_thresh metres.
  2. Pick the one with highest confidence → gives the best timestamp.
  3. Load the RGB frame from VRS at that timestamp.
  4. Project the 3D OBB corners into the fisheye image (CW-90° corrected).
  5. Crop the axis-aligned bounding rect + padding; save as JPEG.

Outputs:
  - output/crops/{scene_name}/{obj_id:03d}_{class}.jpg   (one per object)
  - {scene_obbs_dir}/scene_obbs_crops.csv                (scene_obbs + crop_path, best_ts_ns columns)
"""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import pandas as pd

from projectaria_tools.core import data_provider, sensor_data

from spatialcortex.config import RGB_SID
from spatialcortex.geometry import obb_corners_world, load_trajectory, interp_pose, project_pt


# ── Crop extraction ───────────────────────────────────────────────────────────

def center_dist_3d(row_a, row_b) -> float:
    return float(np.sqrt(
        (row_a['tx_world_object'] - row_b['tx_world_object'])**2 +
        (row_a['ty_world_object'] - row_b['ty_world_object'])**2 +
        (row_a['tz_world_object'] - row_b['tz_world_object'])**2
    ))


def find_best_snippet(scene_row, snippets: pd.DataFrame, dist_thresh: float):
    """
    Return the single snippet detection closest to this fused object
    (same class, within dist_thresh) with the highest probability.
    Returns a Series or None.
    """
    same_class = snippets[snippets['name'] == scene_row['name']]
    if len(same_class) == 0:
        return None

    dists = np.sqrt(
        (same_class['tx_world_object'] - scene_row['tx_world_object'])**2 +
        (same_class['ty_world_object'] - scene_row['ty_world_object'])**2 +
        (same_class['tz_world_object'] - scene_row['tz_world_object'])**2
    )
    candidates = same_class[dists < dist_thresh]
    if len(candidates) == 0:
        return None
    return candidates.loc[candidates['prob'].idxmax()]


def crop_object(provider, cam_calib, R_dc, t_dc, img_N,
                times_us, Rs_wd, ts_wd,
                snip_row, padding: float):
    """
    Load VRS frame at snip_row['time_ns'], project OBB, crop with padding.
    Returns (rotated_frame, crop) or (None, None).
    """
    ts_ns = int(snip_row['time_ns'])

    img_data, _ = provider.get_image_data_by_time_ns(
        RGB_SID, ts_ns,
        sensor_data.TimeDomain.DEVICE_TIME,
        sensor_data.TimeQueryOptions.CLOSEST,
    )
    if not img_data.is_valid():
        return None, None

    frame_raw = img_data.to_numpy_array().copy()
    if frame_raw.ndim == 2:
        frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_GRAY2BGR)
    elif frame_raw.shape[2] == 1:
        frame_raw = cv2.cvtColor(frame_raw[:, :, 0], cv2.COLOR_GRAY2BGR)
    frame = cv2.rotate(frame_raw, cv2.ROTATE_90_CLOCKWISE)
    H, W = frame.shape[:2]

    R_wd, t_wd = interp_pose(times_us, Rs_wd, ts_wd, ts_ns)
    corners_w = obb_corners_world(
        snip_row['tx_world_object'], snip_row['ty_world_object'],
        snip_row['tz_world_object'],
        snip_row['qw_world_object'], snip_row['qx_world_object'],
        snip_row['qy_world_object'], snip_row['qz_world_object'],
        snip_row['scale_x'], snip_row['scale_y'], snip_row['scale_z'],
    )

    px_pts = [project_pt(c, R_wd, t_wd, R_dc, t_dc, cam_calib, img_N)
              for c in corners_w]
    valid = [p for p in px_pts if p is not None]

    if len(valid) < 4:
        return frame, None   # frame returned so caller can try fallback

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    bw, bh = x1 - x0, y1 - y0
    pad_x = bw * padding
    pad_y = bh * padding
    x0 = max(0, int(x0 - pad_x))
    y0 = max(0, int(y0 - pad_y))
    x1 = min(W - 1, int(x1 + pad_x))
    y1 = min(H - 1, int(y1 + pad_y))

    if x1 <= x0 or y1 <= y0:
        return frame, None

    crop = frame[y0:y1, x0:x1]
    return frame, crop


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene-obbs',   required=True, help='scene_obbs.csv')
    ap.add_argument('--snippet-obbs', required=True, help='snippet_obbs.csv')
    ap.add_argument('--vrs',          required=True, help='main.vrs')
    ap.add_argument('--traj',         required=True, help='closed_loop_trajectory.csv')
    ap.add_argument('--output-dir',   required=True, help='directory for crop JPEGs')
    ap.add_argument('--scene-name',   required=True, help='short scene identifier, e.g. seq01')
    ap.add_argument('--dist-thresh',  type=float, default=0.80,
                    help='max centre distance (m) to match fused→snippet (default 0.80)')
    ap.add_argument('--padding',      type=float, default=0.25,
                    help='fractional padding around crop bbox (default 0.25)')
    ap.add_argument('--min-crop-px',  type=int, default=48,
                    help='discard crops smaller than this in either dimension (default 48)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scene_obbs   = pd.read_csv(args.scene_obbs)
    snippet_obbs = pd.read_csv(args.snippet_obbs)
    print(f"Loaded {len(scene_obbs)} fused objects, {len(snippet_obbs)} snippet detections")

    times_us, Rs_wd, ts_wd = load_trajectory(args.traj)
    print(f"Loaded {len(times_us)} trajectory poses")

    provider   = data_provider.create_vrs_data_provider(args.vrs)
    dev_calib  = provider.get_device_calibration()
    cam_calib  = dev_calib.get_camera_calib("camera-rgb")
    T_dc       = cam_calib.get_transform_device_camera()
    R_dc       = T_dc.rotation().to_matrix()
    t_dc       = T_dc.translation().flatten()
    img_w, _   = cam_calib.get_image_size()
    img_N      = int(img_w)
    print(f"Camera: {img_N}×{img_N}, CW-90° rotation applied")

    # Sort snippet_obbs by prob descending so idxmax is fast
    snippet_obbs = snippet_obbs.sort_values('prob', ascending=False).reset_index(drop=True)

    crop_paths  = []
    best_ts_ns  = []
    saved = 0
    failed = 0

    for obj_id, scene_row in scene_obbs.iterrows():
        best_snip = find_best_snippet(scene_row, snippet_obbs, args.dist_thresh)

        if best_snip is None:
            print(f"  [{obj_id:03d}] {scene_row['name']:20s} — no matching snippet")
            crop_paths.append('')
            best_ts_ns.append(-1)
            failed += 1
            continue

        _, crop = crop_object(
            provider, cam_calib, R_dc, t_dc, img_N,
            times_us, Rs_wd, ts_wd,
            best_snip, args.padding,
        )

        if crop is None or crop.shape[0] < args.min_crop_px or crop.shape[1] < args.min_crop_px:
            print(f"  [{obj_id:03d}] {scene_row['name']:20s} ts={best_snip['time_ns']} — crop too small or no valid projection")
            crop_paths.append('')
            best_ts_ns.append(int(best_snip['time_ns']))
            failed += 1
            continue

        fname = f"{obj_id:03d}_{scene_row['name'].replace(' ', '_')}.jpg"
        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

        crop_paths.append(out_path)
        best_ts_ns.append(int(best_snip['time_ns']))
        saved += 1
        print(f"  [{obj_id:03d}] {scene_row['name']:20s} ts={best_snip['time_ns']}  "
              f"prob={best_snip['prob']:.3f}  crop={crop.shape[1]}×{crop.shape[0]}  → {fname}")

    # Write augmented CSV
    scene_obbs = scene_obbs.copy()
    scene_obbs.insert(0, 'obj_id', range(len(scene_obbs)))
    scene_obbs.insert(1, 'scene', args.scene_name)
    scene_obbs['crop_path'] = crop_paths
    scene_obbs['best_ts_ns'] = best_ts_ns

    out_csv = os.path.join(os.path.dirname(args.scene_obbs), 'scene_obbs_crops.csv')
    scene_obbs.to_csv(out_csv, index=False)

    print(f"\nSaved {saved} crops ({failed} failed) → {args.output_dir}")
    print(f"Augmented CSV → {out_csv}")


if __name__ == '__main__':
    main()
