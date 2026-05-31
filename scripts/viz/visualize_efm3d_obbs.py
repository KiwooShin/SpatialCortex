"""
Visualize EFM3D oriented bounding boxes.

Produces two images per frame (combined side-by-side):
  - LEFT:  fisheye RGB with 3D OBB overlay (correctly oriented + fisheye edge sampling)
  - RIGHT: top-down bird's-eye view of OBBs and camera position

Correctness notes:
  - Aria RGB raw image is 90° CCW from upright → rotate CW 90° before display.
    After CW 90° rotation on square N×N: (u_raw, v_raw) → (N-1-v_raw, u_raw).
  - scale_x/y/z in snippet_obbs.csv are FULL dimensions; half-extents = scale/2.
  - Fisheye edges: sample N points per edge and project each individually
    (straight 3D lines curve under fisheye distortion).
  - World frame: Z is up (gravity-aligned); top-down view looks down -Z, plots XY.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import pandas as pd

from projectaria_tools.core import data_provider, sensor_data

from spatialcortex.config import RGB_SID
from spatialcortex.geometry import obb_corners_world, load_trajectory, interp_pose
from spatialcortex.drawing import get_color, draw_obb


# ── Top-down view ─────────────────────────────────────────────────────────────

def create_topdown(frame_obbs, cam_pos, R_wd, R_dc,
                   all_cam_positions=None, view_size=704, radius_m=6.0):
    """
    Bird's-eye view looking down -Z (world Z is up, plot XY plane).

    cam_pos: (3,) world position of device at this frame
    R_wd:    (3,3) R_world_device rotation
    R_dc:    (3,3) R_device_camera rotation
    all_cam_positions: list of (x,y) world positions for trajectory line
    """
    img = np.full((view_size, view_size, 3), 18, dtype=np.uint8)
    scale = (view_size / 2) / radius_m   # pixels / meter
    ox, oy = view_size // 2, view_size // 2   # origin in image = camera position

    def w2p(wx, wy):
        """World XY → image pixel (camera at center, Y-up)."""
        px = int(ox + (wx - cam_pos[0]) * scale)
        py = int(oy - (wy - cam_pos[1]) * scale)   # flip: world+Y → screen-Y
        return px, py

    # ── Grid (every 1 m) ──────────────────────────────────────────────────────
    for dm in range(-int(radius_m)-1, int(radius_m)+2):
        xw = cam_pos[0] + dm
        yw = cam_pos[1] + dm
        p0 = w2p(xw, cam_pos[1] - radius_m - 1)
        p1 = w2p(xw, cam_pos[1] + radius_m + 1)
        cv2.line(img, p0, p1, (38, 38, 38), 1)
        p0 = w2p(cam_pos[0] - radius_m - 1, yw)
        p1 = w2p(cam_pos[0] + radius_m + 1, yw)
        cv2.line(img, p0, p1, (38, 38, 38), 1)

    # ── Trajectory ────────────────────────────────────────────────────────────
    if all_cam_positions and len(all_cam_positions) > 1:
        pts = [w2p(x, y) for x, y in all_cam_positions]
        for k in range(len(pts) - 1):
            cv2.line(img, pts[k], pts[k+1], (60, 60, 60), 1)

    # ── OBB footprints ────────────────────────────────────────────────────────
    for _, row in frame_obbs.iterrows():
        corners = obb_corners_world(
            row["tx_world_object"], row["ty_world_object"], row["tz_world_object"],
            row["qw_world_object"], row["qx_world_object"],
            row["qy_world_object"], row["qz_world_object"],
            row["scale_x"], row["scale_y"], row["scale_z"],
        )
        color = get_color(row["name"])

        # Use all 8 corners projected to XY → convex hull gives the footprint
        pts_px = np.array([w2p(c[0], c[1]) for c in corners], dtype=np.int32)
        hull = cv2.convexHull(pts_px)

        # Clip check: skip if entirely outside view
        if (pts_px[:,0].max() < 0 or pts_px[:,0].min() >= view_size or
                pts_px[:,1].max() < 0 or pts_px[:,1].min() >= view_size):
            continue

        overlay = img.copy()
        cv2.fillConvexPoly(overlay, hull, color)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.polylines(img, [hull], True, color, 2, cv2.LINE_AA)

        # Short label at object center
        cx_px, cy_px = w2p(row["tx_world_object"], row["ty_world_object"])
        short = row["name"][:5]
        cv2.putText(img, short, (cx_px - 12, cy_px + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # ── Camera: position + optical axis arrow ─────────────────────────────────
    cam_px = (ox, oy)
    cv2.circle(img, cam_px, 7, (0, 255, 100), -1)
    cv2.circle(img, cam_px, 7, (255, 255, 255), 1)

    # Optical axis in world XY plane: R_wd @ R_dc @ [0,0,1]
    optical_world = R_wd @ (R_dc @ np.array([0.0, 0.0, 1.0]))
    opt_xy = optical_world[:2]
    norm = np.linalg.norm(opt_xy)
    if norm > 1e-4:
        opt_xy /= norm
        arrow_end = w2p(cam_pos[0] + opt_xy[0] * 1.5,
                        cam_pos[1] + opt_xy[1] * 1.5)
        cv2.arrowedLine(img, cam_px, arrow_end, (0, 255, 100), 2,
                        cv2.LINE_AA, tipLength=0.25)

    # ── Decorations ───────────────────────────────────────────────────────────
    bar_px = int(scale)
    bx, by = 16, view_size - 22
    cv2.line(img, (bx, by), (bx + bar_px, by), (180, 180, 180), 2)
    cv2.line(img, (bx, by-4), (bx, by+4), (180, 180, 180), 1)
    cv2.line(img, (bx+bar_px, by-4), (bx+bar_px, by+4), (180, 180, 180), 1)
    cv2.putText(img, "1 m", (bx, by - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Axes labels
    ax = w2p(cam_pos[0] + radius_m * 0.85, cam_pos[1])
    ay = w2p(cam_pos[0], cam_pos[1] + radius_m * 0.85)
    cv2.putText(img, "+X", (ax[0]-10, ax[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 220), 1)
    cv2.putText(img, "+Y", (ay[0]+5, ay[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 100), 1)

    cv2.putText(img, "Top-down (world XY)",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obbs", required=True,
                    help="snippet_obbs.csv (per-timestamp) or scene_obbs.csv "
                         "(stable, broadcast to every frame with --scene-mode)")
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scene-mode", action="store_true",
                    help="Treat --obbs as scene-level (no time_ns): show the same "
                         "stable fused objects at every VRS frame.")
    ap.add_argument("--prob-thresh", type=float, default=0.2)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Output image scale for fisheye panel (default 0.5)")
    ap.add_argument("--topdown-radius", type=float, default=6.0,
                    help="Metres to show around camera in top-down view (default 6)")
    ap.add_argument("--n-edge-samples", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load OBBs
    obbs = pd.read_csv(args.obbs)
    obbs = obbs[obbs["prob"] >= args.prob_thresh].copy()

    # Load trajectory
    times_us, Rs_wd, ts_wd = load_trajectory(args.traj)
    print(f"Loaded {len(times_us)} trajectory poses")

    # Build device world positions for trajectory overlay in top-down view
    traj_xy = [(ts_wd[i][0], ts_wd[i][1]) for i in range(0, len(ts_wd), 50)]

    # Open VRS + calibration
    provider = data_provider.create_vrs_data_provider(args.vrs)
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib("camera-rgb")

    T_dc = cam_calib.get_transform_device_camera()
    R_dc = T_dc.rotation().to_matrix()
    t_dc = T_dc.translation().flatten()

    img_w, img_h = cam_calib.get_image_size()
    N = int(img_w)
    print(f"RGB image size: {N}×{int(img_h)}, applying CW-90° rotation")

    panel_size = int(N * args.scale)
    topdown_size = panel_size

    if args.scene_mode:
        print(f"Scene mode: {len(obbs)} fused objects → broadcast to all frames")
        t_start, t_end = times_us[0], times_us[-1]
        step_us = 100_000  # 100 ms = 10 Hz
        timestamps_us = list(range(int(t_start), int(t_end), step_us))
        timestamps_ns = [t * 1000 for t in timestamps_us]
        scene_obbs = obbs
        print(f"Rendering {len(timestamps_ns)} frames at ~10 Hz")
    else:
        timestamps_ns = sorted(obbs["time_ns"].unique())
        scene_obbs = None
        print(f"Snippet mode: {len(obbs)} OBBs at {len(timestamps_ns)} timestamps "
              f"(thresh={args.prob_thresh})")

    for i, ts_ns in enumerate(timestamps_ns):
        img_data, _ = provider.get_image_data_by_time_ns(
            RGB_SID, int(ts_ns),
            sensor_data.TimeDomain.DEVICE_TIME,
            sensor_data.TimeQueryOptions.CLOSEST,
        )
        if not img_data.is_valid():
            continue

        frame_raw = img_data.to_numpy_array().copy()
        if frame_raw.ndim == 2:
            frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_GRAY2BGR)
        elif frame_raw.shape[2] == 1:
            frame_raw = cv2.cvtColor(frame_raw[:,:,0], cv2.COLOR_GRAY2BGR)
        frame = cv2.rotate(frame_raw, cv2.ROTATE_90_CLOCKWISE)

        R_wd, t_wd = interp_pose(times_us, Rs_wd, ts_wd, int(ts_ns))

        raw_obbs = scene_obbs if args.scene_mode else obbs[obbs["time_ns"] == ts_ns]

        # Frustum cull in scene mode: within topdown_radius AND forward hemisphere.
        if args.scene_mode:
            optical_w = R_wd @ (R_dc @ np.array([0.0, 0.0, 1.0]))
            centres = raw_obbs[['tx_world_object',
                                'ty_world_object',
                                'tz_world_object']].values
            to_obj = centres - t_wd[None, :]
            dist   = np.linalg.norm(to_obj, axis=1)
            dot    = (to_obj / (dist[:, None] + 1e-6)) @ optical_w
            mask   = (dist < args.topdown_radius) & (dot > -0.2)
            frame_obbs = raw_obbs[mask]
        else:
            frame_obbs = raw_obbs

        drawn = 0
        for _, row in frame_obbs.iterrows():
            corners_w = obb_corners_world(
                row["tx_world_object"], row["ty_world_object"],
                row["tz_world_object"],
                row["qw_world_object"], row["qx_world_object"],
                row["qy_world_object"], row["qz_world_object"],
                row["scale_x"], row["scale_y"], row["scale_z"],
            )
            color = get_color(row["name"])
            label = f"{row['name']} {row['prob']:.2f}"
            draw_obb(frame, corners_w, R_wd, t_wd, R_dc, t_dc,
                     cam_calib, color, label, N,
                     n_samples=args.n_edge_samples)
            drawn += 1

        if args.scale != 1.0:
            frame = cv2.resize(frame, (panel_size, panel_size),
                               interpolation=cv2.INTER_AREA)

        topdown = create_topdown(
            frame_obbs, t_wd, R_wd, R_dc,
            all_cam_positions=traj_xy,
            view_size=topdown_size,
            radius_m=args.topdown_radius,
        )

        combined = np.concatenate([frame, topdown], axis=1)
        out_path = os.path.join(args.output_dir, f"frame_{i:04d}_{ts_ns}.jpg")
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i % 100 == 0 or i < 5:
            print(f"  [{i+1}/{len(timestamps_ns)}] ts={ts_ns}  drawn={drawn}")

    print(f"\nDone. {len(timestamps_ns)} frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
