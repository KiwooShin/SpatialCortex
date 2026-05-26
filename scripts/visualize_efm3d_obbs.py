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
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from projectaria_tools.core import data_provider, sensor_data
from projectaria_tools.core.stream_id import StreamId

RGB_SID = StreamId(214, 1)

# ── Edge connectivity (BB3D_LINE_ORDERS from efm3d) ───────────────────────────
BB3D_LINE_ORDERS = [
    [0, 1], [1, 2], [2, 3], [3, 0],   # bottom face ring
    [4, 5], [5, 6], [6, 7], [7, 4],   # top face ring
    [0, 4], [1, 5], [2, 6], [3, 7],   # vertical pillars
]

# ── Canonical EFM3D class colors (SSI_SEM_COLORS) ────────────────────────────
_SSI_RGB = {
    "chair":        (0.20, 0.60, 1.00),
    "sofa":         (0.10, 0.50, 0.10),
    "table":        (1.00, 1.00, 0.00),
    "shelf":        (0.50, 0.00, 0.50),
    "lamp":         (1.00, 0.80, 0.25),
    "bed":          (0.90, 0.40, 0.60),
    "monitor":      (0.00, 0.80, 0.80),
    "ladder":       (0.50, 0.80, 0.30),
    "container":    (0.80, 0.50, 0.20),
    "mirror":       (0.60, 0.70, 1.00),
    "cabinet":      (0.80, 0.60, 0.20),
    "tv":           (0.30, 0.90, 0.70),
    "plant":        (0.20, 0.80, 0.20),
    "microwave":    (1.00, 0.40, 0.40),
    "refrigerator": (0.40, 0.40, 1.00),
    "oven":         (0.70, 0.60, 0.30),
    "whiteboard":   (0.90, 0.90, 0.70),
    "curtain":      (0.70, 0.50, 1.00),
    "window":       (0.70, 0.90, 1.00),
    "door":         (0.80, 0.70, 0.60),
    "picture_frame":(1.00, 0.60, 0.20),
    "floor_mat":    (0.60, 0.40, 0.20),
    "trash_can":    (0.50, 0.50, 0.50),
    "book":         (0.90, 0.70, 0.40),
    "bottle":       (0.40, 0.80, 0.60),
}
_DEFAULT_RGB = (0.70, 0.70, 0.70)


def _bgr255(r, g, b):
    return (int(b * 255), int(g * 255), int(r * 255))


def get_color(name: str):
    return _bgr255(*_SSI_RGB.get(name.lower(), _DEFAULT_RGB))


# ── Geometry ──────────────────────────────────────────────────────────────────

def quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ])


def obb_corners_world(tx, ty, tz, qw, qx, qy, qz, sx, sy, sz) -> np.ndarray:
    """8 OBB corners in world space. sx/sy/sz are FULL dimensions."""
    R = quat_to_rotmat(qw, qx, qy, qz)
    center = np.array([tx, ty, tz])
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    xs, ys, zs = [-hx, hx], [-hy, hy], [-hz, hz]
    ids = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
           (0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    corners_obj = np.array([[xs[xi], ys[yi], zs[zi]] for xi, yi, zi in ids])
    return center + corners_obj @ R.T   # (8, 3)


def load_trajectory(traj_csv: str):
    df = pd.read_csv(traj_csv)
    times_us = df["tracking_timestamp_us"].values.astype(np.int64)
    Rs, ts = [], []
    for _, row in df.iterrows():
        R = quat_to_rotmat(row["qw_world_device"], row["qx_world_device"],
                           row["qy_world_device"], row["qz_world_device"])
        t = np.array([row["tx_world_device"], row["ty_world_device"],
                      row["tz_world_device"]])
        Rs.append(R); ts.append(t)
    return times_us, Rs, ts


def interp_pose(times_us, Rs, ts, query_ns: int):
    query_us = query_ns // 1000
    idx = min(max(bisect_left(times_us, query_us), 0), len(times_us) - 1)
    return Rs[idx], ts[idx]


def world_to_camera(pt_world, R_wd, t_wd, R_dc, t_dc):
    p_dev = R_wd.T @ (pt_world - t_wd)
    return R_dc.T @ (p_dev - t_dc)


def rotate_cw90(u, v, N):
    """Transform raw pixel (u,v) → (u',v') after 90° CW rotation of square N×N image."""
    return N - 1 - v, u


# ── Fisheye overlay ───────────────────────────────────────────────────────────

def _project_pt(pw, R_wd, t_wd, R_dc, t_dc, cam_calib, img_N):
    """Project world point → rotated-image pixel, or None."""
    pc = world_to_camera(pw, R_wd, t_wd, R_dc, t_dc)
    if pc[2] <= 0.05:
        return None
    uv = cam_calib.project(pc)
    if uv is None:
        return None
    u_rot, v_rot = rotate_cw90(uv[0], uv[1], img_N)
    return float(u_rot), float(v_rot)


def draw_obb_fisheye(img, corners_world, R_wd, t_wd, R_dc, t_dc,
                     cam_calib, color, label, n_samples=12, alpha=0.18):
    H, W = img.shape[:2]
    N = W  # square image

    # Project all 8 corners
    corners_px = [_project_pt(pw, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
                  for pw in corners_world]

    # Filled hull (only when all 8 corners project)
    if all(p is not None for p in corners_px):
        pts = np.array(corners_px, dtype=np.int32)
        if (pts[:,0].min() >= 0 and pts[:,0].max() < W and
                pts[:,1].min() >= 0 and pts[:,1].max() < H):
            overlay = img.copy()
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(overlay, hull, color)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Fisheye-correct edges (sample along each edge)
    any_drawn = False
    for i, j in BB3D_LINE_ORDERS:
        ts_edge = np.linspace(0, 1, n_samples)
        pts3d = [corners_world[i] + t * (corners_world[j] - corners_world[i])
                 for t in ts_edge]
        pts2d = [_project_pt(p, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
                 for p in pts3d]
        for k in range(len(pts2d) - 1):
            p0, p1 = pts2d[k], pts2d[k+1]
            if p0 is None or p1 is None:
                continue
            x0, y0 = int(round(p0[0])), int(round(p0[1]))
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            in_slack = lambda x, y: -60 <= x < W+60 and -60 <= y < H+60
            if not (in_slack(x0, y0) or in_slack(x1, y1)):
                continue
            cv2.line(img, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            any_drawn = True

    # Label at centroid of visible corners
    if any_drawn:
        valid = [p for p in corners_px if p is not None]
        if valid:
            cx = int(np.mean([p[0] for p in valid]))
            cy = int(np.mean([p[1] for p in valid]))
            cx = max(2, min(cx, W - 80))
            cy = max(12, min(cy, H - 4))
            font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
            (tw, tht), _ = cv2.getTextSize(label, font, sc, th)
            cv2.rectangle(img, (cx-2, cy-tht-3), (cx+tw+2, cy+2), (20,20,20), -1)
            cv2.putText(img, label, (cx, cy), font, sc, color, th, cv2.LINE_AA)


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
    # Scale bar (1 m)
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
    ap.add_argument("--obbs", required=True)
    ap.add_argument("--vrs", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--output-dir", required=True)
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
    timestamps = sorted(obbs["time_ns"].unique())
    print(f"Loaded {len(obbs)} OBBs at {len(timestamps)} timestamps "
          f"(thresh={args.prob_thresh})")

    # Load trajectory
    times_us, Rs_wd, ts_wd = load_trajectory(args.traj)
    print(f"Loaded {len(times_us)} trajectory poses")

    # Build device world positions for trajectory overlay in top-down view
    # Sample every 50th pose to keep the list small
    traj_xy = [(ts_wd[i][0], ts_wd[i][1]) for i in range(0, len(ts_wd), 50)]

    # Open VRS + calibration
    provider = data_provider.create_vrs_data_provider(args.vrs)
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib("camera-rgb")

    T_dc = cam_calib.get_transform_device_camera()
    R_dc = T_dc.rotation().to_matrix()   # R_device_camera
    t_dc = T_dc.translation().flatten()

    img_w, img_h = cam_calib.get_image_size()
    N = int(img_w)   # square: 1408
    print(f"RGB image size: {N}×{int(img_h)}, applying CW-90° rotation")

    # Target fisheye panel size after downscale
    panel_size = int(N * args.scale)
    topdown_size = panel_size   # square top-down view, same height as fisheye

    for i, ts_ns in enumerate(timestamps):
        img_data, _ = provider.get_image_data_by_time_ns(
            RGB_SID, int(ts_ns),
            sensor_data.TimeDomain.DEVICE_TIME,
            sensor_data.TimeQueryOptions.CLOSEST,
        )
        if not img_data.is_valid():
            print(f"  skip ts={ts_ns}: no image")
            continue

        # Raw frame (1408×1408); rotate CW 90° to correct sensor orientation
        frame_raw = img_data.to_numpy_array().copy()
        if frame_raw.ndim == 2:
            frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_GRAY2BGR)
        elif frame_raw.shape[2] == 1:
            frame_raw = cv2.cvtColor(frame_raw[:,:,0], cv2.COLOR_GRAY2BGR)
        frame = cv2.rotate(frame_raw, cv2.ROTATE_90_CLOCKWISE)

        R_wd, t_wd = interp_pose(times_us, Rs_wd, ts_wd, int(ts_ns))

        frame_obbs = obbs[obbs["time_ns"] == ts_ns]
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
            draw_obb_fisheye(frame, corners_w, R_wd, t_wd, R_dc, t_dc,
                             cam_calib, color, label,
                             n_samples=args.n_edge_samples)
            drawn += 1

        # Downscale fisheye panel
        if args.scale != 1.0:
            frame = cv2.resize(frame, (panel_size, panel_size),
                               interpolation=cv2.INTER_AREA)

        # Top-down panel
        topdown = create_topdown(
            frame_obbs, t_wd, R_wd, R_dc,
            all_cam_positions=traj_xy,
            view_size=topdown_size,
            radius_m=args.topdown_radius,
        )

        # Combine side-by-side
        combined = np.concatenate([frame, topdown], axis=1)

        out_path = os.path.join(args.output_dir, f"frame_{i:04d}_{ts_ns}.jpg")
        cv2.imwrite(out_path, combined,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  [{i+1}/{len(timestamps)}] ts={ts_ns}  drawn={drawn}  → {out_path}")

    print(f"\nDone. {len(timestamps)} combined frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
