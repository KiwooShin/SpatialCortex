"""
render_nav_video.py — render a navigation-annotated MP4 for an entire VRS sequence.

For every RGB frame in the recording:
  - Projects all scene OBBs (from snippet_obbs.csv, correct per-frame orientations)
  - Highlights the queried target object (white box + label)
  - Draws a HUD compass arrow in the bottom-right that always points toward the target
  - Shows live distance to target
  - Writes all frames to an MP4 video

Design note: snippet_obbs.csv is used instead of scene_obbs.csv.
The fused scene_obbs naively averages quaternions from observations that
oscillate between two 90°-ambiguous orientations, producing a wrong average.
Per-frame snippet_obbs carry the correct orientation for each frame.

Usage:
    conda activate efm3d
    python scripts/navigate/render_nav_video.py \\
        --scene seq01 --find "lamp" \\
        --out output/nav_video_seq01_lamp.mp4

    # Quick preview (every 3rd frame):
    python scripts/navigate/render_nav_video.py \\
        --scene seq01 --find "sofa" --stride 3 --out output/nav_preview.mp4
"""

import argparse
import os
import sqlite3
import sys
from bisect import bisect_left
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import pandas as pd

from projectaria_tools.core import data_provider, sensor_data

from spatialcortex.config import BASE, RGB_SID, SCENES, DB_PATH, OUTPUT_DIR
from spatialcortex.geometry import obb_corners_world, load_trajectory, interp_pose
from spatialcortex.drawing import get_color, draw_obb, draw_hud_arrow


# ── Target lookup ─────────────────────────────────────────────────────────────

def get_target(scene: str, find_str: str, db_path: str) -> dict:
    """Find the best-matching object in the DB for the given scene + query string."""
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(objects)").fetchall()]
    rows = conn.execute("SELECT * FROM objects WHERE scene=?", (scene,)).fetchall()
    objs = [dict(zip(cols, r)) for r in rows]
    conn.close()
    if not objs:
        raise ValueError(f"No objects in scene={scene}")
    q = find_str.lower()
    exact = [o for o in objs if o['name'].lower() == q]
    if exact:
        return max(exact, key=lambda o: o['prob'])
    partial = [o for o in objs if q in o['name'].lower() or o['name'].lower() in q]
    if partial:
        return max(partial, key=lambda o: o['prob'])
    best = max(objs, key=lambda o: o['prob'])
    print(f"  Warning: no match for '{find_str}', using {best['name']}")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene',  required=True, choices=list(SCENES))
    ap.add_argument('--find',   required=True,
                    help='Object name to navigate to, e.g. "lamp" or "sofa"')
    ap.add_argument('--out',    required=True, help='Output MP4 path')
    ap.add_argument('--stride', type=int,   default=1,
                    help='Process every Nth frame (default 1 = all frames)')
    ap.add_argument('--fps',    type=float, default=10.0,
                    help='Source VRS FPS (output FPS = fps / stride)')
    ap.add_argument('--db',     default=str(DB_PATH))
    args = ap.parse_args()

    cfg = SCENES[args.scene]

    # Target object
    target     = get_target(args.scene, args.find, args.db)
    target_pos = np.array([target['tx'], target['ty'], target['tz']])
    print(f"Target : {target['name'].upper()}  "
          f"pos=({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) m")

    # Snippet OBBs — grouped by timestamp for fast per-frame lookup
    snip_df = pd.read_csv(str(cfg['snippet_obbs']))
    snip_df = snip_df[snip_df['prob'] >= 0.25].copy()
    print(f"Snippet OBBs: {len(snip_df)} rows across {snip_df['time_ns'].nunique()} timestamps")

    snip_timestamps = sorted(snip_df['time_ns'].unique())
    snip_groups: dict[int, list] = {}
    for ts_key, grp in snip_df.groupby('time_ns'):
        obbs = []
        for _, row in grp.iterrows():
            corners = obb_corners_world(
                row['tx_world_object'], row['ty_world_object'], row['tz_world_object'],
                row['qw_world_object'], row['qx_world_object'],
                row['qy_world_object'], row['qz_world_object'],
                row['scale_x'], row['scale_y'], row['scale_z'],
            )
            is_target = (row['name'].lower() == target['name'].lower() and
                         abs(row['tx_world_object'] - target['tx']) < 0.5 and
                         abs(row['ty_world_object'] - target['ty']) < 0.5)
            obbs.append({
                'corners':   corners,
                'name':      row['name'],
                'color':     get_color(row['name']),
                'is_target': is_target,
            })
        snip_groups[int(ts_key)] = obbs

    def nearest_snip(ts_ns: int) -> list:
        """OBB list for the snippet timestamp closest to ts_ns."""
        idx = bisect_left(snip_timestamps, ts_ns)
        if idx == 0:
            return snip_groups[snip_timestamps[0]]
        if idx >= len(snip_timestamps):
            return snip_groups[snip_timestamps[-1]]
        before = snip_timestamps[idx - 1]
        after  = snip_timestamps[idx]
        key = before if (ts_ns - before) <= (after - ts_ns) else after
        return snip_groups[key]

    # Trajectory
    print("Loading trajectory …")
    times_us, Rs_wd, ts_pos = load_trajectory(str(cfg['traj']))

    # VRS provider + camera calibration
    print("Opening VRS …", flush=True)
    provider  = data_provider.create_vrs_data_provider(str(cfg['vrs']))
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib("camera-rgb")
    T_dc  = cam_calib.get_transform_device_camera()
    R_dc  = T_dc.rotation().to_matrix()
    t_dc  = T_dc.translation().flatten()
    img_w, _ = cam_calib.get_image_size()
    N = int(img_w)   # 1408

    all_ts    = provider.get_timestamps_ns(RGB_SID, sensor_data.TimeDomain.DEVICE_TIME)
    ts_sample = all_ts[::args.stride]
    print(f"Frames: {len(all_ts)} total → {len(ts_sample)} to render (stride={args.stride})")

    # Video writer
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_fps = args.fps / args.stride
    writer  = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*'mp4v'),
        out_fps,
        (N, N),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {args.out}")

    print(f"Rendering {len(ts_sample)} frames → {args.out} …", flush=True)

    for fi, ts_ns in enumerate(ts_sample):
        img_data, _ = provider.get_image_data_by_time_ns(
            RGB_SID, int(ts_ns),
            sensor_data.TimeDomain.DEVICE_TIME,
            sensor_data.TimeQueryOptions.CLOSEST,
        )
        if not img_data.is_valid():
            continue

        arr = img_data.to_numpy_array().copy()
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 1:
            arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
        frame = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)

        R_wd, t_wd = interp_pose(times_us, Rs_wd, ts_pos, int(ts_ns))
        obb_data   = nearest_snip(int(ts_ns))

        # Non-target OBBs first, then target on top
        for obj in obb_data:
            if obj['is_target']:
                continue
            draw_obb(frame, obj['corners'], R_wd, t_wd, R_dc, t_dc,
                     cam_calib, obj['color'], obj['name'], N)
        for obj in obb_data:
            if obj['is_target']:
                draw_obb(frame, obj['corners'], R_wd, t_wd, R_dc, t_dc,
                         cam_calib, (255, 255, 255),
                         f">>> {obj['name'].upper()} <<<", N,
                         lw=4, highlight=True)

        draw_hud_arrow(frame, R_wd, t_wd, R_dc, t_dc, target_pos)

        dist_m = float(np.linalg.norm(target_pos - t_wd))
        banner = (f"Scene: {args.scene}  |  Target: {target['name'].upper()}  |  "
                  f"Dist: {dist_m:.1f} m  |  Frame {fi+1}/{len(ts_sample)}")
        cv2.rectangle(frame, (0, 0), (N, 32), (20, 20, 20), -1)
        cv2.putText(frame, banner, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        if (fi + 1) % 50 == 0:
            print(f"  {fi+1}/{len(ts_sample)} frames …", flush=True)

    writer.release()
    print(f"\n✓ Done. Video saved to: {args.out}")
    print(f"  {len(ts_sample)} frames @ {out_fps:.1f} fps")


if __name__ == '__main__':
    main()
