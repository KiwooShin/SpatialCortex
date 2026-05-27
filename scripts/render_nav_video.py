"""
render_nav_video.py — render a navigation-annotated video for an entire VRS sequence.

For every RGB frame in the recording:
  - Projects all scene OBBs (fused, from scene_obbs.csv)
  - Highlights the queried target object (white box + label)
  - Draws a HUD compass arrow in the corner that always points toward the target
  - Shows distance to target (updates each frame as camera moves)
  - Writes all frames to an MP4 video

Usage:
    conda activate efm3d
    python scripts/render_nav_video.py \
        --scene seq01 \
        --find "lamp" \
        --out output/nav_video_seq01_lamp.mp4

    # Every Nth frame for a quick preview:
    python scripts/render_nav_video.py --scene seq01 --find "sofa" --stride 3 --out output/nav_preview.mp4
"""

import argparse
import os
import sqlite3
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from projectaria_tools.core import data_provider, sensor_data
from projectaria_tools.core.stream_id import StreamId

BASE    = Path(__file__).parent.parent
RGB_SID = StreamId(214, 1)

SCENES = {
    "seq00": {
        "vrs":          BASE / "data/aeo/aeo_seq00_173376298563204/main.vrs",
        "traj":         BASE / "data/aeo/aeo_seq00_173376298563204/mps/mps/slam/closed_loop_trajectory.csv",
        "snippet_obbs": BASE / "output/efm3d_aeo_seq00/model_release/aeo_seq00_173376298563204/snippet_obbs.csv",
    },
    "seq01": {
        "vrs":          BASE / "data/aeo/aeo_seq01_208838848508107/main.vrs",
        "traj":         BASE / "data/aeo/aeo_seq01_208838848508107/mps/slam/closed_loop_trajectory.csv",
        "snippet_obbs": BASE / "output/efm3d_aeo_seq01/model_release/aeo_seq01_208838848508107/snippet_obbs.csv",
    },
    "seq02": {
        "vrs":          BASE / "data/aeo/aeo_seq02_181771578105956/main.vrs",
        "traj":         BASE / "data/aeo/aeo_seq02_181771578105956/mps/slam/closed_loop_trajectory.csv",
        "snippet_obbs": BASE / "output/efm3d_aeo_seq02/model_release/aeo_seq02_181771578105956/snippet_obbs.csv",
    },
}

BB3D_LINE_ORDERS = [
    [0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7],
]

_SSI_RGB = {
    "chair":(0.20,0.60,1.00),"sofa":(0.10,0.50,0.10),"table":(1.00,1.00,0.00),
    "shelf":(0.50,0.00,0.50),"lamp":(1.00,0.80,0.25),"bed":(0.90,0.40,0.60),
    "monitor":(0.00,0.80,0.80),"cabinet":(0.80,0.60,0.20),"tv":(0.30,0.90,0.70),
    "window":(0.70,0.90,1.00),"door":(0.80,0.70,0.60),"picture_frame":(1.00,0.60,0.20),
    "dresser":(0.85,0.55,0.15),"mirror":(0.60,0.70,1.00),"plant":(0.20,0.80,0.20),
    "pillow":(0.80,0.60,0.80),"floor_mat":(0.60,0.40,0.20),"cart":(0.60,0.80,0.40),
    "container":(0.80,0.50,0.20),"ladder":(0.50,0.80,0.30),
}
_DEFAULT_RGB = (0.70, 0.70, 0.70)

def _bgr255(r,g,b): return (int(b*255),int(g*255),int(r*255))
def get_color(name): return _bgr255(*_SSI_RGB.get(name.lower(), _DEFAULT_RGB))


# ── Geometry ───────────────────────────────────────────────────────────────────

def quat_to_rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw**2+qx**2+qy**2+qz**2)
    qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

def obb_corners_world(tx,ty,tz,qw,qx,qy,qz,sx,sy,sz):
    R = quat_to_rotmat(qw,qx,qy,qz)
    c = np.array([tx,ty,tz])
    hx,hy,hz = sx/2,sy/2,sz/2
    ids = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    xs,ys,zs = [-hx,hx],[-hy,hy],[-hz,hz]
    return c + np.array([[xs[xi],ys[yi],zs[zi]] for xi,yi,zi in ids]) @ R.T

def load_trajectory(traj_csv):
    df = pd.read_csv(traj_csv)
    times_us = df["tracking_timestamp_us"].values.astype(np.int64)
    Rs, ts_pos = [], []
    for _, row in df.iterrows():
        Rs.append(quat_to_rotmat(row["qw_world_device"],row["qx_world_device"],
                                 row["qy_world_device"],row["qz_world_device"]))
        ts_pos.append(np.array([row["tx_world_device"],row["ty_world_device"],
                                 row["tz_world_device"]]))
    return times_us, Rs, ts_pos

def interp_pose(times_us, Rs, ts_pos, query_ns):
    idx = min(max(bisect_left(times_us, query_ns//1000), 0), len(times_us)-1)
    return Rs[idx], ts_pos[idx]

def rotate_cw90(u, v, N): return N-1-v, u

def project_pt(pw, R_wd, t_wd, R_dc, t_dc, cam_calib, N):
    pc = R_dc.T @ (R_wd.T @ (pw - t_wd) - t_dc)
    if pc[2] <= 0.05: return None
    uv = cam_calib.project(pc)
    if uv is None: return None
    return rotate_cw90(uv[0], uv[1], N)


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_obb(img, corners_world, R_wd, t_wd, R_dc, t_dc,
             cam_calib, color, label, N, lw=2, alpha=0.15, highlight=False):
    H, W = img.shape[:2]
    corners_px = [project_pt(c, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
                  for c in corners_world]

    # Filled hull for highlight
    if highlight and all(p is not None for p in corners_px):
        pts = np.array(corners_px, dtype=np.int32)
        if pts[:,0].min()>=0 and pts[:,0].max()<W and pts[:,1].min()>=0 and pts[:,1].max()<H:
            ov = img.copy()
            cv2.fillConvexPoly(ov, cv2.convexHull(pts), color)
            cv2.addWeighted(ov, 0.28, img, 0.72, 0, img)

    any_drawn = False
    for i, j in BB3D_LINE_ORDERS:
        pts3 = [corners_world[i] + t*(corners_world[j]-corners_world[i])
                for t in np.linspace(0,1,10)]
        pts2 = [project_pt(p, R_wd, t_wd, R_dc, t_dc, cam_calib, N) for p in pts3]
        for k in range(len(pts2)-1):
            p0,p1 = pts2[k],pts2[k+1]
            if p0 is None or p1 is None: continue
            x0,y0 = int(round(p0[0])),int(round(p0[1]))
            x1,y1 = int(round(p1[0])),int(round(p1[1]))
            sl = lambda x,y: -60<=x<W+60 and -60<=y<H+60
            if not (sl(x0,y0) or sl(x1,y1)): continue
            cv2.line(img,(x0,y0),(x1,y1),color,lw,cv2.LINE_AA)
            any_drawn = True

    if any_drawn and label:
        valid = [p for p in corners_px if p is not None]
        if valid:
            cx = int(np.mean([p[0] for p in valid]))
            cy = int(np.mean([p[1] for p in valid]))
            cx = max(2,min(cx,W-120)); cy = max(16,min(cy,H-4))
            sc = 0.58 if highlight else 0.42
            th = 2 if highlight else 1
            (tw,tht),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,sc,th)
            cv2.rectangle(img,(cx-2,cy-tht-3),(cx+tw+2,cy+3),(10,10,10),-1)
            cv2.putText(img,label,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,sc,color,th,cv2.LINE_AA)


def draw_hud_arrow(img, R_wd, t_wd, R_dc, t_dc, target_pos):
    """Compass HUD in bottom-right corner always pointing toward target."""
    H, W = img.shape[:2]
    cx, cy = W - 70, H - 70
    r = 48

    dist_m = np.linalg.norm(np.array(target_pos) - t_wd)

    # Semi-transparent background
    ov = img.copy()
    cv2.circle(ov, (cx,cy), r+4, (20,20,20), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
    cv2.circle(img, (cx,cy), r+4, (160,160,160), 1, cv2.LINE_AA)

    if dist_m < 0.3:
        cv2.putText(img,"HERE",(cx-22,cy+6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,100),2,cv2.LINE_AA)
    else:
        # Direction to target in world XY
        delta_xy = np.array(target_pos[:2]) - t_wd[:2]
        delta_xy /= np.linalg.norm(delta_xy)

        # Camera forward in world XY
        fwd = R_wd @ (R_dc @ np.array([0.,0.,1.]))
        fwd_xy = fwd[:2]
        nl = np.linalg.norm(fwd_xy)
        if nl > 1e-4: fwd_xy /= nl

        # Signed angle (right-hand: positive = target to the left in world)
        angle = -np.arctan2(
            fwd_xy[0]*delta_xy[1] - fwd_xy[1]*delta_xy[0],
            fwd_xy[0]*delta_xy[0] + fwd_xy[1]*delta_xy[1],
        )

        tip  = (int(cx + (r-8)*np.sin(angle)),  int(cy - (r-8)*np.cos(angle)))
        tail = (int(cx - (r-20)*np.sin(angle)), int(cy + (r-20)*np.cos(angle)))
        cv2.arrowedLine(img, tail, tip, (0,200,255), 3, cv2.LINE_AA, tipLength=0.32)

        # Cardinal ticks
        for a in [0, np.pi/2, np.pi, 3*np.pi/2]:
            cv2.line(img,
                     (int(cx+(r)*np.sin(a)),   int(cy-(r)*np.cos(a))),
                     (int(cx+(r-7)*np.sin(a)), int(cy-(r-7)*np.cos(a))),
                     (120,120,120), 1, cv2.LINE_AA)

    # Distance text
    dist_str = f"{dist_m:.1f} m"
    (tw,_),_ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(img, dist_str, (cx-tw//2, cy+r+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1, cv2.LINE_AA)


# ── Lookup target from DB ──────────────────────────────────────────────────────

def get_target(scene, find_str, db_path):
    """
    Find the best matching object in the DB for the given scene + query string.
    Prefers exact name match, then partial match.
    Returns a dict with tx, ty, tz, name, scale_z.
    """
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(objects)").fetchall()]
    rows = conn.execute("SELECT * FROM objects WHERE scene=?", (scene,)).fetchall()
    objs = [dict(zip(cols,r)) for r in rows]
    conn.close()
    if not objs:
        raise ValueError(f"No objects in scene={scene}")

    q = find_str.lower()
    # Exact name first, then substring
    exact = [o for o in objs if o['name'].lower() == q]
    if exact:
        best = max(exact, key=lambda o: o['prob'])
    else:
        partial = [o for o in objs if q in o['name'].lower() or o['name'].lower() in q]
        if partial:
            best = max(partial, key=lambda o: o['prob'])
        else:
            # Fall back: highest prob object
            best = max(objs, key=lambda o: o['prob'])
            print(f"  Warning: no match for '{find_str}', using {best['name']}")
    return best


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene',  required=True, choices=list(SCENES))
    ap.add_argument('--find',   required=True,
                    help='Object name to navigate to, e.g. "lamp" or "sofa"')
    ap.add_argument('--out',    required=True, help='Output MP4 path')
    ap.add_argument('--stride', type=int, default=1,
                    help='Process every Nth frame (default 1 = all frames)')
    ap.add_argument('--fps',    type=float, default=10.0,
                    help='Output video FPS (default 10 = original VRS rate)')
    ap.add_argument('--db',     default=str(BASE/'data/scene_db.sqlite'))
    args = ap.parse_args()

    cfg = SCENES[args.scene]

    # ── Load target ────────────────────────────────────────────────────────────
    target = get_target(args.scene, args.find, args.db)
    target_pos = np.array([target['tx'], target['ty'], target['tz']])
    floor_z    = target['tz'] - target['scale_z'] / 2.0
    print(f"Target : {target['name'].upper()}  pos=({target_pos[0]:.2f}, "
          f"{target_pos[1]:.2f}, {target_pos[2]:.2f}) m  floor_z={floor_z:.2f}")

    # ── Load snippet OBBs (per-frame, correct orientations) ───────────────────
    # Use snippet_obbs.csv rather than scene_obbs.csv.
    # scene_obbs fuses quaternions across frames that often flip 90°, producing
    # an averaged orientation that matches neither — boxes appear tilted.
    # snippet_obbs stores the per-frame detections so each box is correctly
    # oriented for the frame it was observed in.
    snip_df = pd.read_csv(str(cfg['snippet_obbs']))
    snip_df = snip_df[snip_df['prob'] >= 0.25].copy()
    print(f"Snippet OBBs: {len(snip_df)} rows across "
          f"{snip_df['time_ns'].nunique()} timestamps")

    # Pre-compute corners for every snippet row and group by timestamp
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
        """Return the OBB list for the snippet timestamp closest to ts_ns."""
        idx = bisect_left(snip_timestamps, ts_ns)
        if idx == 0:
            return snip_groups[snip_timestamps[0]]
        if idx >= len(snip_timestamps):
            return snip_groups[snip_timestamps[-1]]
        before = snip_timestamps[idx - 1]
        after  = snip_timestamps[idx]
        key = before if (ts_ns - before) <= (after - ts_ns) else after
        return snip_groups[key]

    # ── Load trajectory ────────────────────────────────────────────────────────
    print("Loading trajectory …")
    times_us, Rs_wd, ts_pos = load_trajectory(str(cfg['traj']))

    # ── Open VRS ───────────────────────────────────────────────────────────────
    print("Opening VRS …", flush=True)
    provider  = data_provider.create_vrs_data_provider(str(cfg['vrs']))
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib("camera-rgb")
    T_dc  = cam_calib.get_transform_device_camera()
    R_dc  = T_dc.rotation().to_matrix()
    t_dc  = T_dc.translation().flatten()
    img_w, img_h = cam_calib.get_image_size()
    N = int(img_w)   # 1408 (square)

    all_ts = provider.get_timestamps_ns(RGB_SID, sensor_data.TimeDomain.DEVICE_TIME)
    ts_sample = all_ts[::args.stride]
    print(f"Frames: {len(all_ts)} total → {len(ts_sample)} to render (stride={args.stride})")

    # ── Video writer ───────────────────────────────────────────────────────────
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

    # ── Render loop ────────────────────────────────────────────────────────────
    print(f"Rendering {len(ts_sample)} frames → {args.out} …", flush=True)

    for fi, ts_ns in enumerate(ts_sample):
        # Load frame
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
            arr = cv2.cvtColor(arr[:,:,0], cv2.COLOR_GRAY2BGR)
        frame = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)

        # Camera pose at this timestamp
        R_wd, t_wd = interp_pose(times_us, Rs_wd, ts_pos, int(ts_ns))

        # Get OBBs from nearest snippet (correct per-frame orientations)
        obb_data = nearest_snip(int(ts_ns))

        # Draw non-target OBBs first
        for obj in obb_data:
            if obj['is_target']:
                continue   # draw target last (on top)
            draw_obb(frame, obj['corners'], R_wd, t_wd, R_dc, t_dc,
                     cam_calib, obj['color'], obj['name'], N, lw=2)

        # Draw target (highlighted)
        for obj in obb_data:
            if obj['is_target']:
                draw_obb(frame, obj['corners'], R_wd, t_wd, R_dc, t_dc,
                         cam_calib, (255,255,255),
                         f">>> {obj['name'].upper()} <<<",
                         N, lw=4, highlight=True)

        # HUD compass arrow
        draw_hud_arrow(frame, R_wd, t_wd, R_dc, t_dc, target_pos)

        # Frame counter + timestamp banner
        dist_m = np.linalg.norm(target_pos - t_wd)
        banner = (f"Scene: {args.scene}  |  Target: {target['name'].upper()}  |  "
                  f"Dist: {dist_m:.1f} m  |  Frame {fi+1}/{len(ts_sample)}")
        cv2.rectangle(frame, (0,0), (N,32), (20,20,20), -1)
        cv2.putText(frame, banner, (10,22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1, cv2.LINE_AA)

        writer.write(frame)

        if (fi+1) % 50 == 0:
            print(f"  {fi+1}/{len(ts_sample)} frames …", flush=True)

    writer.release()
    print(f"\n✓ Done. Video saved to: {args.out}")
    print(f"  {len(ts_sample)} frames @ {out_fps:.1f} fps")


if __name__ == '__main__':
    main()
