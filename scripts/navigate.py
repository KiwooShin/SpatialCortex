"""
navigate.py — re-localize then navigate to a queried object.

Flow:
  query image  → CLIP image embed → keyframe FAISS → user pose (scene + 6-DoF)
  object query → CLIP text  embed → object  FAISS → target object pose
  user pose + target pose → Dijkstra on trajectory waypoints → ordered path
  render: fisheye frame at user position with 3D OBBs + projected nav path

Usage:
  # First run build_keyframe_index.py once:
  python scripts/build_keyframe_index.py

  # Then navigate:
  python scripts/navigate.py \
      --query-image output/crops/seq01/004_sofa.jpg \
      --find "where is the bed" \
      --out output/nav_result.jpg

  # Use any VRS frame as the query image (simulate user's current view):
  python scripts/navigate.py \
      --query-ts 12000000000 --query-scene seq01 \
      --find "where is the lamp" \
      --out output/nav_result.jpg
"""

import argparse
import heapq
import os
import sqlite3
from bisect import bisect_left
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import cv2
import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from projectaria_tools.core import data_provider, sensor_data
from projectaria_tools.core.stream_id import StreamId

BASE    = Path(__file__).parent.parent
RGB_SID = StreamId(214, 1)

CLIP_MODEL      = 'ViT-L-14-quickgelu'
CLIP_PRETRAINED = 'openai'
DEFAULT_KF_FAISS = str(BASE / 'data/keyframe_index.faiss')
DEFAULT_KF_CSV   = str(BASE / 'data/keyframe_index.csv')
DEFAULT_OBJ_FAISS= str(BASE / 'data/scene.faiss')
DEFAULT_DB       = str(BASE / 'data/scene_db.sqlite')
DEFAULT_OUT      = str(BASE / 'output/nav_result.jpg')
WAYPOINT_STEP_M  = 0.5   # sample trajectory every 0.5 m for waypoint graph

SCENES = {
    "seq00": {
        "vrs":   BASE / "data/aeo/aeo_seq00_173376298563204/main.vrs",
        "traj":  BASE / "data/aeo/aeo_seq00_173376298563204/mps/mps/slam/closed_loop_trajectory.csv",
        "snips": BASE / "output/efm3d_aeo_seq00/model_release/aeo_seq00_173376298563204/snippet_obbs.csv",
    },
    "seq01": {
        "vrs":   BASE / "data/aeo/aeo_seq01_208838848508107/main.vrs",
        "traj":  BASE / "data/aeo/aeo_seq01_208838848508107/mps/slam/closed_loop_trajectory.csv",
        "snips": BASE / "output/efm3d_aeo_seq01/model_release/aeo_seq01_208838848508107/snippet_obbs.csv",
    },
    "seq02": {
        "vrs":   BASE / "data/aeo/aeo_seq02_181771578105956/main.vrs",
        "traj":  BASE / "data/aeo/aeo_seq02_181771578105956/mps/slam/closed_loop_trajectory.csv",
        "snips": BASE / "output/efm3d_aeo_seq02/model_release/aeo_seq02_181771578105956/snippet_obbs.csv",
    },
}

SYNONYMS = {
    "sit": ["chair","sofa"], "sleep": ["bed","sofa"], "light": ["lamp"],
    "couch": ["sofa"], "screen": ["monitor","tv"], "storage": ["cabinet","shelf","dresser"],
}

# ── Geometry helpers ───────────────────────────────────────────────────────────

def quat_to_rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw**2+qx**2+qy**2+qz**2)
    qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])

def load_trajectory(traj_csv):
    df = pd.read_csv(traj_csv)
    times_us = df["tracking_timestamp_us"].values.astype(np.int64)
    Rs, ts_pos = [], []
    for _, row in df.iterrows():
        R = quat_to_rotmat(row["qw_world_device"], row["qx_world_device"],
                           row["qy_world_device"], row["qz_world_device"])
        t = np.array([row["tx_world_device"], row["ty_world_device"],
                      row["tz_world_device"]])
        Rs.append(R); ts_pos.append(t)
    return times_us, Rs, ts_pos

def interp_pose(times_us, Rs, ts_pos, query_ns):
    query_us = query_ns // 1000
    idx = min(max(bisect_left(times_us, query_us), 0), len(times_us)-1)
    return Rs[idx], ts_pos[idx]

def rotate_cw90(u, v, N):
    return N-1-v, u

def _project_pt(pw, R_wd, t_wd, R_dc, t_dc, cam_calib, N):
    p_dev = R_wd.T @ (pw - t_wd)
    pc    = R_dc.T @ (p_dev - t_dc)
    if pc[2] <= 0.05: return None
    uv = cam_calib.project(pc)
    if uv is None: return None
    return rotate_cw90(uv[0], uv[1], N)

def obb_corners_world(tx,ty,tz,qw,qx,qy,qz,sx,sy,sz):
    R = quat_to_rotmat(qw,qx,qy,qz)
    c = np.array([tx,ty,tz])
    hx,hy,hz = sx/2,sy/2,sz/2
    ids = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    xs,ys,zs = [-hx,hx],[-hy,hy],[-hz,hz]
    corners = np.array([[xs[xi],ys[yi],zs[zi]] for xi,yi,zi in ids])
    return c + corners @ R.T

BB3D_LINE_ORDERS = [
    [0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7],
]

_SSI_RGB = {
    "chair":(0.20,0.60,1.00),"sofa":(0.10,0.50,0.10),"table":(1.00,1.00,0.00),
    "shelf":(0.50,0.00,0.50),"lamp":(1.00,0.80,0.25),"bed":(0.90,0.40,0.60),
    "monitor":(0.00,0.80,0.80),"cabinet":(0.80,0.60,0.20),"tv":(0.30,0.90,0.70),
    "window":(0.70,0.90,1.00),"door":(0.80,0.70,0.60),"picture_frame":(1.00,0.60,0.20),
    "dresser":(0.85,0.55,0.15),"mirror":(0.60,0.70,1.00),"plant":(0.20,0.80,0.20),
}
_DEFAULT_RGB = (0.70, 0.70, 0.70)

def _bgr255(r,g,b): return (int(b*255),int(g*255),int(r*255))
def get_color(name): return _bgr255(*_SSI_RGB.get(name.lower(), _DEFAULT_RGB))

def draw_obb_fisheye(img, corners_world, R_wd, t_wd, R_dc, t_dc,
                     cam_calib, color, label, n_samples=12, thickness=2,
                     highlight=False, alpha=0.18):
    H, W = img.shape[:2]; N = W
    corners_px = [_project_pt(pw, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
                  for pw in corners_world]
    if highlight and all(p is not None for p in corners_px):
        pts = np.array(corners_px, dtype=np.int32)
        if pts[:,0].min()>=0 and pts[:,0].max()<W and pts[:,1].min()>=0 and pts[:,1].max()<H:
            overlay = img.copy(); hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(overlay, hull, color)
            cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
    lw = 4 if highlight else thickness
    any_drawn = False
    for i,j in BB3D_LINE_ORDERS:
        ts_e = np.linspace(0,1,n_samples)
        pts3d = [corners_world[i]+t*(corners_world[j]-corners_world[i]) for t in ts_e]
        pts2d = [_project_pt(p, R_wd, t_wd, R_dc, t_dc, cam_calib, N) for p in pts3d]
        for k in range(len(pts2d)-1):
            p0,p1 = pts2d[k],pts2d[k+1]
            if p0 is None or p1 is None: continue
            x0,y0 = int(round(p0[0])),int(round(p0[1]))
            x1,y1 = int(round(p1[0])),int(round(p1[1]))
            if not ((-60<=x0<W+60 and -60<=y0<H+60) or (-60<=x1<W+60 and -60<=y1<H+60)): continue
            cv2.line(img,(x0,y0),(x1,y1),color,lw,cv2.LINE_AA); any_drawn=True
    if any_drawn and label:
        valid = [p for p in corners_px if p is not None]
        if valid:
            cx = int(np.mean([p[0] for p in valid])); cy = int(np.mean([p[1] for p in valid]))
            cx = max(2,min(cx,W-120)); cy = max(16,min(cy,H-4))
            font,sc,th = cv2.FONT_HERSHEY_SIMPLEX, 0.60 if highlight else 0.45, 2 if highlight else 1
            (tw,tht),_ = cv2.getTextSize(label,font,sc,th)
            cv2.rectangle(img,(cx-3,cy-tht-4),(cx+tw+3,cy+3),(10,10,10),-1)
            cv2.putText(img,label,(cx,cy),font,sc,color,th,cv2.LINE_AA)

# ── CLIP ───────────────────────────────────────────────────────────────────────

_clip_cache = {}

def load_clip(device):
    if 'model' not in _clip_cache:
        print("Loading CLIP …", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        _clip_cache.update({'model': model, 'pre': preprocess, 'tok': tokenizer})
    return _clip_cache['model'], _clip_cache['pre'], _clip_cache['tok']

@torch.no_grad()
def embed_image(model, preprocess, img_pil, device):
    x = preprocess(img_pil).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]

@torch.no_grad()
def embed_text(model, tokenizer, text, device):
    tokens = tokenizer([text]).to(device)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]

# ── Re-localization ────────────────────────────────────────────────────────────

def relocalize(query_img_pil, kf_index, kf_meta_df, model, preprocess, device, top_k=3):
    """
    Encode query image → FAISS search in keyframe index → return top-k matches.
    Returns list of dicts: {scene, ts_ns, tx, ty, tz, qw, qx, qy, qz, sim}.
    """
    vec = embed_image(model, preprocess, query_img_pil, device).reshape(1,-1).astype(np.float32)
    sims, idxs = kf_index.search(vec, top_k)
    results = []
    for rank, (idx, sim) in enumerate(zip(idxs[0], sims[0])):
        if idx < 0: continue
        row = kf_meta_df.iloc[int(idx)]
        results.append({
            "rank": rank+1, "sim": float(sim),
            "scene": row["scene"], "ts_ns": int(row["ts_ns"]),
            "frame_idx": int(row["frame_idx"]),
            "tx": row["tx"], "ty": row["ty"], "tz": row["tz"],
            "qw": row["qw"], "qx": row["qx"], "qy": row["qy"], "qz": row["qz"],
        })
    return results

# ── Object retrieval (same as query_visual.py) ─────────────────────────────────

def expand_query(query):
    words = query.lower().split()
    extras = []
    for w in words:
        extras.extend(SYNONYMS.get(w, []))
    return extras

def find_object(query, obj_faiss, db_path, scene_filter, model, tokenizer, device):
    index = faiss.read_index(obj_faiss)
    conn  = sqlite3.connect(db_path)
    cols  = [r[1] for r in conn.execute("PRAGMA table_info(objects)").fetchall()]
    vec   = embed_text(model, tokenizer, query, device).reshape(1,-1).astype(np.float32)
    k     = min(50, index.ntotal)
    sims, idxs = index.search(vec, k)
    clip_ids = [int(i) for i in idxs[0] if i >= 0]
    ph   = ','.join('?'*len(clip_ids))
    rows = conn.execute(f"SELECT * FROM objects WHERE clip_idx IN ({ph})", clip_ids).fetchall()
    objs = [dict(zip(cols, r)) for r in rows]
    sim_map = {int(idxs[0][j]): float(sims[0][j]) for j in range(len(idxs[0]))}
    for o in objs: o['sim'] = sim_map.get(o['clip_idx'], 0.0)
    if scene_filter:
        objs = [o for o in objs if o['scene'] == scene_filter]
    qwords = set(query.lower().split()) | set(expand_query(query))
    def name_matches(name):
        return bool(set(name.lower().replace('_',' ').split()) & qwords)
    objs.sort(key=lambda o: (0 if name_matches(o['name']) else 1, -o['sim']))
    conn.close()
    return objs[0] if objs else None

# ── Navigation (Dijkstra) ──────────────────────────────────────────────────────

def build_waypoints(ts_pos_list, step_m=0.5):
    """
    Subsample the trajectory to ~step_m spacing.
    Returns list of (x, y, z) waypoints.
    """
    if not ts_pos_list:
        return []
    waypoints = [ts_pos_list[0].copy()]
    for pt in ts_pos_list[1:]:
        if np.linalg.norm(pt - waypoints[-1]) >= step_m:
            waypoints.append(pt.copy())
    return waypoints

def dijkstra(waypoints, start_idx, end_idx):
    """
    Dijkstra on a kNN-5 graph of waypoints.
    Returns list of waypoint indices from start to end.
    """
    n = len(waypoints)
    if n == 0 or start_idx == end_idx:
        return [start_idx]

    pts = np.array(waypoints)

    # Build adjacency: connect each node to its 5 nearest neighbours
    k_nn = min(6, n)
    from scipy.spatial import KDTree
    tree = KDTree(pts)
    dists_all, nbrs_all = tree.query(pts, k=k_nn)

    dist = [float('inf')] * n
    prev = [-1] * n
    dist[start_idx] = 0.0
    heap = [(0.0, start_idx)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == end_idx:
            break
        for i in range(1, k_nn):   # skip 0 (self)
            v   = int(nbrs_all[u, i])
            w   = float(dists_all[u, i])
            nd  = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    # Reconstruct path
    path = []
    cur  = end_idx
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if path[0] != start_idx:
        return [start_idx, end_idx]  # fallback: direct
    return path

def plan_path(user_pos, target_pos, ts_pos_list, step_m=0.5):
    """
    Build a waypoint graph from the recorded trajectory, then run Dijkstra.
    Returns (waypoints list, path indices).
    """
    waypoints = build_waypoints(ts_pos_list, step_m)
    if not waypoints:
        return [], []

    pts = np.array(waypoints)
    start_idx = int(np.argmin(np.linalg.norm(pts - user_pos, axis=1)))
    end_idx   = int(np.argmin(np.linalg.norm(pts - target_pos, axis=1)))

    path_idxs = dijkstra(waypoints, start_idx, end_idx)
    return waypoints, path_idxs

# ── Frame loading ──────────────────────────────────────────────────────────────

def load_vrs_frame(vrs_path, ts_ns):
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    img_data, _ = provider.get_image_data_by_time_ns(
        RGB_SID, int(ts_ns),
        sensor_data.TimeDomain.DEVICE_TIME,
        sensor_data.TimeQueryOptions.CLOSEST,
    )
    if not img_data.is_valid():
        raise RuntimeError(f"No image at ts={ts_ns}")
    arr = img_data.to_numpy_array().copy()
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:,:,0], cv2.COLOR_GRAY2BGR)
    return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE), provider

def get_cam_calib(provider):
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib("camera-rgb")
    T_dc  = cam_calib.get_transform_device_camera()
    R_dc  = T_dc.rotation().to_matrix()
    t_dc  = T_dc.translation().flatten()
    img_w, _ = cam_calib.get_image_size()
    return cam_calib, R_dc, t_dc, int(img_w)

# ── Rendering ─────────────────────────────────────────────────────────────────

def project_path_on_frame(img, waypoints, path_idxs, R_wd, t_wd, R_dc, t_dc,
                           cam_calib, N, floor_z):
    """
    Draw navigation path as dots + connecting lines on the frame.
    All waypoints are snapped to floor_z so they appear on the ground plane,
    not at eye level (trajectory records device/head height, not foot level).
    """
    path_pts = [waypoints[i] for i in path_idxs]
    prev_px  = None
    PATH_COLOR  = (0, 200, 255)   # orange
    DOT_COLOR   = (0, 255, 100)   # green
    ARROW_COLOR = (0, 100, 255)   # red-orange for final arrow

    for k, wp in enumerate(path_pts):
        # Snap XY from trajectory, Z clamped to floor level
        wp_floor = np.array([wp[0], wp[1], floor_z])
        px = _project_pt(wp_floor, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
        if px is None:
            prev_px = None
            continue
        xi, yi = int(round(px[0])), int(round(px[1]))
        H, W = img.shape[:2]
        in_img = 0 <= xi < W and 0 <= yi < H

        if in_img:
            r = 10 if k in (0, len(path_pts)-1) else 6
            cv2.circle(img, (xi, yi), r, DOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(img, (xi, yi), r, (255,255,255), 1, cv2.LINE_AA)
            # Label first and last dots
            if k == 0:
                cv2.putText(img, "START", (xi+12, yi+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            elif k == len(path_pts)-1:
                cv2.putText(img, "DEST", (xi+12, yi+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        if prev_px is not None:
            x0,y0 = int(round(prev_px[0])),int(round(prev_px[1]))
            x1,y1 = xi, yi
            if not ((-60<=x0<W+60 and -60<=y0<H+60) or (-60<=x1<W+60 and -60<=y1<H+60)):
                prev_px = px; continue
            color = ARROW_COLOR if k == len(path_pts)-1 else PATH_COLOR
            lw = 4
            if k == len(path_pts)-1:
                cv2.arrowedLine(img,(x0,y0),(x1,y1),color,lw,cv2.LINE_AA,tipLength=0.25)
            else:
                cv2.line(img,(x0,y0),(x1,y1),color,lw,cv2.LINE_AA)

        prev_px = px

def draw_direction_hud(img, R_wd, t_wd, R_dc, t_dc, target_pos, dist_m):
    """
    Draw a compass-style HUD in the bottom-right corner that always points
    toward the target, regardless of whether it's in frame.

    - Outer ring = compass
    - Arrow inside = direction to target in camera space
    - Text below = distance
    """
    H, W = img.shape[:2]
    cx, cy = W - 70, H - 70   # compass centre
    r      = 48                # compass radius

    # Background disc
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r+4, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.circle(img, (cx, cy), r+4, (180, 180, 180), 1, cv2.LINE_AA)

    # Compute direction from user to target in world XY
    user_xy   = t_wd[:2]
    target_xy = target_pos[:2]
    delta_xy  = target_xy - user_xy
    d_norm    = np.linalg.norm(delta_xy)

    if d_norm < 0.01:
        # Already at target — draw a check mark
        cv2.putText(img, "HERE", (cx-20, cy+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,100), 2, cv2.LINE_AA)
    else:
        # Camera forward direction in world XY (optical axis projected to ground)
        forward_world = R_wd @ (R_dc @ np.array([0.0, 0.0, 1.0]))
        fwd_xy  = forward_world[:2]
        fwd_len = np.linalg.norm(fwd_xy)
        if fwd_len > 1e-4:
            fwd_xy /= fwd_len

        # Angle between camera forward and direction to target
        # (atan2 of cross/dot gives signed angle: positive = target is to the right)
        dir_xy = delta_xy / d_norm
        angle  = np.arctan2(
            fwd_xy[0]*dir_xy[1] - fwd_xy[1]*dir_xy[0],   # cross (z-component)
            fwd_xy[0]*dir_xy[0] + fwd_xy[1]*dir_xy[1],   # dot
        )
        # Flip sign: positive angle in world = right in camera image
        angle = -angle

        # Arrow tip and tail in compass image space
        tip_x  = int(cx + (r - 10) * np.sin(angle))
        tip_y  = int(cy - (r - 10) * np.cos(angle))
        tail_x = int(cx - (r - 22) * np.sin(angle))
        tail_y = int(cy + (r - 22) * np.cos(angle))

        cv2.arrowedLine(img, (tail_x, tail_y), (tip_x, tip_y),
                        (0, 200, 255), 3, cv2.LINE_AA, tipLength=0.35)

        # Tick marks at N/E/S/W
        for a in [0, np.pi/2, np.pi, 3*np.pi/2]:
            tx_ = int(cx + r * np.sin(a))
            ty_ = int(cy - r * np.cos(a))
            tx2 = int(cx + (r-6) * np.sin(a))
            ty2 = int(cy - (r-6) * np.cos(a))
            cv2.line(img, (tx_, ty_), (tx2, ty2), (160,160,160), 1, cv2.LINE_AA)

    # Distance text below compass
    dist_str = f"{dist_m:.1f} m"
    (tw, _), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(img, dist_str, (cx - tw//2, cy + r + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)


def draw_user_marker(img, R_wd, t_wd, R_dc, t_dc, cam_calib, N, floor_z):
    """Project the user's world position (at floor level) onto the frame."""
    # User is at the camera optical centre — project a point 0.5 m in front on the floor
    # In camera frame, forward = +Z. Transform to world, drop to floor_z.
    forward_cam = np.array([0.0, 0.0, 0.5])
    p_dev = R_dc @ forward_cam + t_dc
    p_world = R_wd @ p_dev + t_wd
    p_world[2] = floor_z   # snap to floor

    px = _project_pt(p_world, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
    H, W = img.shape[:2]
    if px is None:
        # Fallback: draw at image bottom-centre
        xi, yi = W//2, int(H * 0.82)
    else:
        xi, yi = int(round(px[0])), int(round(px[1]))
        if not (0 <= xi < W and 0 <= yi < H):
            xi, yi = W//2, int(H * 0.82)

    cv2.circle(img, (xi, yi), 14, (0, 255, 200), -1, cv2.LINE_AA)
    cv2.circle(img, (xi, yi), 14, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "YOU", (xi-16, yi-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load indices
    print("Loading keyframe index …", flush=True)
    kf_index  = faiss.read_index(args.kf_faiss)
    kf_meta   = pd.read_csv(args.kf_csv)
    print(f"  {kf_index.ntotal} keyframes in index")

    model, preprocess, tokenizer = load_clip(device)

    # ── Step 1: get query image ────────────────────────────────────────────────
    if args.query_image:
        print(f"\nLoading query image: {args.query_image}")
        query_pil = Image.open(args.query_image).convert('RGB')
    elif args.query_ts is not None and args.query_scene:
        print(f"\nLoading query frame from {args.query_scene} ts={args.query_ts}")
        vrs_path  = SCENES[args.query_scene]['vrs']
        arr, _    = load_vrs_frame(vrs_path, args.query_ts)
        query_pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Provide --query-image or (--query-ts + --query-scene)")

    # ── Step 2: re-localize ────────────────────────────────────────────────────
    print("\nRe-localizing …", flush=True)
    reloc_hits = relocalize(query_pil, kf_index, kf_meta, model, preprocess, device)
    if not reloc_hits:
        print("Re-localization failed — no keyframe matches.")
        return

    best_loc = reloc_hits[0]
    print(f"  Top match: scene={best_loc['scene']}  "
          f"ts={best_loc['ts_ns']}  sim={best_loc['sim']:.3f}")
    print(f"  User pose: ({best_loc['tx']:.2f}, {best_loc['ty']:.2f}, {best_loc['tz']:.2f}) m")

    # User is in the re-localized scene; object may be in any scene
    user_scene = best_loc['scene']

    # ── Step 3: find target object ────────────────────────────────────────────
    # Search restricted scene if given, otherwise ALL scenes
    scene_filter = args.scene  # None = search all
    print(f"\nSearching for: \"{args.find}\"" +
          (f" in scene={scene_filter}" if scene_filter else " across all scenes"))
    target = find_object(args.find, args.obj_faiss, args.db, scene_filter,
                         model, tokenizer, device)
    if target is None:
        print("No matching object found.")
        return
    target_scene = target['scene']
    print(f"  Found: {target['name'].upper()} in scene={target_scene}  "
          f"pos=({target['tx']:.2f}, {target['ty']:.2f}, {target['tz']:.2f}) m")

    same_scene = (user_scene == target_scene)
    if not same_scene:
        print(f"  ⚠ Target is in a different room ({target_scene}) from user ({user_scene}).")
        print(f"    Will show target frame; path planning requires same scene.")

    # Use target scene for rendering (show where the object is)
    scene = target_scene

    # ── Step 4: plan path (only if same scene) ────────────────────────────────
    print("\nPlanning navigation path …", flush=True)
    paths_cfg = SCENES[user_scene]
    times_us, Rs_wd, ts_pos_list = load_trajectory(str(paths_cfg['traj']))

    user_pos   = np.array([best_loc['tx'], best_loc['ty'], best_loc['tz']])
    target_pos = np.array([target['tx'],   target['ty'],   target['tz']])

    if same_scene:
        waypoints, path_idxs = plan_path(user_pos, target_pos, ts_pos_list)
        path_pts = [waypoints[i] for i in path_idxs]
        total_dist = sum(np.linalg.norm(np.array(path_pts[i+1]) - np.array(path_pts[i]))
                         for i in range(len(path_pts)-1))
        print(f"  Path: {len(path_idxs)} waypoints, {total_dist:.1f} m total distance")
    else:
        waypoints, path_idxs, path_pts = [], [], []
        total_dist = np.linalg.norm(target_pos - user_pos)
        print(f"  Cross-scene: straight-line distance {total_dist:.1f} m (different rooms)")

    # ── Step 5: render ────────────────────────────────────────────────────────
    print("\nRendering …", flush=True)

    # Floor level = bottom face of target object (target_tz - half scale_z)
    # This is the Z the nav dots must be snapped to so they appear on the ground.
    floor_z = target['tz'] - target['scale_z'] / 2.0 - 0.05  # 5 cm below bottom face

    if same_scene:
        # Render from USER's re-localized frame — this is the natural AR view:
        # "here is what you see now, here is where to walk"
        render_cfg = SCENES[user_scene]
        frame_ts   = best_loc['ts_ns']
    else:
        # Cross-scene: show target's best frame so user knows what to look for
        render_cfg = SCENES[scene]
        frame_ts   = int(target['best_ts_ns'])

    render_traj_tus, render_Rs, render_ts_pos = load_trajectory(str(render_cfg['traj']))
    frame, provider = load_vrs_frame(render_cfg['vrs'], frame_ts)
    cam_calib, R_dc, t_dc, N = get_cam_calib(provider)
    R_wd, t_wd = interp_pose(render_traj_tus, render_Rs, render_ts_pos, frame_ts)

    # Draw context OBBs from snippet closest to this frame
    snips      = pd.read_csv(str(render_cfg['snips']))
    snip_times = snips['time_ns'].unique()
    closest_snip = snip_times[np.argmin(np.abs(snip_times - frame_ts))]
    frame_snips  = snips[snips['time_ns'] == closest_snip]

    for _, row in frame_snips.iterrows():
        corners_w = obb_corners_world(
            row['tx_world_object'], row['ty_world_object'], row['tz_world_object'],
            row['qw_world_object'], row['qx_world_object'],
            row['qy_world_object'], row['qz_world_object'],
            row['scale_x'], row['scale_y'], row['scale_z'],
        )
        draw_obb_fisheye(frame, corners_w, R_wd, t_wd, R_dc, t_dc,
                         cam_calib, get_color(row['name']), row['name'])

    # Draw target OBB (highlighted white)
    tgt_corners = obb_corners_world(
        target['tx'], target['ty'], target['tz'],
        target['qw'], target['qx'], target['qy'], target['qz'],
        target['scale_x'], target['scale_y'], target['scale_z'],
    )
    draw_obb_fisheye(frame, tgt_corners, R_wd, t_wd, R_dc, t_dc,
                     cam_calib, (255, 255, 255),
                     f">>> {target['name'].upper()} <<<",
                     highlight=True)

    # Draw navigation path + YOU marker (same scene only, floor-snapped)
    if same_scene and waypoints:
        project_path_on_frame(frame, waypoints, path_idxs,
                              R_wd, t_wd, R_dc, t_dc, cam_calib, N, floor_z)
        draw_user_marker(frame, R_wd, t_wd, R_dc, t_dc, cam_calib, N, floor_z)

    # HUD direction arrow — always drawn (works even when target is off-screen)
    draw_direction_hud(frame, R_wd, t_wd, R_dc, t_dc, target_pos, total_dist)

    # Banner
    room_note = "" if same_scene else f"  [user in {user_scene}]"
    banner = (f"Scene: {scene}{room_note}  |  Re-loc sim: {best_loc['sim']:.3f}  |  "
              f"Target: {target['name'].upper()}  |  Dist: {total_dist:.1f} m")
    cv2.rectangle(frame, (0,0), (frame.shape[1], 36), (20,20,20), -1)
    cv2.putText(frame, banner, (10,24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cv2.imwrite(args.out, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ── Final answer ──────────────────────────────────────────────────────────
    print(f"\n── Navigation answer ───────────────────────────────────────────")
    print(f"  You are at  : ({user_pos[0]:.2f}, {user_pos[1]:.2f}, {user_pos[2]:.2f}) m  "
          f"(scene={scene}, sim={best_loc['sim']:.3f})")
    print(f"  Target      : {target['name'].upper()} at "
          f"({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) m")
    print(f"  Distance    : {total_dist:.1f} m  ({len(path_idxs)} waypoints)")
    print(f"  Image       : {args.out}\n")

    # Print step-by-step directions
    if same_scene and len(path_pts) >= 2:
        print("  Step-by-step waypoints (world XYZ, m):")
        for k, pt in enumerate(path_pts):
            marker = " ← YOU" if k == 0 else (" ← TARGET" if k == len(path_pts)-1 else "")
            print(f"    [{k+1:2d}] ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}){marker}")
    elif not same_scene:
        print(f"  Navigate to {target_scene} first, then find {target['name'].upper()} "
              f"at ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) m")


def main():
    ap = argparse.ArgumentParser()
    # Query image (two ways to specify)
    ap.add_argument('--query-image',  default=None,
                    help='Path to a query image (simulate user\'s current view)')
    ap.add_argument('--query-ts',     type=int, default=None,
                    help='VRS timestamp (ns) to use as query frame')
    ap.add_argument('--query-scene',  default=None, choices=list(SCENES),
                    help='Scene for --query-ts (required if --query-ts is used)')
    # Object query
    ap.add_argument('--find', '-f',   required=True,
                    help='Natural language object query, e.g. "where is the sofa"')
    ap.add_argument('--scene',        default=None, choices=list(SCENES),
                    help='Restrict object search to this scene (default: use re-loc scene)')
    # Paths
    ap.add_argument('--kf-faiss',     default=DEFAULT_KF_FAISS)
    ap.add_argument('--kf-csv',       default=DEFAULT_KF_CSV)
    ap.add_argument('--obj-faiss',    default=DEFAULT_OBJ_FAISS)
    ap.add_argument('--db',           default=DEFAULT_DB)
    ap.add_argument('--out',          default=DEFAULT_OUT)
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
