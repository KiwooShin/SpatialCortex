"""
navigate.py — re-localize then navigate to a queried object.

Flow:
  query image  → CLIP image embed → keyframe FAISS → user pose (scene + 6-DoF)
  object query → CLIP text  embed → object  FAISS → target object pose
  user pose + target pose → Dijkstra on trajectory waypoints → ordered path
  render: fisheye frame at user position with 3D OBBs + projected nav path

Usage:
  # Build the keyframe index once:
  python scripts/detect/build_keyframe_index.py

  # Navigate (re-localize from an image):
  python scripts/navigate/navigate.py \\
      --query-image output/crops/seq01/004_sofa.jpg \\
      --find "where is the bed" \\
      --out output/nav_result.jpg

  # Use a VRS frame directly as the query:
  python scripts/navigate/navigate.py \\
      --query-ts 12000000000 --query-scene seq01 \\
      --find "where is the lamp" \\
      --out output/nav_result.jpg
"""

import argparse
import heapq
import os
import sqlite3
import sys
from pathlib import Path

# ── Package path (find spatialcortex/ at repo root) ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

os.environ.setdefault('HF_HUB_OFFLINE', '1')

import cv2
import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

from projectaria_tools.core import data_provider, sensor_data

from spatialcortex.config import (
    BASE, RGB_SID, SCENES,
    CLIP_MODEL, CLIP_PRETRAINED,
    DB_PATH, SCENE_FAISS, KEYFRAME_FAISS, KEYFRAME_CSV, OUTPUT_DIR,
)
from spatialcortex.geometry import (
    obb_corners_world, load_trajectory, interp_pose, project_pt,
)
from spatialcortex.drawing import get_color, draw_obb, draw_hud_arrow

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_KF_FAISS  = str(KEYFRAME_FAISS)
DEFAULT_KF_CSV    = str(KEYFRAME_CSV)
DEFAULT_OBJ_FAISS = str(SCENE_FAISS)
DEFAULT_DB        = str(DB_PATH)
DEFAULT_OUT       = str(OUTPUT_DIR / 'nav_result.jpg')
WAYPOINT_STEP_M   = 0.5

SYNONYMS = {
    "sit":     ["chair", "sofa"],
    "sleep":   ["bed", "sofa"],
    "light":   ["lamp"],
    "couch":   ["sofa"],
    "screen":  ["monitor", "tv"],
    "storage": ["cabinet", "shelf", "dresser"],
}

# ── CLIP ───────────────────────────────────────────────────────────────────────
_clip_cache: dict = {}

def load_clip(device: str):
    if 'model' not in _clip_cache:
        print("Loading CLIP …", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        _clip_cache.update({'model': model, 'pre': preprocess, 'tok': tokenizer})
    return _clip_cache['model'], _clip_cache['pre'], _clip_cache['tok']

@torch.no_grad()
def embed_image(model, preprocess, img_pil, device) -> np.ndarray:
    x    = preprocess(img_pil).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]

@torch.no_grad()
def embed_text(model, tokenizer, text: str, device) -> np.ndarray:
    tokens = tokenizer([text]).to(device)
    feat   = model.encode_text(tokens)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]

# ── Re-localization ────────────────────────────────────────────────────────────

def relocalize(query_img_pil, kf_index, kf_meta_df, model, preprocess, device,
               top_k: int = 3) -> list[dict]:
    """CLIP image embed → FAISS keyframe search → top-k matches with scene/pose."""
    vec = embed_image(model, preprocess, query_img_pil, device).reshape(1, -1).astype(np.float32)
    sims, idxs = kf_index.search(vec, top_k)
    results = []
    for rank, (idx, sim) in enumerate(zip(idxs[0], sims[0])):
        if idx < 0:
            continue
        row = kf_meta_df.iloc[int(idx)]
        results.append({
            "rank": rank + 1, "sim": float(sim),
            "scene": row["scene"], "ts_ns": int(row["ts_ns"]),
            "frame_idx": int(row["frame_idx"]),
            "tx": row["tx"], "ty": row["ty"], "tz": row["tz"],
            "qw": row["qw"], "qx": row["qx"], "qy": row["qy"], "qz": row["qz"],
        })
    return results

# ── Object retrieval ───────────────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    words = query.lower().split()
    return [syn for w in words for syn in SYNONYMS.get(w, [])]

def find_object(query: str, obj_faiss: str, db_path: str,
                scene_filter, model, tokenizer, device) -> dict | None:
    index = faiss.read_index(obj_faiss)
    conn  = sqlite3.connect(db_path)
    cols  = [r[1] for r in conn.execute("PRAGMA table_info(objects)").fetchall()]
    vec   = embed_text(model, tokenizer, query, device).reshape(1, -1).astype(np.float32)
    k     = min(50, index.ntotal)
    sims, idxs = index.search(vec, k)
    clip_ids = [int(i) for i in idxs[0] if i >= 0]
    ph   = ','.join('?' * len(clip_ids))
    rows = conn.execute(f"SELECT * FROM objects WHERE clip_idx IN ({ph})", clip_ids).fetchall()
    objs = [dict(zip(cols, r)) for r in rows]
    sim_map = {int(idxs[0][j]): float(sims[0][j]) for j in range(len(idxs[0]))}
    for o in objs:
        o['sim'] = sim_map.get(o['clip_idx'], 0.0)
    if scene_filter:
        objs = [o for o in objs if o['scene'] == scene_filter]
    qwords = set(query.lower().split()) | set(expand_query(query))
    def name_matches(name: str) -> bool:
        return bool(set(name.lower().replace('_', ' ').split()) & qwords)
    objs.sort(key=lambda o: (0 if name_matches(o['name']) else 1, -o['sim']))
    conn.close()
    return objs[0] if objs else None

# ── Dijkstra path planning ─────────────────────────────────────────────────────

def build_waypoints(ts_pos_list: list, step_m: float = 0.5) -> list:
    """Subsample the trajectory to ~step_m spacing."""
    if not ts_pos_list:
        return []
    waypoints = [ts_pos_list[0].copy()]
    for pt in ts_pos_list[1:]:
        if np.linalg.norm(pt - waypoints[-1]) >= step_m:
            waypoints.append(pt.copy())
    return waypoints

def dijkstra(waypoints: list, start_idx: int, end_idx: int) -> list[int]:
    """Shortest path on a kNN-5 waypoint graph."""
    from scipy.spatial import KDTree
    n = len(waypoints)
    if n == 0 or start_idx == end_idx:
        return [start_idx]
    pts   = np.array(waypoints)
    k_nn  = min(6, n)
    tree  = KDTree(pts)
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
        for i in range(1, k_nn):
            v  = int(nbrs_all[u, i])
            nd = dist[u] + float(dists_all[u, i])
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    path, cur = [], end_idx
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path if path[0] == start_idx else [start_idx, end_idx]

def plan_path(user_pos: np.ndarray, target_pos: np.ndarray,
              ts_pos_list: list, step_m: float = 0.5):
    waypoints = build_waypoints(ts_pos_list, step_m)
    if not waypoints:
        return [], []
    pts = np.array(waypoints)
    start_idx = int(np.argmin(np.linalg.norm(pts - user_pos, axis=1)))
    end_idx   = int(np.argmin(np.linalg.norm(pts - target_pos, axis=1)))
    return waypoints, dijkstra(waypoints, start_idx, end_idx)

# ── VRS frame loading ──────────────────────────────────────────────────────────

def load_vrs_frame(vrs_path, ts_ns: int):
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
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE), provider

def get_cam_calib(provider):
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib("camera-rgb")
    T_dc  = cam_calib.get_transform_device_camera()
    R_dc  = T_dc.rotation().to_matrix()
    t_dc  = T_dc.translation().flatten()
    img_w, _ = cam_calib.get_image_size()
    return cam_calib, R_dc, t_dc, int(img_w)

# ── Navigation rendering ───────────────────────────────────────────────────────

def project_path_on_frame(img, waypoints, path_idxs,
                           R_wd, t_wd, R_dc, t_dc, cam_calib, N, floor_z):
    """Draw nav path as floor-snapped dots + lines on the fisheye frame."""
    path_pts   = [waypoints[i] for i in path_idxs]
    prev_px    = None
    H, W       = img.shape[:2]
    PATH_COLOR = (0, 200, 255)
    DOT_COLOR  = (0, 255, 100)
    ARR_COLOR  = (0, 100, 255)

    for k, wp in enumerate(path_pts):
        wp_floor = np.array([wp[0], wp[1], floor_z])
        px = project_pt(wp_floor, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
        if px is None:
            prev_px = None
            continue
        xi, yi = int(round(px[0])), int(round(px[1]))
        in_img = 0 <= xi < W and 0 <= yi < H
        if in_img:
            r = 10 if k in (0, len(path_pts) - 1) else 6
            cv2.circle(img, (xi, yi), r, DOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(img, (xi, yi), r, (255, 255, 255), 1, cv2.LINE_AA)
            if k == 0:
                cv2.putText(img, "START", (xi + 12, yi + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            elif k == len(path_pts) - 1:
                cv2.putText(img, "DEST", (xi + 12, yi + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if prev_px is not None:
            x0, y0 = int(round(prev_px[0])), int(round(prev_px[1]))
            in_slack = lambda x, y: -60 <= x < W + 60 and -60 <= y < H + 60
            if not (in_slack(x0, y0) or in_slack(xi, yi)):
                prev_px = px
                continue
            color = ARR_COLOR if k == len(path_pts) - 1 else PATH_COLOR
            if k == len(path_pts) - 1:
                cv2.arrowedLine(img, (x0, y0), (xi, yi), color, 4, cv2.LINE_AA, tipLength=0.25)
            else:
                cv2.line(img, (x0, y0), (xi, yi), color, 4, cv2.LINE_AA)
        prev_px = px

def draw_user_marker(img, R_wd, t_wd, R_dc, t_dc, cam_calib, N, floor_z):
    """Project a YOU marker 0.5 m ahead of the camera at floor level."""
    forward_cam = np.array([0.0, 0.0, 0.5])
    p_dev       = R_dc @ forward_cam + t_dc
    p_world     = R_wd @ p_dev + t_wd
    p_world[2]  = floor_z
    px = project_pt(p_world, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
    H, W = img.shape[:2]
    if px is None or not (0 <= px[0] < W and 0 <= px[1] < H):
        xi, yi = W // 2, int(H * 0.82)
    else:
        xi, yi = int(round(px[0])), int(round(px[1]))
    cv2.circle(img, (xi, yi), 14, (0, 255, 200), -1, cv2.LINE_AA)
    cv2.circle(img, (xi, yi), 14, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "YOU", (xi - 16, yi - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading keyframe index …", flush=True)
    kf_index = faiss.read_index(args.kf_faiss)
    kf_meta  = pd.read_csv(args.kf_csv)
    print(f"  {kf_index.ntotal} keyframes in index")

    model, preprocess, tokenizer = load_clip(device)

    # Step 1: query image
    if args.query_image:
        print(f"\nLoading query image: {args.query_image}")
        query_pil = Image.open(args.query_image).convert('RGB')
    elif args.query_ts is not None and args.query_scene:
        print(f"\nLoading query frame from {args.query_scene} ts={args.query_ts}")
        arr, _ = load_vrs_frame(SCENES[args.query_scene]['vrs'], args.query_ts)
        query_pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Provide --query-image or (--query-ts + --query-scene)")

    # Step 2: re-localize
    print("\nRe-localizing …", flush=True)
    reloc_hits = relocalize(query_pil, kf_index, kf_meta, model, preprocess, device)
    if not reloc_hits:
        print("Re-localization failed.")
        return
    best_loc   = reloc_hits[0]
    user_scene = best_loc['scene']
    print(f"  Top match: scene={user_scene}  ts={best_loc['ts_ns']}  sim={best_loc['sim']:.3f}")
    print(f"  User pose: ({best_loc['tx']:.2f}, {best_loc['ty']:.2f}, {best_loc['tz']:.2f}) m")

    # Step 3: find object
    scene_filter = args.scene
    print(f"\nSearching: \"{args.find}\"" +
          (f" in scene={scene_filter}" if scene_filter else " across all scenes"))
    target = find_object(args.find, args.obj_faiss, args.db,
                         scene_filter, model, tokenizer, device)
    if target is None:
        print("No matching object found.")
        return
    target_scene = target['scene']
    print(f"  Found: {target['name'].upper()} in scene={target_scene}  "
          f"pos=({target['tx']:.2f}, {target['ty']:.2f}, {target['tz']:.2f}) m")

    same_scene = (user_scene == target_scene)
    if not same_scene:
        print(f"  ⚠ Target is in a different room ({target_scene}) from user ({user_scene}).")

    # Step 4: plan path
    paths_cfg = SCENES[user_scene]
    times_us, Rs_wd, ts_pos_list = load_trajectory(str(paths_cfg['traj']))
    user_pos   = np.array([best_loc['tx'], best_loc['ty'], best_loc['tz']])
    target_pos = np.array([target['tx'],   target['ty'],   target['tz']])

    if same_scene:
        waypoints, path_idxs = plan_path(user_pos, target_pos, ts_pos_list)
        path_pts  = [waypoints[i] for i in path_idxs]
        total_dist = sum(np.linalg.norm(np.array(path_pts[i+1]) - np.array(path_pts[i]))
                         for i in range(len(path_pts) - 1))
        print(f"  Path: {len(path_idxs)} waypoints, {total_dist:.1f} m total")
    else:
        waypoints, path_idxs, path_pts = [], [], []
        total_dist = float(np.linalg.norm(target_pos - user_pos))

    # Step 5: render
    floor_z = target['tz'] - target['scale_z'] / 2.0 - 0.05

    render_cfg = SCENES[user_scene if same_scene else target_scene]
    frame_ts   = best_loc['ts_ns'] if same_scene else int(target['best_ts_ns'])

    render_traj_tus, render_Rs, render_ts_pos = load_trajectory(str(render_cfg['traj']))
    frame, provider = load_vrs_frame(render_cfg['vrs'], frame_ts)
    cam_calib, R_dc, t_dc, N = get_cam_calib(provider)
    R_wd, t_wd = interp_pose(render_traj_tus, render_Rs, render_ts_pos, frame_ts)

    # Context OBBs from nearest snippet
    snips_df    = pd.read_csv(str(render_cfg['snippet_obbs']))
    snip_times  = snips_df['time_ns'].unique()
    closest_ts  = snip_times[np.argmin(np.abs(snip_times - frame_ts))]
    frame_snips = snips_df[snips_df['time_ns'] == closest_ts]
    for _, row in frame_snips.iterrows():
        corners_w = obb_corners_world(
            row['tx_world_object'], row['ty_world_object'], row['tz_world_object'],
            row['qw_world_object'], row['qx_world_object'],
            row['qy_world_object'], row['qz_world_object'],
            row['scale_x'], row['scale_y'], row['scale_z'],
        )
        draw_obb(frame, corners_w, R_wd, t_wd, R_dc, t_dc,
                 cam_calib, get_color(row['name']), row['name'], N)

    # Target OBB (highlighted)
    tgt_corners = obb_corners_world(
        target['tx'], target['ty'], target['tz'],
        target['qw'], target['qx'], target['qy'], target['qz'],
        target['scale_x'], target['scale_y'], target['scale_z'],
    )
    draw_obb(frame, tgt_corners, R_wd, t_wd, R_dc, t_dc,
             cam_calib, (255, 255, 255),
             f">>> {target['name'].upper()} <<<", N, highlight=True)

    if same_scene and waypoints:
        project_path_on_frame(frame, waypoints, path_idxs,
                              R_wd, t_wd, R_dc, t_dc, cam_calib, N, floor_z)
        draw_user_marker(frame, R_wd, t_wd, R_dc, t_dc, cam_calib, N, floor_z)

    draw_hud_arrow(frame, R_wd, t_wd, R_dc, t_dc, target_pos)

    room_note = "" if same_scene else f"  [user in {user_scene}]"
    banner = (f"Scene: {target_scene}{room_note}  |  "
              f"Re-loc sim: {best_loc['sim']:.3f}  |  "
              f"Target: {target['name'].upper()}  |  Dist: {total_dist:.1f} m")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(frame, banner, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n── Navigation answer ───────────────────────────────────────────")
    print(f"  You are at  : ({user_pos[0]:.2f}, {user_pos[1]:.2f}, {user_pos[2]:.2f}) m")
    print(f"  Target      : {target['name'].upper()} at "
          f"({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) m")
    print(f"  Distance    : {total_dist:.1f} m  ({len(path_idxs)} waypoints)")
    print(f"  Image       : {args.out}\n")
    if same_scene and len(path_pts) >= 2:
        print("  Waypoints (world XYZ, m):")
        for k, pt in enumerate(path_pts):
            tag = " ← YOU" if k == 0 else (" ← TARGET" if k == len(path_pts) - 1 else "")
            print(f"    [{k+1:2d}] ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}){tag}")
    elif not same_scene:
        print(f"  Navigate to {target_scene} first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query-image',  default=None)
    ap.add_argument('--query-ts',     type=int, default=None)
    ap.add_argument('--query-scene',  default=None, choices=list(SCENES))
    ap.add_argument('--find', '-f',   required=True)
    ap.add_argument('--scene',        default=None, choices=list(SCENES))
    ap.add_argument('--kf-faiss',     default=DEFAULT_KF_FAISS)
    ap.add_argument('--kf-csv',       default=DEFAULT_KF_CSV)
    ap.add_argument('--obj-faiss',    default=DEFAULT_OBJ_FAISS)
    ap.add_argument('--db',           default=DEFAULT_DB)
    ap.add_argument('--out',          default=DEFAULT_OUT)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
