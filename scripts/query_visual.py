"""
query_visual.py — visual spatial memory query.

Flow:
  text query
    → CLIP text embed → FAISS → best matching object in scene_db.sqlite
    → load best_ts_ns frame from VRS
    → draw 3D OBBs of target (highlighted) + nearby objects on frame
    → feed annotated frame to LLaVA
    → print: "The bed is at frame <ts>, near the dresser and lamp" + show image

Usage:
  python scripts/query_visual.py --query "where is the bed"
  python scripts/query_visual.py --query "find the lamp" --scene seq02
  python scripts/query_visual.py --query "where is the sofa" --out output/query_result.jpg
  python scripts/query_visual.py --interactive
"""

import argparse
import os
import sqlite3
import sys
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

# ── Scene registry ─────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent  # SpatialCortex/

SCENES = {
    "seq00": {
        "vrs":    BASE / "data/aeo/aeo_seq00_173376298563204/main.vrs",
        "traj":   BASE / "data/aeo/aeo_seq00_173376298563204/mps/mps/slam/closed_loop_trajectory.csv",
        "snips":  BASE / "output/efm3d_aeo_seq00/model_release/aeo_seq00_173376298563204/snippet_obbs.csv",
    },
    "seq01": {
        "vrs":    BASE / "data/aeo/aeo_seq01_208838848508107/main.vrs",
        "traj":   BASE / "data/aeo/aeo_seq01_208838848508107/mps/slam/closed_loop_trajectory.csv",
        "snips":  BASE / "output/efm3d_aeo_seq01/model_release/aeo_seq01_208838848508107/snippet_obbs.csv",
    },
    "seq02": {
        "vrs":    BASE / "data/aeo/aeo_seq02_181771578105956/main.vrs",
        "traj":   BASE / "data/aeo/aeo_seq02_181771578105956/mps/slam/closed_loop_trajectory.csv",
        "snips":  BASE / "output/efm3d_aeo_seq02/model_release/aeo_seq02_181771578105956/snippet_obbs.csv",
    },
}

# ── Config ─────────────────────────────────────────────────────────────────────

CLIP_MODEL      = 'ViT-L-14-quickgelu'
CLIP_PRETRAINED = 'openai'
DEFAULT_DB      = str(BASE / 'data/scene_db.sqlite')
DEFAULT_FAISS   = str(BASE / 'data/scene.faiss')
DEFAULT_OUT     = str(BASE / 'output/query_result.jpg')
VLM_MODEL       = 'llava-hf/llava-1.5-7b-hf'

from projectaria_tools.core import data_provider, sensor_data
from projectaria_tools.core.stream_id import StreamId

RGB_SID = StreamId(214, 1)

BB3D_LINE_ORDERS = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7],
]

_SSI_RGB = {
    "chair":(0.20,0.60,1.00),"sofa":(0.10,0.50,0.10),"table":(1.00,1.00,0.00),
    "shelf":(0.50,0.00,0.50),"lamp":(1.00,0.80,0.25),"bed":(0.90,0.40,0.60),
    "monitor":(0.00,0.80,0.80),"ladder":(0.50,0.80,0.30),"container":(0.80,0.50,0.20),
    "mirror":(0.60,0.70,1.00),"cabinet":(0.80,0.60,0.20),"tv":(0.30,0.90,0.70),
    "plant":(0.20,0.80,0.20),"microwave":(1.00,0.40,0.40),"refrigerator":(0.40,0.40,1.00),
    "oven":(0.70,0.60,0.30),"whiteboard":(0.90,0.90,0.70),"curtain":(0.70,0.50,1.00),
    "window":(0.70,0.90,1.00),"door":(0.80,0.70,0.60),"picture_frame":(1.00,0.60,0.20),
    "floor_mat":(0.60,0.40,0.20),"trash_can":(0.50,0.50,0.50),"book":(0.90,0.70,0.40),
    "bottle":(0.40,0.80,0.60),"dresser":(0.85,0.55,0.15),"cart":(0.60,0.80,0.40),
}
_DEFAULT_RGB = (0.70, 0.70, 0.70)

SYNONYMS = {
    "sit": ["chair","sofa"], "sleep": ["bed","sofa"],
    "light": ["lamp"], "couch": ["sofa"], "screen": ["monitor","tv"],
    "storage": ["cabinet","shelf","dresser"],
}

def _bgr255(r, g, b):
    return (int(b*255), int(g*255), int(r*255))

def get_color(name):
    return _bgr255(*_SSI_RGB.get(name.lower(), _DEFAULT_RGB))

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
    center = np.array([tx,ty,tz])
    hx,hy,hz = sx/2,sy/2,sz/2
    ids = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    xs,ys,zs = [-hx,hx],[-hy,hy],[-hz,hz]
    corners = np.array([[xs[xi],ys[yi],zs[zi]] for xi,yi,zi in ids])
    return center + corners @ R.T

def load_trajectory(traj_csv):
    df = pd.read_csv(traj_csv)
    times_us = df["tracking_timestamp_us"].values.astype(np.int64)
    Rs, ts = [], []
    for _, row in df.iterrows():
        R = quat_to_rotmat(row["qw_world_device"],row["qx_world_device"],
                           row["qy_world_device"],row["qz_world_device"])
        t = np.array([row["tx_world_device"],row["ty_world_device"],row["tz_world_device"]])
        Rs.append(R); ts.append(t)
    return times_us, Rs, ts

def interp_pose(times_us, Rs, ts, query_ns):
    query_us = query_ns // 1000
    idx = min(max(bisect_left(times_us, query_us), 0), len(times_us)-1)
    return Rs[idx], ts[idx]

def rotate_cw90(u, v, N):
    return N-1-v, u

def _project_pt(pw, R_wd, t_wd, R_dc, t_dc, cam_calib, N):
    p_dev = R_wd.T @ (pw - t_wd)
    pc = R_dc.T @ (p_dev - t_dc)
    if pc[2] <= 0.05:
        return None
    uv = cam_calib.project(pc)
    if uv is None:
        return None
    u_rot, v_rot = rotate_cw90(uv[0], uv[1], N)
    return float(u_rot), float(v_rot)

def draw_obb_fisheye(img, corners_world, R_wd, t_wd, R_dc, t_dc,
                     cam_calib, color, label, n_samples=12,
                     alpha=0.18, thickness=2, highlight=False):
    H, W = img.shape[:2]
    N = W

    corners_px = [_project_pt(pw, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
                  for pw in corners_world]

    if highlight and all(p is not None for p in corners_px):
        pts = np.array(corners_px, dtype=np.int32)
        if (pts[:,0].min()>=0 and pts[:,0].max()<W and
                pts[:,1].min()>=0 and pts[:,1].max()<H):
            overlay = img.copy()
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(overlay, hull, color)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    elif all(p is not None for p in corners_px):
        pts = np.array(corners_px, dtype=np.int32)
        if (pts[:,0].min()>=0 and pts[:,0].max()<W and
                pts[:,1].min()>=0 and pts[:,1].max()<H):
            overlay = img.copy()
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(overlay, hull, color)
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    any_drawn = False
    lw = 4 if highlight else thickness
    for i, j in BB3D_LINE_ORDERS:
        ts_edge = np.linspace(0,1,n_samples)
        pts3d = [corners_world[i]+t*(corners_world[j]-corners_world[i]) for t in ts_edge]
        pts2d = [_project_pt(p, R_wd, t_wd, R_dc, t_dc, cam_calib, N) for p in pts3d]
        for k in range(len(pts2d)-1):
            p0,p1 = pts2d[k], pts2d[k+1]
            if p0 is None or p1 is None: continue
            x0,y0 = int(round(p0[0])),int(round(p0[1]))
            x1,y1 = int(round(p1[0])),int(round(p1[1]))
            in_slack = lambda x,y: -60<=x<W+60 and -60<=y<H+60
            if not (in_slack(x0,y0) or in_slack(x1,y1)): continue
            cv2.line(img,(x0,y0),(x1,y1),color,lw,cv2.LINE_AA)
            any_drawn = True

    if any_drawn and label:
        valid = [p for p in corners_px if p is not None]
        if valid:
            cx = int(np.mean([p[0] for p in valid]))
            cy = int(np.mean([p[1] for p in valid]))
            cx = max(2, min(cx, W-120)); cy = max(16, min(cy, H-4))
            font, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.60 if highlight else 0.48
            th = 2 if highlight else 1
            (tw,tht),_ = cv2.getTextSize(label, font, sc, th)
            cv2.rectangle(img,(cx-3,cy-tht-4),(cx+tw+3,cy+3),(10,10,10),-1)
            cv2.putText(img,label,(cx,cy),font,sc,color,th,cv2.LINE_AA)
            if highlight:
                # Arrow pointing to the label from slightly above
                cv2.arrowedLine(img,(cx+tw//2,cy-tht-20),(cx+tw//2,cy-tht-5),
                                color,2,cv2.LINE_AA,tipLength=0.4)

    return any_drawn

# ── CLIP ───────────────────────────────────────────────────────────────────────

_clip_cache = {}

def load_clip(device):
    if 'model' not in _clip_cache:
        print("Loading CLIP …", flush=True)
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        _clip_cache['model'] = model
        _clip_cache['tokenizer'] = tokenizer
    return _clip_cache['model'], _clip_cache['tokenizer']

@torch.no_grad()
def embed_text(model, tokenizer, text, device):
    tokens = tokenizer([text]).to(device)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]

# ── VLM ────────────────────────────────────────────────────────────────────────

_vlm_cache = {}

def load_vlm():
    if 'model' not in _vlm_cache:
        print(f"Loading LLaVA …", flush=True)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(VLM_MODEL)
        model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL, torch_dtype=torch.float16, device_map='auto')
        model.eval()
        _vlm_cache['model'] = model
        _vlm_cache['proc'] = proc
        print("LLaVA loaded.", flush=True)
    return _vlm_cache['model'], _vlm_cache['proc']

def vlm_describe(annotated_img_pil, target_name, nearby_names, query):
    """
    Feed the full annotated frame to LLaVA.
    Ask it to describe WHERE the target object is and what's around it.
    Returns a natural language string.
    """
    model, processor = load_vlm()

    nearby_str = ", ".join(nearby_names) if nearby_names else "nothing nearby"
    prompt = (
        f"USER: <image>\n"
        f"This is a frame from a room scan. "
        f"The {target_name} is highlighted with a bright bounding box. "
        f"Nearby objects detected: {nearby_str}.\n"
        f"In one or two sentences, describe where the {target_name} is in the room "
        f"and what you can see around it.\n"
        f"ASSISTANT:"
    )

    # Resize to max 672px (LLaVA's native resolution) to reduce hallucination
    w, h = annotated_img_pil.size
    max_sz = 672
    if max(w,h) > max_sz:
        scale = max_sz / max(w,h)
        annotated_img_pil = annotated_img_pil.resize(
            (int(w*scale), int(h*scale)), Image.LANCZOS)

    inputs = processor(text=prompt, images=annotated_img_pil,
                       return_tensors='pt').to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                 temperature=1.0)
    new_ids = out_ids[0][inputs['input_ids'].shape[1]:]
    reply = processor.decode(new_ids, skip_special_tokens=True).strip()
    return reply

# ── Query ──────────────────────────────────────────────────────────────────────

def expand_query(query):
    """Return extra class names implied by the query."""
    words = query.lower().split()
    extras = []
    for word in words:
        extras.extend(SYNONYMS.get(word, []))
    return extras

def retrieve_object(query, db_path, faiss_path, scene_filter, top_k, device):
    """CLIP text → FAISS → SQLite → best matching object dict."""
    index = faiss.read_index(faiss_path)
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(objects)").fetchall()]

    model, tokenizer = load_clip(device)
    vec = embed_text(model, tokenizer, query, device).reshape(1,-1).astype(np.float32)

    k = min(top_k * 10, index.ntotal)
    sims, idx = index.search(vec, k)
    clip_ids = [int(i) for i in idx[0] if i >= 0]

    # Fetch all matching rows
    placeholders = ','.join('?'*len(clip_ids))
    rows = conn.execute(
        f"SELECT * FROM objects WHERE clip_idx IN ({placeholders})", clip_ids
    ).fetchall()
    objects = [dict(zip(cols, r)) for r in rows]

    # Attach similarity scores
    sim_map = {int(idx[0][j]): float(sims[0][j]) for j in range(len(idx[0]))}
    for o in objects:
        o['sim'] = sim_map.get(o['clip_idx'], 0.0)

    # Filter by scene
    if scene_filter:
        objects = [o for o in objects if o['scene'] == scene_filter]

    # Two-bucket ranking: name match first, then CLIP sim
    # Use exact word matching (not substring) to avoid "a" in "ladder" false positives
    qwords = set(query.lower().split()) | set(expand_query(query))
    def name_matches(obj_name):
        name_words = set(obj_name.lower().replace('_', ' ').split())
        return bool(name_words & qwords)  # any object word appears in query words
    def rank_key(o):
        bucket = 0 if name_matches(o['name']) else 1
        return (bucket, -o['sim'])
    objects.sort(key=rank_key)

    # Also add cropless objects that match by name (SQLite fallback)
    name_match_ids = set(o['id'] for o in objects)
    fallback = conn.execute(
        "SELECT * FROM objects WHERE has_image=0"
    ).fetchall()
    for r in fallback:
        o = dict(zip(cols, r))
        if scene_filter and o['scene'] != scene_filter:
            continue
        if any(w in o['name'].lower() for w in qwords) and o['id'] not in name_match_ids:
            o['sim'] = 0.0
            objects.append(o)

    conn.close()
    return objects[:top_k]

def nearby_names(conn, obj, radius_m=2.0):
    rows = conn.execute("""
        SELECT name FROM objects
        WHERE scene=? AND id!=?
          AND ((tx-?)*(tx-?)+(ty-?)*(ty-?)+(tz-?)*(tz-?)) < ?
        ORDER BY ((tx-?)*(tx-?)+(ty-?)*(ty-?)+(tz-?)*(tz-?))
        LIMIT 5
    """, (
        obj['scene'], obj['id'],
        obj['tx'],obj['tx'], obj['ty'],obj['ty'], obj['tz'],obj['tz'], radius_m**2,
        obj['tx'],obj['tx'], obj['ty'],obj['ty'], obj['tz'],obj['tz'],
    )).fetchall()
    return [r[0] for r in rows]

# ── Frame rendering ────────────────────────────────────────────────────────────

def render_query_frame(target_obj, conn, out_path):
    """
    Load the best VRS frame for target_obj, draw 3D OBBs (target highlighted,
    nearby objects normal), save to out_path. Returns PIL image.
    """
    scene = target_obj['scene']
    if scene not in SCENES:
        raise ValueError(f"Unknown scene: {scene}")

    paths = SCENES[scene]
    ts_ns = int(target_obj['best_ts_ns'])

    # Load trajectory
    times_us, Rs_wd, ts_wd = load_trajectory(str(paths['traj']))
    R_wd, t_wd = interp_pose(times_us, Rs_wd, ts_wd, ts_ns)

    # Open VRS
    provider = data_provider.create_vrs_data_provider(str(paths['vrs']))
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib("camera-rgb")
    T_dc = cam_calib.get_transform_device_camera()
    R_dc = T_dc.rotation().to_matrix()
    t_dc = T_dc.translation().flatten()
    img_w, _ = cam_calib.get_image_size()
    N = int(img_w)

    # Load frame
    img_data, _ = provider.get_image_data_by_time_ns(
        RGB_SID, ts_ns,
        sensor_data.TimeDomain.DEVICE_TIME,
        sensor_data.TimeQueryOptions.CLOSEST,
    )
    if not img_data.is_valid():
        raise RuntimeError(f"No image at ts={ts_ns}")

    frame = img_data.to_numpy_array().copy()
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame[:,:,0], cv2.COLOR_GRAY2BGR)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Load snippet OBBs at this timestamp to draw context objects
    snips = pd.read_csv(str(paths['snips']))
    # Find the closest timestamp in snippet OBBs to our target ts_ns
    snip_times = snips['time_ns'].unique()
    closest_snip_ts = snip_times[np.argmin(np.abs(snip_times - ts_ns))]
    frame_snips = snips[snips['time_ns'] == closest_snip_ts]

    # Draw context objects from snippets (normal color, thin)
    target_name = target_obj['name'].lower()
    for _, row in frame_snips.iterrows():
        if row['name'].lower() == target_name:
            continue  # draw target last (on top)
        corners_w = obb_corners_world(
            row['tx_world_object'], row['ty_world_object'], row['tz_world_object'],
            row['qw_world_object'], row['qx_world_object'],
            row['qy_world_object'], row['qz_world_object'],
            row['scale_x'], row['scale_y'], row['scale_z'],
        )
        color = get_color(row['name'])
        label = f"{row['name']}"
        draw_obb_fisheye(frame, corners_w, R_wd, t_wd, R_dc, t_dc,
                         cam_calib, color, label, highlight=False)

    # Draw the TARGET object (use its scene-fused position, highlighted)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(objects)").fetchall()]
    target_corners = obb_corners_world(
        target_obj['tx'], target_obj['ty'], target_obj['tz'],
        target_obj['qw'], target_obj['qx'], target_obj['qy'], target_obj['qz'],
        target_obj['scale_x'], target_obj['scale_y'], target_obj['scale_z'],
    )
    highlight_color = (255, 255, 255)  # bright white for target — stands out from all class colors
    draw_obb_fisheye(frame, target_corners, R_wd, t_wd, R_dc, t_dc,
                     cam_calib, highlight_color,
                     f">>> {target_obj['name'].upper()} <<<",
                     highlight=True, alpha=0.30)

    # Overlay: frame info banner at top
    banner = f"Scene: {scene}  |  ts: {ts_ns}  |  Query target: {target_obj['name'].upper()}"
    cv2.rectangle(frame, (0,0), (frame.shape[1], 36), (20,20,20), -1)
    cv2.putText(frame, banner, (10,24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

    # Save
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Return as PIL for LLaVA
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ── Main query function ────────────────────────────────────────────────────────

def run_query(query, db_path, faiss_path, scene_filter, use_vlm, out_path, device):
    print(f'\nSearching: "{query}"', flush=True)

    candidates = retrieve_object(query, db_path, faiss_path, scene_filter,
                                 top_k=5, device=device)
    if not candidates:
        print("  No results found.")
        return

    # Pick top result
    best = candidates[0]
    conn = sqlite3.connect(db_path)
    nearby = nearby_names(conn, best)

    print(f"\n  Found: {best['name'].upper()}  (scene={best['scene']})")
    print(f"  Position : ({best['tx']:.2f}, {best['ty']:.2f}, {best['tz']:.2f}) m")
    print(f"  Size     : {best['scale_x']:.2f}×{best['scale_y']:.2f}×{best['scale_z']:.2f} m")
    print(f"  Confidence: {best['prob']:.2f}  ({best['count']} observations)")
    print(f"  Nearby   : {', '.join(nearby) if nearby else 'none'}")
    print(f"  Frame ts : {best['best_ts_ns']}")

    # Render annotated frame
    print(f"\n  Rendering annotated frame …", flush=True)
    try:
        pil_frame = render_query_frame(best, conn, out_path)
        print(f"  Saved    : {out_path}")
    except Exception as e:
        print(f"  Frame render failed: {e}")
        conn.close()
        return

    conn.close()

    # Always print structured frame info
    nearby_str = ", ".join(nearby) if nearby else "none"
    ts_ns      = best['best_ts_ns']
    ts_s       = ts_ns / 1e9
    print(f"\n── Answer ─────────────────────────────────────────────────────")
    print(f"  Object   : {best['name'].upper()}")
    print(f"  Scene    : {best['scene']}")
    print(f"  Frame ts : {ts_ns}  ({ts_s:.3f} s into recording)")
    print(f"  Position : ({best['tx']:.2f}, {best['ty']:.2f}, {best['tz']:.2f}) m")
    print(f"  Nearby   : {nearby_str}")
    print(f"  Image    : {out_path}")

    # LLaVA visual description (appended below the frame info)
    if use_vlm:
        try:
            reply = vlm_describe(pil_frame, best['name'], nearby, query)
            print(f"\n  LLaVA: \"{reply}\"\n")
        except Exception as e:
            print(f"\n  LLaVA error: {e}\n")

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query',  '-q', default=None)
    ap.add_argument('--scene',        default=None, choices=list(SCENES)+[None])
    ap.add_argument('--db',           default=DEFAULT_DB)
    ap.add_argument('--faiss',        default=DEFAULT_FAISS)
    ap.add_argument('--out',          default=DEFAULT_OUT,
                    help='Output image path (default: output/query_result.jpg)')
    ap.add_argument('--vlm',          action='store_true',
                    help='Use LLaVA to describe the annotated frame')
    ap.add_argument('--no-vlm',       action='store_true',
                    help='Skip VLM, use programmatic description only')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_vlm = args.vlm and not args.no_vlm

    if args.query:
        run_query(args.query, args.db, args.faiss, args.scene,
                  use_vlm, args.out, device)
    else:
        # Interactive loop
        print("Spatial Visual Query  (Ctrl-C to exit)")
        print(f"VLM: {'on' if use_vlm else 'off'}  |  scenes: {list(SCENES.keys())}")
        i = 0
        while True:
            try:
                q = input("\nQuery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not q:
                continue
            out = args.out.replace('.jpg', f'_{i:03d}.jpg')
            run_query(q, args.db, args.faiss, args.scene,
                      use_vlm, out, device)
            i += 1

if __name__ == '__main__':
    main()
