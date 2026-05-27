"""
query_visual.py — visual spatial memory query.

Flow:
  text query
    → CLIP text embed → FAISS → best matching object in scene_db.sqlite
    → load best_ts_ns frame from VRS
    → draw 3D OBBs of target (highlighted) + nearby objects on frame
    → feed annotated frame to LLaVA
    → print structured answer + image path

Usage:
  python scripts/query/query_visual.py --query "where is the bed"
  python scripts/query/query_visual.py --query "find the lamp" --scene seq02
  python scripts/query/query_visual.py --interactive --vlm
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

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
    DB_PATH, SCENE_FAISS, OUTPUT_DIR,
)
from spatialcortex.geometry import obb_corners_world, load_trajectory, interp_pose
from spatialcortex.drawing import get_color, draw_obb

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DB    = str(DB_PATH)
DEFAULT_FAISS = str(SCENE_FAISS)
DEFAULT_OUT   = str(OUTPUT_DIR / 'query_result.jpg')
VLM_MODEL     = 'llava-hf/llava-1.5-7b-hf'

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
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        _clip_cache['model'] = model
        _clip_cache['tokenizer'] = tokenizer
    return _clip_cache['model'], _clip_cache['tokenizer']

@torch.no_grad()
def embed_text(model, tokenizer, text: str, device) -> np.ndarray:
    tokens = tokenizer([text]).to(device)
    feat   = model.encode_text(tokens)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]

# ── VLM ───────────────────────────────────────────────────────────────────────
_vlm_cache: dict = {}

def load_vlm():
    if 'model' not in _vlm_cache:
        print("Loading LLaVA …", flush=True)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc  = AutoProcessor.from_pretrained(VLM_MODEL)
        model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL, torch_dtype=torch.float16, device_map='auto')
        model.eval()
        _vlm_cache['model'] = model
        _vlm_cache['proc']  = proc
        print("LLaVA loaded.", flush=True)
    return _vlm_cache['model'], _vlm_cache['proc']

def vlm_describe(annotated_img_pil, target_name: str, nearby_names: list,
                 query: str) -> str:
    """Feed the annotated frame to LLaVA; return a natural language description."""
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
    w, h = annotated_img_pil.size
    max_sz = 672
    if max(w, h) > max_sz:
        scale = max_sz / max(w, h)
        annotated_img_pil = annotated_img_pil.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS)
    inputs = processor(text=prompt, images=annotated_img_pil,
                       return_tensors='pt').to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                                 temperature=1.0)
    new_ids = out_ids[0][inputs['input_ids'].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()

# ── Retrieval ─────────────────────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    return [syn for w in query.lower().split() for syn in SYNONYMS.get(w, [])]

def retrieve_object(query: str, db_path: str, faiss_path: str,
                    scene_filter, top_k: int, device) -> list[dict]:
    """CLIP text → FAISS → SQLite → ranked list of matching object dicts."""
    index = faiss.read_index(faiss_path)
    conn  = sqlite3.connect(db_path)
    cols  = [r[1] for r in conn.execute("PRAGMA table_info(objects)").fetchall()]

    model, tokenizer = load_clip(device)
    vec = embed_text(model, tokenizer, query, device).reshape(1, -1).astype(np.float32)
    k   = min(top_k * 10, index.ntotal)
    sims, idx = index.search(vec, k)
    clip_ids  = [int(i) for i in idx[0] if i >= 0]

    ph   = ','.join('?' * len(clip_ids))
    rows = conn.execute(f"SELECT * FROM objects WHERE clip_idx IN ({ph})", clip_ids).fetchall()
    objects = [dict(zip(cols, r)) for r in rows]
    sim_map = {int(idx[0][j]): float(sims[0][j]) for j in range(len(idx[0]))}
    for o in objects:
        o['sim'] = sim_map.get(o['clip_idx'], 0.0)

    if scene_filter:
        objects = [o for o in objects if o['scene'] == scene_filter]

    # Two-bucket ranking: exact word match first, then CLIP similarity
    qwords = set(query.lower().split()) | set(expand_query(query))
    def name_matches(name: str) -> bool:
        return bool(set(name.lower().replace('_', ' ').split()) & qwords)
    objects.sort(key=lambda o: (0 if name_matches(o['name']) else 1, -o['sim']))

    # SQLite fallback: cropless objects matched by class name
    seen_ids = {o['id'] for o in objects}
    for r in conn.execute("SELECT * FROM objects WHERE has_image=0").fetchall():
        o = dict(zip(cols, r))
        if scene_filter and o['scene'] != scene_filter:
            continue
        if any(w in o['name'].lower() for w in qwords) and o['id'] not in seen_ids:
            o['sim'] = 0.0
            objects.append(o)

    conn.close()
    return objects[:top_k]

def nearby_names(conn, obj, radius_m: float = 2.0) -> list[str]:
    rows = conn.execute("""
        SELECT name FROM objects
        WHERE scene=? AND id!=?
          AND ((tx-?)*(tx-?)+(ty-?)*(ty-?)+(tz-?)*(tz-?)) < ?
        ORDER BY ((tx-?)*(tx-?)+(ty-?)*(ty-?)+(tz-?)*(tz-?))
        LIMIT 5
    """, (
        obj['scene'], obj['id'],
        obj['tx'], obj['tx'], obj['ty'], obj['ty'], obj['tz'], obj['tz'], radius_m**2,
        obj['tx'], obj['tx'], obj['ty'], obj['ty'], obj['tz'], obj['tz'],
    )).fetchall()
    return [r[0] for r in rows]

# ── Frame rendering ────────────────────────────────────────────────────────────

def render_query_frame(target_obj: dict, conn, out_path: str) -> Image.Image:
    """Load VRS frame for target_obj, draw annotated OBBs, save to out_path."""
    scene = target_obj['scene']
    paths = SCENES[scene]
    ts_ns = int(target_obj['best_ts_ns'])

    times_us, Rs_wd, ts_wd = load_trajectory(str(paths['traj']))
    R_wd, t_wd = interp_pose(times_us, Rs_wd, ts_wd, ts_ns)

    provider = data_provider.create_vrs_data_provider(str(paths['vrs']))
    dev_calib = provider.get_device_calibration()
    cam_calib = dev_calib.get_camera_calib("camera-rgb")
    T_dc  = cam_calib.get_transform_device_camera()
    R_dc  = T_dc.rotation().to_matrix()
    t_dc  = T_dc.translation().flatten()
    img_w, _ = cam_calib.get_image_size()
    N = int(img_w)

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
        frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Context OBBs from nearest snippet timestamp
    snips_df    = pd.read_csv(str(paths['snippet_obbs']))
    snip_times  = snips_df['time_ns'].unique()
    closest_ts  = snip_times[np.argmin(np.abs(snip_times - ts_ns))]
    frame_snips = snips_df[snips_df['time_ns'] == closest_ts]
    target_name = target_obj['name'].lower()

    for _, row in frame_snips.iterrows():
        if row['name'].lower() == target_name:
            continue  # draw target last
        corners_w = obb_corners_world(
            row['tx_world_object'], row['ty_world_object'], row['tz_world_object'],
            row['qw_world_object'], row['qx_world_object'],
            row['qy_world_object'], row['qz_world_object'],
            row['scale_x'], row['scale_y'], row['scale_z'],
        )
        draw_obb(frame, corners_w, R_wd, t_wd, R_dc, t_dc,
                 cam_calib, get_color(row['name']), row['name'], N)

    # Target OBB — highlighted white
    tgt_corners = obb_corners_world(
        target_obj['tx'], target_obj['ty'], target_obj['tz'],
        target_obj['qw'], target_obj['qx'], target_obj['qy'], target_obj['qz'],
        target_obj['scale_x'], target_obj['scale_y'], target_obj['scale_z'],
    )
    draw_obb(frame, tgt_corners, R_wd, t_wd, R_dc, t_dc,
             cam_calib, (255, 255, 255),
             f">>> {target_obj['name'].upper()} <<<", N,
             lw=4, highlight=True)

    banner = (f"Scene: {scene}  |  ts: {ts_ns}  |  "
              f"Query target: {target_obj['name'].upper()}")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(frame, banner, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ── Query runner ──────────────────────────────────────────────────────────────

def run_query(query: str, db_path: str, faiss_path: str, scene_filter,
              use_vlm: bool, out_path: str, device: str):
    print(f'\nSearching: "{query}"', flush=True)
    candidates = retrieve_object(query, db_path, faiss_path, scene_filter,
                                 top_k=5, device=device)
    if not candidates:
        print("  No results found.")
        return

    best = candidates[0]
    conn = sqlite3.connect(db_path)
    nearby = nearby_names(conn, best)

    print(f"  Found: {best['name'].upper()}  (scene={best['scene']})")
    print(f"  Position : ({best['tx']:.2f}, {best['ty']:.2f}, {best['tz']:.2f}) m")
    print(f"  Nearby   : {', '.join(nearby) if nearby else 'none'}")

    print("\n  Rendering annotated frame …", flush=True)
    try:
        pil_frame = render_query_frame(best, conn, out_path)
        print(f"  Saved    : {out_path}")
    except Exception as e:
        print(f"  Frame render failed: {e}")
        conn.close()
        return
    conn.close()

    ts_ns = best['best_ts_ns']
    print(f"\n── Answer ─────────────────────────────────────────────────────")
    print(f"  Object   : {best['name'].upper()}")
    print(f"  Scene    : {best['scene']}")
    print(f"  Frame ts : {ts_ns}  ({ts_ns / 1e9:.3f} s into recording)")
    print(f"  Position : ({best['tx']:.2f}, {best['ty']:.2f}, {best['tz']:.2f}) m")
    print(f"  Nearby   : {', '.join(nearby) if nearby else 'none'}")
    print(f"  Image    : {out_path}")

    if use_vlm:
        try:
            reply = vlm_describe(pil_frame, best['name'], nearby, query)
            print(f"\n  LLaVA: \"{reply}\"\n")
        except Exception as e:
            print(f"\n  LLaVA error: {e}\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query',  '-q', default=None)
    ap.add_argument('--scene',        default=None, choices=list(SCENES) + [None])
    ap.add_argument('--db',           default=DEFAULT_DB)
    ap.add_argument('--faiss',        default=DEFAULT_FAISS)
    ap.add_argument('--out',          default=DEFAULT_OUT)
    ap.add_argument('--vlm',          action='store_true')
    ap.add_argument('--no-vlm',       action='store_true')
    ap.add_argument('--interactive',  action='store_true')
    args = ap.parse_args()

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_vlm = args.vlm and not args.no_vlm

    if args.query or not args.interactive:
        q = args.query or input("Query> ").strip()
        run_query(q, args.db, args.faiss, args.scene, use_vlm, args.out, device)
    else:
        print("Spatial Visual Query  (Ctrl-C to exit)")
        i = 0
        while True:
            try:
                q = input("\nQuery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if q:
                out = args.out.replace('.jpg', f'_{i:03d}.jpg')
                run_query(q, args.db, args.faiss, args.scene, use_vlm, out, device)
                i += 1

if __name__ == '__main__':
    main()
