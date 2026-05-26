"""
build_scene_db.py — encode object crops with CLIP and persist to SQLite + FAISS.

Reads scene_obbs_crops.csv files from one or more sequences, encodes each
valid crop with CLIP ViT-L/14 (openai weights), normalises embeddings for
cosine similarity, and writes:
  - data/scene_db.sqlite   (objects table: metadata + clip_idx)
  - data/scene.faiss       (IndexFlatIP — exact cosine search)

Usage:
  python scripts/build_scene_db.py \
      --crops  output/efm3d_aeo_seq00/.../scene_obbs_crops.csv \
               output/efm3d_aeo_seq01/.../scene_obbs_crops.csv \
               output/efm3d_aeo_seq02/.../scene_obbs_crops.csv \
      --db     data/scene_db.sqlite \
      --faiss  data/scene.faiss
"""

import argparse
import os
import sqlite3
import time

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

CLIP_MODEL  = 'ViT-L-14'
CLIP_PRETRAINED = 'openai'
EMBED_DIM   = 768   # ViT-L-14 output dimension


# ── Database setup ────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS objects (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    scene     TEXT    NOT NULL,
    obj_id    INTEGER NOT NULL,
    name      TEXT    NOT NULL,
    prob      REAL,
    count     INTEGER,
    tx        REAL,
    ty        REAL,
    tz        REAL,
    qw        REAL,
    qx        REAL,
    qy        REAL,
    qz        REAL,
    scale_x   REAL,
    scale_y   REAL,
    scale_z   REAL,
    crop_path TEXT,
    best_ts_ns INTEGER,
    clip_idx  INTEGER
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(DDL)
    conn.commit()
    return conn


def insert_object(conn, row: dict, clip_idx: int):
    conn.execute("""
        INSERT INTO objects
            (scene, obj_id, name, prob, count,
             tx, ty, tz, qw, qx, qy, qz,
             scale_x, scale_y, scale_z,
             crop_path, best_ts_ns, clip_idx)
        VALUES
            (:scene, :obj_id, :name, :prob, :count,
             :tx, :ty, :tz, :qw, :qx, :qy, :qz,
             :scale_x, :scale_y, :scale_z,
             :crop_path, :best_ts_ns, :clip_idx)
    """, {**row, 'clip_idx': clip_idx})


# ── CLIP encoding ─────────────────────────────────────────────────────────────

def load_clip(device: str):
    print(f"Loading {CLIP_MODEL}/{CLIP_PRETRAINED} …", flush=True)
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_image(model, preprocess, img_path: str, device: str) -> np.ndarray:
    """Return L2-normalised 768-dim embedding, or None on error."""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"    WARNING: cannot open {img_path}: {e}")
        return None
    tensor = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(tensor)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]   # (768,)


@torch.no_grad()
def encode_text(model, tokenizer, text: str, device: str) -> np.ndarray:
    """Return L2-normalised text embedding."""
    tokens = tokenizer([text]).to(device)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', nargs='+', required=True,
                    help='one or more scene_obbs_crops.csv paths')
    ap.add_argument('--db',    default='data/scene_db.sqlite')
    ap.add_argument('--faiss', default='data/scene.faiss')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.db)),    exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.faiss)), exist_ok=True)

    # Remove existing DB so we rebuild clean
    if os.path.exists(args.db):
        os.remove(args.db)
        print(f"Removed existing {args.db}")

    model, preprocess, tokenizer = load_clip(args.device)
    print(f"Device: {args.device}")

    conn   = init_db(args.db)
    index  = faiss.IndexFlatIP(EMBED_DIM)   # cosine on normalised vectors

    total_inserted = 0
    total_skipped  = 0

    for csv_path in args.crops:
        df = pd.read_csv(csv_path)
        scene = df['scene'].iloc[0] if 'scene' in df.columns else os.path.basename(csv_path)
        print(f"\n── {scene}  ({len(df)} objects from {csv_path})")

        for _, row in df.iterrows():
            crop_path = str(row.get('crop_path', ''))

            # Objects without a valid crop get a text-only embedding from class name
            if not crop_path or not os.path.exists(crop_path):
                print(f"  [{int(row['obj_id']):03d}] {row['name']:20s}  no crop → text embedding")
                emb = encode_text(model, tokenizer, row['name'], args.device)
            else:
                emb = encode_image(model, preprocess, crop_path, args.device)
                if emb is None:
                    total_skipped += 1
                    continue
                print(f"  [{int(row['obj_id']):03d}] {row['name']:20s}  {crop_path.split('/')[-1]}")

            clip_idx = index.ntotal
            index.add(emb.reshape(1, -1).astype(np.float32))

            rec = {
                'scene':      scene,
                'obj_id':     int(row['obj_id']),
                'name':       row['name'],
                'prob':       float(row['prob']),
                'count':      int(row['count']),
                'tx':         float(row['tx_world_object']),
                'ty':         float(row['ty_world_object']),
                'tz':         float(row['tz_world_object']),
                'qw':         float(row['qw_world_object']),
                'qx':         float(row['qx_world_object']),
                'qy':         float(row['qy_world_object']),
                'qz':         float(row['qz_world_object']),
                'scale_x':    float(row['scale_x']),
                'scale_y':    float(row['scale_y']),
                'scale_z':    float(row['scale_z']),
                'crop_path':  crop_path,
                'best_ts_ns': int(row['best_ts_ns']) if 'best_ts_ns' in row else -1,
            }
            insert_object(conn, rec, clip_idx)
            total_inserted += 1

    conn.commit()
    conn.close()

    faiss.write_index(index, args.faiss)

    print(f"\n{'─'*50}")
    print(f"Objects inserted : {total_inserted}")
    print(f"Objects skipped  : {total_skipped}")
    print(f"FAISS index size : {index.ntotal} vectors × {EMBED_DIM}d")
    print(f"SQLite DB        : {args.db}")
    print(f"FAISS index      : {args.faiss}")


if __name__ == '__main__':
    main()
