"""
query_scene.py — natural language query against the spatial memory database.

Flow:
  text query → CLIP text embedding → FAISS top-k → SQLite metadata
             → (optional) Gemma 3 27B visual confirmation
             → print 3D location + scene

Usage:
  # CLIP retrieval only (no VLM, works immediately):
  python scripts/query_scene.py --query "where is the sofa"

  # With Gemma 3 visual confirmation:
  python scripts/query_scene.py --query "where is the sofa" --vlm

  # Restrict search to one scene:
  python scripts/query_scene.py --query "chair near the window" --scene seq01

  # Interactive mode:
  python scripts/query_scene.py --interactive
"""

import argparse
import os
import sqlite3
import textwrap

# Use locally cached weights — no network calls during inference
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

CLIP_MODEL      = 'ViT-L-14-quickgelu'   # OpenAI CLIP uses QuickGELU; avoids activation mismatch
CLIP_PRETRAINED = 'openai'
DEFAULT_DB      = 'data/scene_db.sqlite'
DEFAULT_FAISS   = 'data/scene.faiss'
DEFAULT_TOP_K   = 5
VLM_MODEL       = 'google/gemma-3-4b-it'   # swap to 27b-it for highest quality


# ── CLIP ─────────────────────────────────────────────────────────────────────

def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer


@torch.no_grad()
def embed_text(model, tokenizer, text, device):
    tokens = tokenizer([text]).to(device)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]


# ── Database ──────────────────────────────────────────────────────────────────

def search_db(conn, ids):
    placeholders = ','.join('?' * len(ids))
    rows = conn.execute(
        f"SELECT * FROM objects WHERE clip_idx IN ({placeholders})",
        ids
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM objects LIMIT 0").description]
    return [dict(zip(cols, r)) for r in rows]


def nearby_objects(conn, obj, radius_m=1.5, exclude_id=None):
    """Return objects within radius_m of obj in the same scene."""
    rows = conn.execute("""
        SELECT name, tx, ty, tz FROM objects
        WHERE scene = ? AND id != ?
          AND ((tx-?)*(tx-?) + (ty-?)*(ty-?) + (tz-?)*(tz-?)) < ?
        ORDER BY ((tx-?)*(tx-?) + (ty-?)*(ty-?) + (tz-?)*(tz-?))
        LIMIT 4
    """, (
        obj['scene'], obj['id'] if exclude_id is None else exclude_id,
        obj['tx'], obj['tx'], obj['ty'], obj['ty'], obj['tz'], obj['tz'],
        radius_m ** 2,
        obj['tx'], obj['tx'], obj['ty'], obj['ty'], obj['tz'], obj['tz'],
    )).fetchall()
    return rows


# ── VLM (Gemma 3) ─────────────────────────────────────────────────────────────

_vlm_model = None
_vlm_processor = None


def load_vlm(device):
    global _vlm_model, _vlm_processor
    if _vlm_model is not None:
        return _vlm_model, _vlm_processor

    print(f"Loading VLM {VLM_MODEL} …", flush=True)
    from transformers import AutoProcessor, AutoModelForImageTextToText

    _vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL)
    _vlm_model = AutoModelForImageTextToText.from_pretrained(
        VLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    _vlm_model.eval()
    print("VLM loaded.")
    return _vlm_model, _vlm_processor


def vlm_confirm(candidates, query, device):
    """
    Show top candidates to Gemma 3 and ask which best matches the query.
    Returns (best_candidate_dict, explanation_str).
    """
    vlm, processor = load_vlm(device)

    # Build a message with images + descriptions
    content = []
    valid_cands = [c for c in candidates if c['crop_path'] and os.path.exists(c['crop_path'])]
    if not valid_cands:
        return candidates[0], "No crop images available for VLM confirmation."

    # Up to 3 images
    imgs = []
    desc_lines = []
    for i, c in enumerate(valid_cands[:3]):
        try:
            img = Image.open(c['crop_path']).convert('RGB')
            # Resize to max 512px on the long side to save tokens
            max_sz = 512
            w, h = img.size
            if max(w, h) > max_sz:
                scale = max_sz / max(w, h)
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            imgs.append(img)
            content.append({"type": "image"})
            desc_lines.append(
                f"Option {i+1}: {c['name']} in scene {c['scene']} "
                f"at position ({c['tx']:.2f}, {c['ty']:.2f}, {c['tz']:.2f})m"
            )
        except Exception:
            pass

    if not imgs:
        return candidates[0], "Could not open crop images."

    desc_text = "\n".join(desc_lines)
    content.append({"type": "text", "text": (
        f"The user is searching for: \"{query}\"\n\n"
        f"These are the top candidate objects found in the room:\n{desc_text}\n\n"
        f"Which image best matches what the user is looking for? "
        f"Reply with the option number (1, 2, or 3) and one sentence describing "
        f"where the object is in the room relative to other objects."
    )})

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(vlm.device)

    with torch.inference_mode():
        out = vlm.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    prompt_len = inputs['input_ids'].shape[1]
    reply = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    # Try to parse which option was chosen
    chosen = valid_cands[0]
    for i, c in enumerate(valid_cands[:3]):
        if str(i + 1) in reply[:10]:
            chosen = c
            break

    return chosen, reply


# ── Result formatting ──────────────────────────────────────────────────────────

def format_result(obj, similarity, conn, rank=1):
    nearby = nearby_objects(conn, obj)
    near_str = ', '.join(r[0] for r in nearby) if nearby else 'none'
    sim_str = f"{similarity:.3f}" if similarity is not None else "name-match (no image embed)"

    lines = [
        f"  [{rank}] {obj['name'].upper()}",
        f"      Scene     : {obj['scene']}",
        f"      Position  : ({obj['tx']:+.2f}, {obj['ty']:+.2f}, {obj['tz']:+.2f}) m  (world XYZ)",
        f"      Size      : {obj['scale_x']:.2f}×{obj['scale_y']:.2f}×{obj['scale_z']:.2f} m",
        f"      Confidence: {obj['prob']:.2f}  ({obj['count']} observations fused)",
        f"      Similarity: {sim_str}",
        f"      Nearby    : {near_str}",
    ]
    if obj['crop_path'] and os.path.exists(obj['crop_path']):
        lines.append(f"      Crop      : {obj['crop_path']}")
    return '\n'.join(lines)


# ── Query ─────────────────────────────────────────────────────────────────────

def name_match_fallback(conn, query_words, scene_filter, exclude_clip_idxs, limit=3):
    """
    Return SQLite objects (has_image=0) whose class name appears in the query.
    Used to surface image-less objects that FAISS cannot rank.
    """
    scene_clause = "AND scene = ?" if scene_filter else ""
    params = [scene_filter] if scene_filter else []
    rows = conn.execute(
        f"SELECT * FROM objects WHERE has_image=0 {scene_clause}", params
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM objects LIMIT 0").description]
    results = []
    for row in rows:
        obj = dict(zip(cols, row))
        if obj['clip_idx'] in exclude_clip_idxs:
            continue
        if any(w in query_words for w in obj['name'].lower().split('_')):
            results.append(obj)
    return results[:limit]


def run_query(query, model, tokenizer, index, conn, device,
              top_k, scene_filter, use_vlm):
    print(f'\nSearching: "{query}"', end='')
    if scene_filter:
        print(f'  [scene={scene_filter}]', end='')
    print()

    q_vec = embed_text(model, tokenizer, query, device).reshape(1, -1).astype(np.float32)

    # Always fetch a wide window — class-matched objects may be ranked low by raw
    # CLIP similarity (domain gap: Aria fisheye vs. CLIP's training images).
    # Fetching ~60% of the index is cheap at 78 vectors and ensures nothing is missed.
    fetch_k = min(max(50, top_k * 10), index.ntotal)

    sims, idxs = index.search(q_vec, fetch_k)
    sims, idxs = sims[0], idxs[0]

    # Load metadata for FAISS hits
    valid_idx = [int(i) for i in idxs if i >= 0]
    results_map = {r['clip_idx']: r for r in search_db(conn, valid_idx)}

    # Build ranked list from FAISS (image embeddings only).
    # Strategy: EFM3D class labels are high quality, so class-keyword match is
    # the primary retrieval signal. CLIP similarity breaks ties within the same class
    # and provides fallback ranking when no class name is found in the query.
    #
    # Two-bucket approach:
    #   Bucket A: objects whose class name words appear in the query → sort by CLIP sim
    #   Bucket B: everything else → sort by CLIP sim
    # Result: Bucket A first (best class match), then Bucket B (CLIP-only fallback).
    query_words = set(query.lower().replace('?', '').replace(',', '').split())

    # Also expand query with synonyms for common concepts
    SYNONYMS = {
        'sit': {'chair', 'sofa', 'bench', 'seat'},
        'sleep': {'bed', 'sofa'},
        'lie': {'bed', 'sofa'},
        'light': {'lamp'},
        'couch': {'sofa'},
        'seat': {'chair', 'sofa'},
        'book': {'shelf'},
        'storage': {'cabinet', 'shelf', 'dresser'},
        'dresser': {'dresser', 'cabinet'},
        'tv': {'monitor', 'tv'},
        'screen': {'monitor', 'tv'},
    }
    expanded_classes = set()
    for word in query_words:
        if word in SYNONYMS:
            expanded_classes |= SYNONYMS[word]

    bucket_a, bucket_b = [], []
    for sim, idx in zip(sims, idxs):
        if idx < 0:
            continue
        obj = results_map.get(int(idx))
        if obj is None:
            continue
        if scene_filter and obj['scene'] != scene_filter:
            continue
        name_words = set(obj['name'].lower().split('_'))
        is_class_match = bool(name_words & query_words) or bool(name_words & expanded_classes)
        if is_class_match:
            bucket_a.append((float(sim), obj))
        else:
            bucket_b.append((float(sim), obj))

    bucket_a.sort(key=lambda x: x[0], reverse=True)
    bucket_b.sort(key=lambda x: x[0], reverse=True)
    all_candidates = bucket_a + bucket_b
    ranked = all_candidates[:top_k]

    # Supplement with class-name matches for objects that have no image embedding
    seen_clip_idxs = {obj['clip_idx'] for _, obj in ranked}
    fallbacks = name_match_fallback(conn, query_words, scene_filter, seen_clip_idxs)
    for obj in fallbacks:
        ranked.append((None, obj))   # None similarity = text-matched, not CLIP-ranked

    if not ranked:
        print("  No results found.")
        return

    print(f"\nTop {len(ranked)} matches:\n")
    for rank, (sim, obj) in enumerate(ranked, 1):
        print(format_result(obj, sim, conn, rank))
        print()

    if use_vlm and ranked:
        print("── Gemma 3 visual confirmation ──────────────────────────────")
        top_objs = [obj for _, obj in ranked[:3]]
        try:
            best, explanation = vlm_confirm(top_objs, query, device)
            print(f"\n  Best match : {best['name']} in {best['scene']}")
            print(f"  Position   : ({best['tx']:+.2f}, {best['ty']:+.2f}, {best['tz']:+.2f}) m")
            print(f"\n  Gemma 3 says:\n  " +
                  textwrap.fill(explanation, width=72, subsequent_indent='  '))
        except Exception as e:
            print(f"  VLM error: {e}")
            print("  (Is the model downloaded? Run: python scripts/query_scene.py --download-vlm)")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query',    '-q', default=None, help='search query text')
    ap.add_argument('--db',       default=DEFAULT_DB)
    ap.add_argument('--faiss',    default=DEFAULT_FAISS)
    ap.add_argument('--top-k',    type=int, default=DEFAULT_TOP_K)
    ap.add_argument('--scene',    default=None,
                    help='restrict search to one scene (e.g. seq01)')
    ap.add_argument('--vlm',      action='store_true',
                    help='use Gemma 3 for visual confirmation of top results')
    ap.add_argument('--vlm-model', default=VLM_MODEL,
                    help=f'HuggingFace model ID for VLM (default: {VLM_MODEL})')
    ap.add_argument('--download-vlm', action='store_true',
                    help='download the VLM model and exit')
    ap.add_argument('--interactive', '-i', action='store_true',
                    help='interactive query loop')
    ap.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.download_vlm:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print(f"Downloading {args.vlm_model} …")
        AutoProcessor.from_pretrained(args.vlm_model)
        AutoModelForImageTextToText.from_pretrained(
            args.vlm_model, torch_dtype=torch.bfloat16
        )
        print("Done.")
        return

    if not os.path.exists(args.db):
        print(f"ERROR: DB not found at {args.db}. Run build_scene_db.py first.")
        return
    if not os.path.exists(args.faiss):
        print(f"ERROR: FAISS index not found at {args.faiss}. Run build_scene_db.py first.")
        return

    print(f"Loading CLIP {CLIP_MODEL}/{CLIP_PRETRAINED} …", flush=True)
    model, _, tokenizer = load_clip(args.device)
    print(f"Loading FAISS index …", flush=True)
    index = faiss.read_index(args.faiss)
    conn  = sqlite3.connect(args.db)

    n_objects = conn.execute('SELECT COUNT(*) FROM objects').fetchone()[0]
    scenes    = [r[0] for r in conn.execute('SELECT DISTINCT scene FROM objects').fetchall()]
    print(f"Spatial memory: {n_objects} objects across {scenes}\n")

    if args.interactive:
        print("Interactive mode. Type a query (empty line to quit).\n")
        while True:
            try:
                q = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run_query(q, model, tokenizer, index, conn, args.device,
                      args.top_k, args.scene, args.vlm)
    elif args.query:
        run_query(args.query, model, tokenizer, index, conn, args.device,
                  args.top_k, args.scene, args.vlm)
    else:
        ap.print_help()

    conn.close()


if __name__ == '__main__':
    main()
