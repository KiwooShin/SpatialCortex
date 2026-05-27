"""
build_keyframe_index.py — build CLIP image embedding index of VRS keyframes.

Samples RGB frames from each VRS recording at regular intervals, encodes
each with CLIP ViT-L/14 image encoder, and stores the embeddings in a FAISS
index alongside pose metadata. Used by navigate.py for re-localization.

NOTE: CLIP is used here as a pragmatic baseline (already in the stack).
      DINOv2 ViT-L/14 (facebook/dinov2-large) would give better viewpoint-
      invariant features for place recognition. Swap load_encoder() if needed.

Usage:
    conda activate efm3d
    python scripts/build_keyframe_index.py
    # Output: data/keyframe_index.faiss + data/keyframe_index.csv (~30 s)
"""

import os
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

from spatialcortex.config import BASE, RGB_SID, SCENES, CLIP_MODEL, CLIP_PRETRAINED, KEYFRAME_FAISS, KEYFRAME_CSV
from spatialcortex.geometry import quat_to_rotmat, load_trajectory, interp_pose

STRIDE    = 10      # sample every Nth frame (~1 fps at 10fps RGB)
OUT_FAISS = KEYFRAME_FAISS
OUT_CSV   = KEYFRAME_CSV

# ── CLIP encoder ───────────────────────────────────────────────────────────────

def load_encoder(device):
    print(f"Loading CLIP {CLIP_MODEL}/{CLIP_PRETRAINED} …", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device)
    model.eval()
    return model, preprocess

@torch.no_grad()
def encode_image(model, preprocess, img_pil, device):
    """Encode a PIL image → L2-normalised 768-dim CLIP embedding."""
    x = preprocess(img_pil).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().float().numpy()[0]

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, preprocess = load_encoder(device)

    all_embeddings = []
    all_meta       = []
    global_idx     = 0

    for scene, paths in SCENES.items():
        print(f"\n── {scene} ─────────────────────────────────────────────────")
        times_us, Rs_traj, ts_pos_traj = load_trajectory(str(paths['traj']))

        provider  = data_provider.create_vrs_data_provider(str(paths['vrs']))
        ts_all    = provider.get_timestamps_ns(RGB_SID, sensor_data.TimeDomain.DEVICE_TIME)
        ts_sample = ts_all[::STRIDE]
        print(f"  {len(ts_all)} frames → {len(ts_sample)} sampled (stride={STRIDE})")

        scene_count = 0
        for frame_idx, ts_ns in enumerate(ts_sample):
            img_data, _ = provider.get_image_data_by_time_ns(
                RGB_SID, int(ts_ns),
                sensor_data.TimeDomain.DEVICE_TIME,
                sensor_data.TimeQueryOptions.CLOSEST,
            )
            if not img_data.is_valid():
                continue

            arr = img_data.to_numpy_array()
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 1:
                arr = cv2.cvtColor(arr[:,:,0], cv2.COLOR_GRAY2RGB)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

            # CW 90° rotation to correct Aria sensor orientation
            arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)

            img_pil = Image.fromarray(arr)
            emb = encode_image(model, preprocess, img_pil, device)
            R_frame, t_frame = interp_pose(times_us, Rs_traj, ts_pos_traj, int(ts_ns))
            pose = {"tx": t_frame[0], "ty": t_frame[1], "tz": t_frame[2],
                    "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}

            all_embeddings.append(emb)
            all_meta.append({
                "faiss_idx": global_idx,
                "scene":     scene,
                "ts_ns":     int(ts_ns),
                "frame_idx": frame_idx,
                **pose,
            })
            global_idx  += 1
            scene_count += 1

            if scene_count % 20 == 0:
                print(f"  … {scene_count}/{len(ts_sample)} frames encoded", flush=True)

        print(f"  {scene}: {scene_count} keyframes encoded")

    # Build FAISS index
    dim = all_embeddings[0].shape[0]
    matrix = np.stack(all_embeddings).astype(np.float32)
    index = faiss.IndexFlatIP(dim)   # cosine (embeddings are already L2-normalised)
    index.add(matrix)

    faiss.write_index(index, str(OUT_FAISS))
    pd.DataFrame(all_meta).to_csv(str(OUT_CSV), index=False)

    print(f"\n✓ {global_idx} keyframes indexed")
    print(f"  FAISS : {OUT_FAISS}")
    print(f"  CSV   : {OUT_CSV}")


if __name__ == '__main__':
    main()
