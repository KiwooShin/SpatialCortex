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

import os, sys
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
STRIDE          = 10      # sample every Nth frame (~1 fps at 10fps RGB)
OUT_FAISS       = BASE / 'data/keyframe_index.faiss'
OUT_CSV         = BASE / 'data/keyframe_index.csv'

SCENES = {
    "seq00": {
        "vrs":  BASE / "data/aeo/aeo_seq00_173376298563204/main.vrs",
        "traj": BASE / "data/aeo/aeo_seq00_173376298563204/mps/mps/slam/closed_loop_trajectory.csv",
    },
    "seq01": {
        "vrs":  BASE / "data/aeo/aeo_seq01_208838848508107/main.vrs",
        "traj": BASE / "data/aeo/aeo_seq01_208838848508107/mps/slam/closed_loop_trajectory.csv",
    },
    "seq02": {
        "vrs":  BASE / "data/aeo/aeo_seq02_181771578105956/main.vrs",
        "traj": BASE / "data/aeo/aeo_seq02_181771578105956/mps/slam/closed_loop_trajectory.csv",
    },
}

# ── Trajectory ─────────────────────────────────────────────────────────────────

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
    poses = []
    for _, row in df.iterrows():
        poses.append({
            "tx": row["tx_world_device"], "ty": row["ty_world_device"],
            "tz": row["tz_world_device"],
            "qw": row["qw_world_device"], "qx": row["qx_world_device"],
            "qy": row["qy_world_device"], "qz": row["qz_world_device"],
        })
    return times_us, poses

def interp_pose(times_us, poses, query_ns):
    query_us = query_ns // 1000
    idx = min(max(bisect_left(times_us, query_us), 0), len(times_us)-1)
    return poses[idx]

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
        times_us, poses = load_trajectory(str(paths['traj']))

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
            pose = interp_pose(times_us, poses, int(ts_ns))

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
