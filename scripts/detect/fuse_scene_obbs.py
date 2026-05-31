"""
fuse_scene_obbs.py — aggregate per-snippet OBBs into a consistent scene map.

Two-stage pipeline:

  Stage 1 — per-class temporal fusion
    * Cluster same-class OBBs across all snippets by centre distance (0.80 m).
    * Fuse each cluster: confidence-weighted position/scale, quaternion mean,
      accumulated evidence confidence, observation count.

  Stage 2 — cross-class NMS
    * Any two fused objects of *different* classes whose centres are within
      --cross-nms-dist (default 0.30 m) are duplicates caused by label
      instability (e.g. the model alternately calls the same fan "pillow").
    * Keep the cluster with the higher observation count; suppress the other.

This fixes two known EFM3D failure modes:
  - Duplicate boxes: multiple boxes for one object → Stage 1 per-class merge.
  - Label flipping: same physical object gets two scene entries under different
    class names → Stage 2 cross-class NMS.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from spatialcortex.geometry import quat_to_rotmat


# ── Quaternion helpers ────────────────────────────────────────────────────────

def quat_mean(qs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Markley et al. weighted quaternion mean via dominant eigenvector."""
    q0 = qs[0]
    signs = np.sign(qs @ q0)
    signs[signs == 0] = 1.0
    qs_aligned = qs * signs[:, None]
    w = weights / weights.sum()
    M = (w[:, None] * qs_aligned).T @ qs_aligned
    _, vecs = np.linalg.eigh(M)
    q_mean = vecs[:, -1]
    if q_mean[0] < 0:
        q_mean = -q_mean
    return q_mean / np.linalg.norm(q_mean)


# ── Stage 1: per-class clustering ─────────────────────────────────────────────

def center_dist(row_a, row_b) -> float:
    return np.sqrt((row_a['tx_world_object'] - row_b['tx_world_object'])**2 +
                   (row_a['ty_world_object'] - row_b['ty_world_object'])**2 +
                   (row_a['tz_world_object'] - row_b['tz_world_object'])**2)


def cluster_obbs(rows: pd.DataFrame, dist_thresh: float) -> list:
    """Greedy proximity clustering of OBBs (same class). O(n²)."""
    n = len(rows)
    assigned = [-1] * n
    clusters = []
    recs = rows.to_dict('records')
    for i in range(n):
        if assigned[i] >= 0:
            continue
        cl_id = len(clusters)
        clusters.append([i])
        assigned[i] = cl_id
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            if center_dist(recs[i], recs[j]) < dist_thresh:
                clusters[-1].append(j)
                assigned[j] = cl_id
    return clusters


def fuse_cluster(rows: pd.DataFrame, label: str) -> dict:
    """Fuse a cluster of detections into one OBB."""
    probs = rows['prob'].values.astype(float)
    w = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)

    tx = np.dot(w, rows['tx_world_object'].values)
    ty = np.dot(w, rows['ty_world_object'].values)
    tz = np.dot(w, rows['tz_world_object'].values)

    log_sx = np.dot(w, np.log(rows['scale_x'].values.clip(1e-4)))
    log_sy = np.dot(w, np.log(rows['scale_y'].values.clip(1e-4)))
    log_sz = np.dot(w, np.log(rows['scale_z'].values.clip(1e-4)))

    qs = rows[['qw_world_object', 'qx_world_object',
               'qy_world_object', 'qz_world_object']].values.astype(float)
    qw, qx, qy, qz = quat_mean(qs, probs)

    fused_prob = float(1.0 - np.prod(1.0 - probs.clip(0, 1)))
    fused_prob = min(fused_prob, 0.99)

    return {
        'tx_world_object': tx,
        'ty_world_object': ty,
        'tz_world_object': tz,
        'qw_world_object': qw,
        'qx_world_object': qx,
        'qy_world_object': qy,
        'qz_world_object': qz,
        'scale_x': float(np.exp(log_sx)),
        'scale_y': float(np.exp(log_sy)),
        'scale_z': float(np.exp(log_sz)),
        'name': label,
        'instance': rows['instance'].iloc[0] if 'instance' in rows.columns else -1,
        'sem_id': rows['sem_id'].iloc[0] if 'sem_id' in rows.columns else -1,
        'prob': fused_prob,
        'count': len(rows),
    }


# ── Stage 2: cross-class NMS ──────────────────────────────────────────────────

def cross_class_nms(scene: pd.DataFrame, dist_thresh: float) -> pd.DataFrame:
    """
    Suppress label-flip duplicates: two fused objects of *different* classes
    whose centres are within dist_thresh → keep the one with more observations.
    Returns filtered DataFrame.
    """
    if len(scene) == 0:
        return scene

    # Sort by count descending so the dominant label is processed first.
    scene = scene.sort_values('count', ascending=False).reset_index(drop=True)
    positions = scene[['tx_world_object', 'ty_world_object', 'tz_world_object']].values
    keep = np.ones(len(scene), dtype=bool)

    suppressed_pairs = []
    for i in range(len(scene)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(scene)):
            if not keep[j]:
                continue
            if scene.iloc[i]['name'] == scene.iloc[j]['name']:
                continue  # same class — Stage 1 already handled this
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < dist_thresh:
                suppressed_pairs.append(
                    f"  {scene.iloc[j]['name']}(n={scene.iloc[j]['count']}) "
                    f"← absorbed by {scene.iloc[i]['name']}(n={scene.iloc[i]['count']}) "
                    f"d={dist:.2f}m"
                )
                keep[j] = False

    if suppressed_pairs:
        print(f"Stage 2 cross-class NMS: suppressed {(~keep).sum()} duplicate(s):")
        for msg in suppressed_pairs:
            print(msg)
    else:
        print("Stage 2 cross-class NMS: no cross-class duplicates found.")

    return scene[keep].reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obbs', required=True, help='snippet_obbs.csv')
    ap.add_argument('--output', default=None,
                    help='output CSV path (default: scene_obbs.csv next to input)')
    ap.add_argument('--prob-thresh', type=float, default=0.20,
                    help='min per-detection confidence (default 0.20)')
    ap.add_argument('--dist-thresh', type=float, default=0.80,
                    help='Stage 1 max centre distance (m) to cluster same-class '
                         'detections (default 0.80)')
    ap.add_argument('--cross-nms-dist', type=float, default=0.30,
                    help='Stage 2 max centre distance (m) to suppress cross-class '
                         'label-flip duplicates (default 0.30)')
    ap.add_argument('--min-obs', type=int, default=3,
                    help='min observations to keep a fused object (default 3)')
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.obbs), 'scene_obbs.csv')

    df = pd.read_csv(args.obbs)
    print(f"Loaded {len(df)} raw OBBs from {df['time_ns'].nunique()} timestamps")
    df = df[df['prob'] >= args.prob_thresh].copy()
    print(f"After prob≥{args.prob_thresh}: {len(df)} OBBs")

    # ── Stage 1: per-class temporal fusion ──────────────────────────────────
    print(f"\nStage 1: per-class clustering (dist_thresh={args.dist_thresh}m) …")
    results = []
    for name, group in df.groupby('name'):
        group = group.reset_index(drop=True)
        clusters = cluster_obbs(group, args.dist_thresh)
        for cl_indices in clusters:
            cl_rows = group.iloc[cl_indices]
            results.append(fuse_cluster(cl_rows, label=name))

    scene = pd.DataFrame(results)
    before_minobs = len(scene)
    scene = scene[scene['count'] >= args.min_obs].copy()
    print(f"  {before_minobs} clusters → {len(scene)} survive min-obs≥{args.min_obs}")

    # ── Stage 2: cross-class NMS ─────────────────────────────────────────────
    print(f"\nStage 2: cross-class NMS (dist_thresh={args.cross_nms_dist}m) …")
    scene = cross_class_nms(scene, dist_thresh=args.cross_nms_dist)

    # ── Output ────────────────────────────────────────────────────────────────
    scene = scene.sort_values('prob', ascending=False).reset_index(drop=True)
    scene.to_csv(args.output, index=False)
    print(f"\nScene OBBs written to {args.output}  ({len(scene)} objects)")
    print(scene[['name', 'prob', 'count',
                 'tx_world_object', 'ty_world_object', 'tz_world_object',
                 'scale_x', 'scale_y', 'scale_z']].to_string(index=False))


if __name__ == '__main__':
    main()
