"""
fuse_scene_obbs.py — aggregate per-snippet OBBs into a consistent scene map.

For a STATIC scene observed from multiple views/times, each physical object
accumulates many independent per-snippet detections.  This script:

  1. Collects all OBBs across all snippet timestamps (world frame).
  2. Clusters same-class OBBs by center proximity (greedy NMS-style grouping).
  3. For each cluster:
       - confidence-weighted centroid  (position)
       - confidence-weighted mean scale (in log space)
       - quaternion mean via eigenvector method (rotation)
       - combined confidence = 1 - prod(1 - p_i)  (evidence accumulation)
       - observation count stored in 'count' column
  4. Filters clusters with fewer than --min-obs observations.
  5. Writes scene_obbs.csv  (one row per physical object).

No pytorch3d required.  Replaces the EFM3D track_obbs() whose dependency
(pytorch3d) is unavailable on aarch64/CUDA 13.0.
"""

import argparse
import os

import numpy as np
import pandas as pd


# ── Quaternion helpers ────────────────────────────────────────────────────────

def quat_mean(qs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Markley et al. weighted quaternion mean via dominant eigenvector of
    the weighted outer-product matrix (works for nearly-aligned quaternions).
    qs: (N, 4)  [qw, qx, qy, qz]
    weights: (N,)  non-negative, need not sum to 1
    Returns: (4,) unit quaternion
    """
    # Ensure sign consistency (flip q if dot with first q is negative)
    q0 = qs[0]
    signs = np.sign(qs @ q0)          # +1 or -1 per row
    signs[signs == 0] = 1.0
    qs_aligned = qs * signs[:, None]

    w = weights / weights.sum()
    M = (w[:, None] * qs_aligned).T @ qs_aligned   # 4×4
    _, vecs = np.linalg.eigh(M)                     # eigenvalues ascending
    q_mean = vecs[:, -1]                            # dominant eigenvector
    if q_mean[0] < 0:
        q_mean = -q_mean
    return q_mean / np.linalg.norm(q_mean)


def quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ])


def center_dist(row_a, row_b) -> float:
    return np.sqrt((row_a['tx_world_object'] - row_b['tx_world_object'])**2 +
                   (row_a['ty_world_object'] - row_b['ty_world_object'])**2 +
                   (row_a['tz_world_object'] - row_b['tz_world_object'])**2)


def aabb_iou_3d(row_a, row_b) -> float:
    """Fast approximate 3D IoU using axis-aligned extents."""
    def vol(r): return r['scale_x'] * r['scale_y'] * r['scale_z']
    def overlap_1d(ca, sa, cb, sb):
        return max(0.0, min(ca + sa/2, cb + sb/2) - max(ca - sa/2, cb - sb/2))
    ix = overlap_1d(row_a['tx_world_object'], row_a['scale_x'], row_b['tx_world_object'], row_b['scale_x'])
    iy = overlap_1d(row_a['ty_world_object'], row_a['scale_y'], row_b['ty_world_object'], row_b['scale_y'])
    iz = overlap_1d(row_a['tz_world_object'], row_a['scale_z'], row_b['tz_world_object'], row_b['scale_z'])
    inter = ix * iy * iz
    if inter <= 0:
        return 0.0
    union = vol(row_a) + vol(row_b) - inter
    return inter / union if union > 0 else 0.0


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_obbs(rows: pd.DataFrame, dist_thresh: float, iou_thresh: float) -> list:
    """
    Greedy proximity clustering of OBBs (all same class).
    Two OBBs belong to the same cluster if:
      center_dist < dist_thresh  AND  aabb_iou_3d >= iou_thresh
    (if iou_thresh=0, only center distance is used)
    Returns list of lists of original indices.
    """
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
                if iou_thresh <= 0 or aabb_iou_3d(recs[i], recs[j]) >= iou_thresh:
                    clusters[-1].append(j)
                    assigned[j] = cl_id
    return clusters


# ── Fusion ────────────────────────────────────────────────────────────────────

def fuse_cluster(rows: pd.DataFrame) -> dict:
    """
    Fuse a cluster of detections of the same physical object into one OBB.
    Position/scale: confidence-weighted mean (scale in log space).
    Rotation: eigenvector quaternion mean.
    Confidence: accumulated evidence  1 - prod(1 - p_i).
    """
    probs = rows['prob'].values.astype(float)
    w = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)

    tx = np.dot(w, rows['tx_world_object'].values)
    ty = np.dot(w, rows['ty_world_object'].values)
    tz = np.dot(w, rows['tz_world_object'].values)

    # Scale in log space avoids bias toward large detections
    log_sx = np.dot(w, np.log(rows['scale_x'].values.clip(1e-4)))
    log_sy = np.dot(w, np.log(rows['scale_y'].values.clip(1e-4)))
    log_sz = np.dot(w, np.log(rows['scale_z'].values.clip(1e-4)))

    qs = rows[['qw_world_object','qx_world_object',
               'qy_world_object','qz_world_object']].values.astype(float)
    qw, qx, qy, qz = quat_mean(qs, probs)

    # Accumulated evidence confidence
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
        'name': rows['name'].iloc[0],
        'instance': rows['instance'].iloc[0] if 'instance' in rows.columns else -1,
        'sem_id': rows['sem_id'].iloc[0] if 'sem_id' in rows.columns else -1,
        'prob': fused_prob,
        'count': len(rows),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obbs', required=True, help='snippet_obbs.csv')
    ap.add_argument('--output', default=None, help='output CSV path (default: scene_obbs.csv next to input)')
    ap.add_argument('--prob-thresh', type=float, default=0.20,
                    help='min per-detection confidence to include (default 0.20)')
    ap.add_argument('--dist-thresh', type=float, default=0.80,
                    help='max center distance (m) to cluster two detections (default 0.80)')
    ap.add_argument('--iou-thresh', type=float, default=0.0,
                    help='min AABB-IoU to cluster (0 = disable, use distance only; default 0.0)')
    ap.add_argument('--min-obs', type=int, default=3,
                    help='min observations to keep a fused object (default 3)')
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.obbs), 'scene_obbs.csv')

    # Load
    df = pd.read_csv(args.obbs)
    print(f"Loaded {len(df)} raw OBBs from {len(df['time_ns'].unique())} snippets")
    df = df[df['prob'] >= args.prob_thresh].copy()
    print(f"After prob≥{args.prob_thresh}: {len(df)} OBBs")

    # Cluster and fuse per class
    results = []
    for name, group in df.groupby('name'):
        group = group.reset_index(drop=True)
        clusters = cluster_obbs(group, args.dist_thresh, args.iou_thresh)
        for cl_indices in clusters:
            cl_rows = group.iloc[cl_indices]
            fused = fuse_cluster(cl_rows)
            results.append(fused)

    scene = pd.DataFrame(results)

    # Filter by minimum observations
    before = len(scene)
    scene = scene[scene['count'] >= args.min_obs].copy()
    print(f"Fused into {before} clusters; {len(scene)} survive min-obs≥{args.min_obs}")

    # Sort by confidence desc
    scene = scene.sort_values('prob', ascending=False).reset_index(drop=True)

    scene.to_csv(args.output, index=False)
    print(f"\nScene OBBs written to {args.output}")
    print(scene[['name','prob','count',
                 'tx_world_object','ty_world_object','tz_world_object',
                 'scale_x','scale_y','scale_z']].to_string(index=False))


if __name__ == '__main__':
    main()
