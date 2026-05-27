"""
scene_topdown.py — render a single top-down scene map from scene_obbs.csv.

Centers the view on the centroid of all fused OBBs and sets the view radius
to cover every object with padding.  Draws the full camera trajectory from
the SLAM closed-loop CSV.

Usage:
  python scripts/scene_topdown.py \
      --obbs  <path/to/scene_obbs.csv> \
      --traj  <path/to/closed_loop_trajectory.csv> \
      --output <path/to/out.jpg> \
      [--size 1024] [--padding 1.5] [--title "seq01"]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import pandas as pd

from spatialcortex.geometry import obb_corners_world
from spatialcortex.drawing import get_color


# ── Legend ────────────────────────────────────────────────────────────────────

def draw_legend(img, names_seen: list):
    """Draw a compact class legend in the bottom-right corner."""
    unique = sorted(set(names_seen))
    font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
    line_h = 18
    pad = 8
    col_w = 110
    rows = unique

    box_h = len(rows) * line_h + 2 * pad
    box_w = col_w + 2 * pad
    H, W = img.shape[:2]
    x0 = W - box_w - 8
    y0 = H - box_h - 8

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (W - 8, H - 8), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    for k, name in enumerate(rows):
        color = get_color(name)
        y = y0 + pad + k * line_h + line_h // 2
        cv2.rectangle(img, (x0 + pad, y - 5), (x0 + pad + 12, y + 5), color, -1)
        cv2.putText(img, name, (x0 + pad + 16, y + 4),
                    font, sc, (210, 210, 210), th, cv2.LINE_AA)


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_scene_topdown(obbs: pd.DataFrame, traj_xy: list,
                         title: str, view_size: int, padding_m: float):
    """
    Render a top-down scene map centered on the OBB centroid.

    traj_xy: list of (x, y) world coordinates for all trajectory poses
    """
    # Scene centroid from OBB centres (weighted by prob)
    cx_w = np.average(obbs["tx_world_object"].values, weights=obbs["prob"].values)
    cy_w = np.average(obbs["ty_world_object"].values, weights=obbs["prob"].values)
    scene_center = np.array([cx_w, cy_w])

    # Radius: furthest OBB corner from centroid + padding
    max_r = 0.0
    for _, row in obbs.iterrows():
        corners = obb_corners_world(
            row["tx_world_object"], row["ty_world_object"], row["tz_world_object"],
            row["qw_world_object"], row["qx_world_object"],
            row["qy_world_object"], row["qz_world_object"],
            row["scale_x"], row["scale_y"], row["scale_z"],
        )
        dists = np.linalg.norm(corners[:, :2] - scene_center, axis=1)
        max_r = max(max_r, dists.max())
    # Also account for trajectory extent
    if traj_xy:
        traj_arr = np.array(traj_xy)
        traj_dists = np.linalg.norm(traj_arr - scene_center, axis=1)
        max_r = max(max_r, traj_dists.max())

    radius_m = max_r + padding_m

    img = np.full((view_size, view_size, 3), 18, dtype=np.uint8)
    scale = (view_size / 2) / radius_m

    def w2p(wx, wy):
        px = int(view_size // 2 + (wx - scene_center[0]) * scale)
        py = int(view_size // 2 - (wy - scene_center[1]) * scale)
        return px, py

    # ── Grid ──────────────────────────────────────────────────────────────────
    grid_step = 1.0
    # Round grid start to nearest meter
    gx_start = int(np.floor(scene_center[0] - radius_m))
    gx_end   = int(np.ceil(scene_center[0] + radius_m)) + 1
    gy_start = int(np.floor(scene_center[1] - radius_m))
    gy_end   = int(np.ceil(scene_center[1] + radius_m)) + 1
    for gx in range(gx_start, gx_end):
        p0 = w2p(gx, scene_center[1] - radius_m)
        p1 = w2p(gx, scene_center[1] + radius_m)
        cv2.line(img, p0, p1, (38, 38, 38), 1)
    for gy in range(gy_start, gy_end):
        p0 = w2p(scene_center[0] - radius_m, gy)
        p1 = w2p(scene_center[0] + radius_m, gy)
        cv2.line(img, p0, p1, (38, 38, 38), 1)

    # ── Camera trajectory ─────────────────────────────────────────────────────
    if len(traj_xy) > 1:
        pts = [w2p(x, y) for x, y in traj_xy]
        for k in range(len(pts) - 1):
            cv2.line(img, pts[k], pts[k+1], (70, 70, 70), 1)
        # Mark start / end
        cv2.circle(img, pts[0],  5, (0, 200, 80),  -1)
        cv2.circle(img, pts[-1], 5, (0, 100, 255), -1)
        cv2.putText(img, "start", (pts[0][0]+6,  pts[0][1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 80),  1)
        cv2.putText(img, "end",   (pts[-1][0]+6, pts[-1][1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 100, 255), 1)

    # ── OBB footprints ────────────────────────────────────────────────────────
    names_seen = []
    for _, row in obbs.sort_values("prob").iterrows():   # draw low-conf first
        corners = obb_corners_world(
            row["tx_world_object"], row["ty_world_object"], row["tz_world_object"],
            row["qw_world_object"], row["qx_world_object"],
            row["qy_world_object"], row["qz_world_object"],
            row["scale_x"], row["scale_y"], row["scale_z"],
        )
        color = get_color(row["name"])
        pts_px = np.array([w2p(c[0], c[1]) for c in corners], dtype=np.int32)

        if (pts_px[:,0].max() < 0 or pts_px[:,0].min() >= view_size or
                pts_px[:,1].max() < 0 or pts_px[:,1].min() >= view_size):
            continue

        hull = cv2.convexHull(pts_px)
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, hull, color)
        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
        cv2.polylines(img, [hull], True, color, 2, cv2.LINE_AA)

        # Label: name + count
        cx_px, cy_px = w2p(row["tx_world_object"], row["ty_world_object"])
        label = f"{row['name'][:6]} x{int(row['count'])}"
        cv2.putText(img, label, (cx_px - 14, cy_px + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
        names_seen.append(row["name"])

    # ── Decorations ───────────────────────────────────────────────────────────
    # Scale bar (1 m)
    bar_px = int(scale)
    bx, by = 16, view_size - 22
    cv2.line(img, (bx, by), (bx + bar_px, by), (180, 180, 180), 2)
    cv2.line(img, (bx, by-4), (bx, by+4), (180, 180, 180), 1)
    cv2.line(img, (bx+bar_px, by-4), (bx+bar_px, by+4), (180, 180, 180), 1)
    cv2.putText(img, "1 m", (bx, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Axis arrows from scene centroid
    origin_px = w2p(scene_center[0], scene_center[1])
    ax_tip = w2p(scene_center[0] + 1.0, scene_center[1])
    ay_tip = w2p(scene_center[0], scene_center[1] + 1.0)
    cv2.arrowedLine(img, origin_px, ax_tip, (100, 100, 220), 2,
                    cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(img, origin_px, ay_tip, (100, 220, 100), 2,
                    cv2.LINE_AA, tipLength=0.3)
    cv2.putText(img, "+X", (ax_tip[0]+4, ax_tip[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 220), 1)
    cv2.putText(img, "+Y", (ay_tip[0]+4, ay_tip[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 220, 100), 1)

    # Title
    cv2.putText(img, f"{title}  |  Top-down (world XY, Z-up)  |  {len(obbs)} objects",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    draw_legend(img, names_seen)

    return img


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obbs",    required=True, help="scene_obbs.csv")
    ap.add_argument("--traj",    required=True, help="closed_loop_trajectory.csv")
    ap.add_argument("--output",  required=True, help="output image path (.jpg)")
    ap.add_argument("--title",   default="",    help="scene title shown in image")
    ap.add_argument("--size",    type=int, default=1024, help="image size in px (square)")
    ap.add_argument("--padding", type=float, default=1.5,
                    help="extra metres of padding around objects (default 1.5)")
    ap.add_argument("--traj-stride", type=int, default=10,
                    help="sample every Nth trajectory pose (default 10)")
    args = ap.parse_args()

    obbs = pd.read_csv(args.obbs)
    print(f"Loaded {len(obbs)} fused OBBs from {args.obbs}")

    traj_df = pd.read_csv(args.traj)
    traj_xy = list(zip(
        traj_df["tx_world_device"].values[::args.traj_stride],
        traj_df["ty_world_device"].values[::args.traj_stride],
    ))
    print(f"Loaded {len(traj_xy)} trajectory points")

    img = render_scene_topdown(obbs, traj_xy, args.title, args.size, args.padding)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
