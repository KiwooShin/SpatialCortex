#!/usr/bin/env python3
"""
Overlay COLMAP sparse 3D points onto the original images.

Reads sparse/0/images.txt and sparse/0/points3D.txt, draws matched 2D
observations (colored by 3D point RGB or by reprojection error) onto
each registered image, and saves results to <output>/overlays/.

Usage:
    python3 scripts/visualize_colmap_points.py \
        [--colmap data/colmap] \
        [--output data/colmap/overlays] \
        [--dot-size 4] \
        [--color rgb|error|uniform]
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── COLMAP file parsers ───────────────────────────────────────────────────────

def read_points3d(path: Path) -> dict:
    """Returns {point3d_id: (x, y, z, r, g, b, error)}"""
    points = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pid   = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            error   = float(parts[7])
            points[pid] = (x, y, z, r, g, b, error)
    return points


def read_images(path: Path) -> list:
    """
    Returns list of dicts:
      {name, observations: [(u, v, point3d_id), ...]}
    observations with point3d_id == -1 are unmatched features (skipped).
    """
    images = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]

    i = 0
    while i < len(lines) - 1:
        header = lines[i].split()
        name = header[9]
        obs_parts = lines[i + 1].split()
        obs = []
        for j in range(0, len(obs_parts) - 2, 3):
            u    = float(obs_parts[j])
            v    = float(obs_parts[j + 1])
            pid  = int(obs_parts[j + 2])
            if pid != -1:
                obs.append((u, v, pid))
        images.append({"name": name, "observations": obs})
        i += 2
    return images


# ── Coloring ──────────────────────────────────────────────────────────────────

def error_to_color(error: float, max_error: float = 2.0) -> tuple:
    """Low error → green, high error → red."""
    t = min(error / max_error, 1.0)
    return (int(255 * t), int(255 * (1 - t)), 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap",  default="data/colmap")
    parser.add_argument("--output",  default="data/colmap/overlays")
    parser.add_argument("--dot-size", type=int, default=4)
    parser.add_argument("--color", choices=["rgb", "error", "uniform"],
                        default="rgb", help="Point coloring mode")
    args = parser.parse_args()

    colmap_dir  = Path(args.colmap)
    sparse_dir  = colmap_dir / "sparse" / "0"
    img_dir     = colmap_dir / "images"
    out_dir     = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading points3D …")
    points3d = read_points3d(sparse_dir / "points3D.txt")
    print(f"  {len(points3d):,} 3D points loaded")

    print(f"Reading images …")
    images = read_images(sparse_dir / "images.txt")
    print(f"  {len(images)} registered images")

    # Compute max reprojection error for color scaling
    max_err = max((p[6] for p in points3d.values()), default=2.0)

    r = args.dot_size
    saved = 0

    for entry in images:
        img_path = img_dir / entry["name"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        obs = entry["observations"]
        for (u, v, pid) in obs:
            if pid not in points3d:
                continue
            pt = points3d[pid]

            if args.color == "rgb":
                color = (pt[3], pt[4], pt[5])          # COLMAP RGB from scene
            elif args.color == "error":
                color = error_to_color(pt[6], max_err)  # green=good, red=bad
            else:
                color = (100, 220, 255)                 # uniform cyan

            draw.ellipse([u - r, v - r, u + r, v + r],
                         fill=color, outline=(0, 0, 0))

        # Overlay stats
        w, h = img.size
        draw.rectangle([0, h - 22, w, h], fill=(0, 0, 0, 180))
        draw.text((6, h - 18),
                  f"{entry['name']}  |  {len(obs)} pts  |  color={args.color}",
                  fill=(180, 220, 255))

        out_path = out_dir / entry["name"]
        img.save(out_path, quality=90)
        saved += 1
        if saved % 10 == 0 or saved == 1:
            print(f"  {saved}/{len(images)}  {entry['name']}  ({len(obs)} pts)")

    print(f"\nSaved {saved} overlay images → {out_dir}/")
    print(f"Open any image in {out_dir}/ to inspect point coverage.")


if __name__ == "__main__":
    main()
