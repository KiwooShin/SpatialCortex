#!/usr/bin/env python3
"""
Convert TUM RGB-D groundtruth data to SLAM viewer JSON format.

Usage:
    python3 tum_to_json.py \
        --groundtruth path/to/groundtruth.txt \
        --output data/tum_session.json \
        [--pointcloud path/to/pointcloud.ply]

TUM RGB-D format reference:
    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
"""

import argparse
import json
import re
import struct
from pathlib import Path


def read_tum_trajectory(filepath: str, subsample: int = 1) -> list[dict]:
    """
    Read TUM groundtruth.txt.
    Format: timestamp tx ty tz qx qy qz qw
    """
    trajectory = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            if i % subsample != 0:
                continue

            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            trajectory.append({
                "timestamp_ns": int(ts * 1e9),
                "position": [round(tx, 4), round(ty, 4), round(tz, 4)],
                "orientation": [round(qw, 4), round(qx, 4), round(qy, 4), round(qz, 4)],
                "is_keyframe": (len(trajectory) % 15 == 0),
            })
    return trajectory


def read_ply_points(filepath: str, max_points: int = 100000) -> list[dict]:
    """Read a PLY point cloud file (ASCII or binary_little_endian)."""
    points = []

    with open(filepath, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header
        num_vertices = 0
        is_binary = False
        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            if "binary_little_endian" in line:
                is_binary = True

        num_to_read = min(num_vertices, max_points)

        if is_binary:
            for _ in range(num_to_read):
                data = f.read(12)  # 3 floats = 12 bytes (minimum)
                if len(data) < 12:
                    break
                x, y, z = struct.unpack("<fff", data[:12])
                points.append({"position": [round(x, 4), round(y, 4), round(z, 4)], "confidence": 0.8})
                # Skip remaining bytes per vertex if any (color etc)
                # This is simplified - real PLY parsing should check properties
        else:
            f.seek(0)
            in_header = True
            for line_bytes in f:
                line = line_bytes.decode("ascii", errors="replace").strip()
                if in_header:
                    if line == "end_header":
                        in_header = False
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append({"position": [round(x, 4), round(y, 4), round(z, 4)], "confidence": 0.8})
                if len(points) >= max_points:
                    break

    return points


def main():
    parser = argparse.ArgumentParser(description="Convert TUM RGB-D data to viewer JSON")
    parser.add_argument("--groundtruth", required=True, help="Path to groundtruth.txt")
    parser.add_argument("--pointcloud", default=None, help="Path to pointcloud.ply (optional)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--subsample", type=int, default=3, help="Take every Nth trajectory pose")
    args = parser.parse_args()

    print(f"Reading trajectory: {args.groundtruth}")
    trajectory = read_tum_trajectory(args.groundtruth, args.subsample)
    print(f"  {len(trajectory)} poses")

    points = []
    if args.pointcloud:
        print(f"Reading point cloud: {args.pointcloud}")
        points = read_ply_points(args.pointcloud, args.max_points)
        print(f"  {len(points)} points")

    seq_name = Path(args.groundtruth).parent.name or "tum_sequence"

    data = {
        "metadata": {
            "dataset": "tum_rgbd",
            "sequence": seq_name,
            "num_frames": len(trajectory),
            "num_points": len(points),
            "coordinate_system": "right-handed, Y-up",
        },
        "trajectory": trajectory,
        "point_cloud": points,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = output_path.stat().st_size / 1024
    print(f"Written to {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
