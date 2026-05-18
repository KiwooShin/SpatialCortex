#!/usr/bin/env python3
"""
Convert Meta Project Aria MPS output to SLAM viewer JSON format.

Usage:
    python3 aria_to_json.py \
        --trajectory path/to/closed_loop_trajectory.csv \
        --points path/to/semidense_points.csv.gz \
        --output data/aria_session.json \
        [--max-points 100000] \
        [--confidence-threshold 0.001]

Aria MPS data format reference:
    https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/
"""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path


def read_closed_loop_trajectory(filepath: str) -> list[dict]:
    """
    Read Aria closed_loop_trajectory.csv.

    Columns: tracking_timestamp_us, utc_timestamp_ns,
             tx_world_device, ty_world_device, tz_world_device,
             qw_world_device, qx_world_device, qy_world_device, qz_world_device,
             device_linear_velocity_x, device_linear_velocity_y, device_linear_velocity_z,
             gravity_x, gravity_y, gravity_z,
             quality_score
    """
    trajectory = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            trajectory.append({
                "timestamp_ns": int(float(row["tracking_timestamp_us"]) * 1000),
                "position": [
                    float(row["tx_world_device"]),
                    float(row["ty_world_device"]),
                    float(row["tz_world_device"]),
                ],
                "orientation": [
                    float(row["qw_world_device"]),
                    float(row["qx_world_device"]),
                    float(row["qy_world_device"]),
                    float(row["qz_world_device"]),
                ],
                # Mark every Nth frame as keyframe (Aria doesn't explicitly label keyframes in CSV)
                "is_keyframe": (i % 10 == 0),
            })
    return trajectory


def read_semidense_points(filepath: str, confidence_threshold: float = 0.001, max_points: int = 100000) -> list[dict]:
    """
    Read Aria semidense_points.csv.gz.

    Columns: uid, graph_uid,
             px_world, py_world, pz_world,
             inverse_distance_std, distance_std
    """
    points = []
    opener = gzip.open if filepath.endswith(".gz") else open

    with opener(filepath, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inv_dist_std = float(row.get("inverse_distance_std", 0))
            dist_std = float(row.get("distance_std", 999))

            # Filter low-confidence points
            if inv_dist_std > confidence_threshold:
                continue
            if dist_std > 0.15:
                continue

            points.append({
                "position": [
                    round(float(row["px_world"]), 4),
                    round(float(row["py_world"]), 4),
                    round(float(row["pz_world"]), 4),
                ],
                "confidence": round(max(0, 1.0 - dist_std * 5), 3),
            })

            if len(points) >= max_points:
                print(f"  Capped at {max_points} points", file=sys.stderr)
                break

    return points


def main():
    parser = argparse.ArgumentParser(description="Convert Aria MPS data to viewer JSON")
    parser.add_argument("--trajectory", required=True, help="Path to closed_loop_trajectory.csv")
    parser.add_argument("--points", required=True, help="Path to semidense_points.csv.gz")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-points", type=int, default=100000, help="Max points to include")
    parser.add_argument("--confidence-threshold", type=float, default=0.001, help="Inverse distance std threshold")
    parser.add_argument("--subsample-trajectory", type=int, default=1, help="Take every Nth trajectory pose")
    args = parser.parse_args()

    print(f"Reading trajectory: {args.trajectory}")
    trajectory = read_closed_loop_trajectory(args.trajectory)
    if args.subsample_trajectory > 1:
        trajectory = trajectory[::args.subsample_trajectory]
    print(f"  {len(trajectory)} poses")

    print(f"Reading point cloud: {args.points}")
    points = read_semidense_points(args.points, args.confidence_threshold, args.max_points)
    print(f"  {len(points)} points (after filtering)")

    # Determine sequence name from path
    seq_name = Path(args.trajectory).parent.name or "aria_session"

    data = {
        "metadata": {
            "dataset": "aria_aea",
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
