#!/usr/bin/env python3
"""
Convert a Project Aria .vrs recording to SLAM viewer JSON format.

Extracts on-device VIO poses for the trajectory. Point cloud is synthesized
from the trajectory envelope (walls + floor) since raw VRS files do not
contain MPS semi-dense maps.

Usage:
    /path/to/miniconda3/envs/aria/bin/python3 scripts/vrs_to_json.py \
        --vrs /path/to/recording.vrs \
        --output data/aria_vrs.json \
        [--subsample 1]

Requires: projectaria-tools (conda env: aria)
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path


def extract_trajectory(vrs_path: str, subsample: int) -> list[dict]:
    from projectaria_tools.core import data_provider

    provider = data_provider.create_vrs_data_provider(vrs_path)
    vio_id = provider.get_stream_id_from_label("vio")
    num = provider.get_num_data(vio_id)

    trajectory = []
    for i in range(0, num, subsample):
        rec = provider.get_vio_data_by_index(vio_id, i)
        T = rec.transform_odometry_bodyimu
        # to_quat_and_translation() -> (1,7): [qx,qy,qz,qw, tx,ty,tz]
        raw = T.to_quat_and_translation()[0]
        qx, qy, qz, qw = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        trans = raw[4:7]
        trajectory.append(
            {
                "timestamp_ns": int(rec.capture_timestamp_ns),
                "position": [round(float(trans[0]), 4), round(float(trans[1]), 4), round(float(trans[2]), 4)],  # noqa: E501
                "orientation": [round(qw, 5), round(qx, 5), round(qy, 5), round(qz, 5)],
                "is_keyframe": (i % 10 == 0),
            }
        )

    return trajectory


def synthesize_point_cloud(trajectory: list[dict], n_points: int = 15000, seed: int = 42) -> list[dict]:
    """Generate sparse environment points around the trajectory for visualization."""
    rng = random.Random(seed)
    positions = [f["position"] for f in trajectory]

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    rx = (max(xs) - min(xs)) / 2 + 2.0
    ry = (max(ys) - min(ys)) / 2 + 1.5
    rz = (max(zs) - min(zs)) / 2 + 2.0

    floor_y = min(ys) - 0.05
    ceiling_y = max(ys) + 1.8

    points = []
    # Floor plane
    n_floor = n_points // 3
    for _ in range(n_floor):
        x = cx + rng.uniform(-rx * 1.2, rx * 1.2)
        z = cz + rng.uniform(-rz * 1.2, rz * 1.2)
        y = floor_y + rng.gauss(0, 0.02)
        points.append({"position": [round(x, 3), round(y, 3), round(z, 3)], "confidence": round(rng.uniform(0.6, 1.0), 3)})

    # Walls / environment scatter
    n_env = n_points - n_floor
    for _ in range(n_env):
        # Pick a random trajectory pose to scatter points near
        ref = rng.choice(positions)
        dx = rng.gauss(0, 1.5)
        dy = rng.uniform(0, (ceiling_y - floor_y) * 0.9)
        dz = rng.gauss(0, 1.5)
        dist = math.sqrt(dx * dx + dz * dz)
        # Bias towards surfaces further from center (wall-like)
        if dist > 0.3:
            scale = (rx * 0.9) / dist
            dx *= scale
            dz *= scale
        x = ref[0] + dx
        y = floor_y + dy
        z = ref[2] + dz
        conf = round(rng.uniform(0.3, 0.9), 3)
        points.append({"position": [round(x, 3), round(y, 3), round(z, 3)], "confidence": conf})

    return points


def main():
    parser = argparse.ArgumentParser(description="Convert Aria VRS to viewer JSON")
    parser.add_argument("--vrs", required=True, help="Path to .vrs recording")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--subsample", type=int, default=1, help="Take every Nth VIO pose")
    parser.add_argument("--points", type=int, default=15000, help="Synthetic point cloud size")
    args = parser.parse_args()

    print(f"Reading VIO trajectory: {args.vrs}")
    trajectory = extract_trajectory(args.vrs, args.subsample)
    print(f"  {len(trajectory)} poses")

    print("Synthesizing point cloud…")
    point_cloud = synthesize_point_cloud(trajectory, n_points=args.points)
    print(f"  {len(point_cloud)} points")

    seq_name = Path(args.vrs).stem
    data = {
        "metadata": {
            "dataset": "aria_vrs",
            "sequence": seq_name,
            "num_frames": len(trajectory),
            "num_points": len(point_cloud),
            "coordinate_system": "right-handed (odometry frame)",
        },
        "trajectory": trajectory,
        "point_cloud": point_cloud,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"Written to {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
