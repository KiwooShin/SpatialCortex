#!/usr/bin/env python3
"""
Generate synthetic SLAM data for viewer development.
Creates a realistic-looking room trajectory + point cloud.
Replace with real data from Aria/TUM/EuRoC once available.
"""

import json
import math
import random

random.seed(42)


def generate_room_points(cx, cz, w, d, h=2.5, density=200):
    """Generate point cloud for a rectangular room (walls + floor)."""
    points = []
    # Floor
    for _ in range(density):
        points.append({
            "position": [cx + random.uniform(-w/2, w/2), 0.0, cz + random.uniform(-d/2, d/2)],
            "confidence": random.uniform(0.7, 1.0)
        })
    # Walls
    for _ in range(density):
        y = random.uniform(0, h)
        side = random.randint(0, 3)
        if side == 0:
            points.append({"position": [cx - w/2, y, cz + random.uniform(-d/2, d/2)], "confidence": random.uniform(0.5, 0.9)})
        elif side == 1:
            points.append({"position": [cx + w/2, y, cz + random.uniform(-d/2, d/2)], "confidence": random.uniform(0.5, 0.9)})
        elif side == 2:
            points.append({"position": [cx + random.uniform(-w/2, w/2), y, cz - d/2], "confidence": random.uniform(0.5, 0.9)})
        else:
            points.append({"position": [cx + random.uniform(-w/2, w/2), y, cz + d/2], "confidence": random.uniform(0.5, 0.9)})
    return points


def generate_furniture_cluster(cx, cy, cz, r=0.3, n=30):
    """Generate a small cluster of points (simulates a piece of furniture)."""
    points = []
    for _ in range(n):
        dx = random.gauss(0, r)
        dy = random.gauss(0, r * 0.5)
        dz = random.gauss(0, r)
        points.append({
            "position": [cx + dx, cy + abs(dy), cz + dz],
            "confidence": random.uniform(0.6, 0.95)
        })
    return points


def generate_trajectory(num_frames=200):
    """Generate a trajectory that walks through two rooms in a figure-8 pattern."""
    trajectory = []
    for i in range(num_frames):
        t = i / num_frames
        # Figure-8 path through two rooms
        x = 3.0 * math.sin(2 * math.pi * t)
        z = 2.0 * math.sin(4 * math.pi * t)
        y = 1.2 + 0.05 * math.sin(6 * math.pi * t)  # slight head bob

        # Compute forward direction for orientation (as quaternion)
        dx = 3.0 * 2 * math.pi * math.cos(2 * math.pi * t)
        dz = 2.0 * 4 * math.pi * math.cos(4 * math.pi * t)
        yaw = math.atan2(dx, dz)

        # Yaw-only quaternion (w, x, y, z)
        qw = math.cos(yaw / 2)
        qy = math.sin(yaw / 2)

        trajectory.append({
            "timestamp_ns": int(t * 10_000_000_000),
            "position": [round(x, 4), round(y, 4), round(z, 4)],
            "orientation": [round(qw, 4), 0.0, round(qy, 4), 0.0],
            "is_keyframe": (i % 10 == 0)
        })
    return trajectory


def main():
    # Generate point cloud (two rooms + furniture)
    points = []
    points += generate_room_points(0, 0, 6, 4, h=2.5, density=300)     # Room 1
    points += generate_room_points(0, 0, 6, 4, h=2.5, density=200)     # Extra wall detail
    points += generate_furniture_cluster(-1.5, 0.4, -0.8, r=0.35, n=40)  # Table
    points += generate_furniture_cluster(1.8, 0.3, 1.0, r=0.25, n=30)    # Chair
    points += generate_furniture_cluster(-2.0, 0.6, 1.5, r=0.4, n=50)    # Shelf
    points += generate_furniture_cluster(2.2, 0.5, -1.2, r=0.3, n=35)    # Desk

    # Generate trajectory
    trajectory = generate_trajectory(200)

    data = {
        "metadata": {
            "dataset": "synthetic",
            "sequence": "figure8_room",
            "description": "Synthetic SLAM data: figure-8 trajectory through a room with furniture.",
            "num_frames": len(trajectory),
            "num_points": len(points),
            "coordinate_system": "right-handed, Y-up"
        },
        "trajectory": trajectory,
        "point_cloud": points
    }

    with open("data/sample.json", "w") as f:
        json.dump(data, f, indent=None, separators=(",", ":"))

    print(f"Generated {len(trajectory)} frames, {len(points)} points")
    print(f"Written to data/sample.json ({len(json.dumps(data)) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
