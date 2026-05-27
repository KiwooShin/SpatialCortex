#!/usr/bin/env python3
"""
Extract evenly-spaced RGB keyframes from a Project Aria .vrs file.
Saves downscaled JPEGs to data/keyframes/ for display in the web viewer.

Usage:
    /path/to/miniconda3/envs/aria/bin/python3 scripts/extract_keyframes.py \
        --vrs /path/to/recording.vrs \
        [--n 12] [--width 480] [--output data/keyframes]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrs", required=True)
    parser.add_argument("--n", type=int, default=12, help="Number of keyframes to extract")
    parser.add_argument("--width", type=int, default=480, help="Output width in pixels")
    parser.add_argument("--output", default="data/keyframes")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = data_provider.create_vrs_data_provider(args.vrs)
    rgb_id = StreamId("214-1")
    total = provider.get_num_data(rgb_id)
    target_indices = set(round(i * (total - 1) / (args.n - 1)) for i in range(args.n))

    opt = provider.get_default_deliver_queued_options()
    opt.deactivate_stream_all()
    opt.activate_stream(rgb_id)

    saved = []
    frame_idx = 0
    for data in provider.deliver_queued_sensor_data(opt):
        if data.sensor_data_type() != SensorDataType.IMAGE:
            continue
        if frame_idx in target_indices:
            img_arr = data.image_data_and_record()[0].to_numpy_array()
            img = Image.fromarray(img_arr)
            # Maintain aspect ratio
            w, h = img.size
            new_h = int(args.width * h / w)
            img = img.resize((args.width, new_h), Image.LANCZOS)
            ts_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
            fname = f"frame_{frame_idx:04d}.jpg"
            img.save(out_dir / fname, quality=82)
            saved.append({"file": fname, "frame": frame_idx, "timestamp_ns": ts_ns})
            print(f"  Saved {fname}  ({frame_idx}/{total})")
        frame_idx += 1

    # Write manifest so the viewer can load them dynamically
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nExtracted {len(saved)} keyframes → {out_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
