#!/usr/bin/env python3
"""
Extract all RGB frames from a Project Aria .vrs file and encode as MP4.
Output: data/rgb_video.mp4  (gitignored — regenerate from VRS as needed)

Usage:
    /path/to/miniconda3/envs/aria/bin/python3 scripts/extract_video.py \
        --vrs /path/to/recording.vrs \
        [--output data/rgb_video.mp4] \
        [--width 640] \
        [--fps 10]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrs", required=True)
    parser.add_argument("--output", default="data/rgb_video.mp4")
    parser.add_argument("--width", type=int, default=640, help="Output width in pixels")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provider = data_provider.create_vrs_data_provider(args.vrs)
    rgb_id = StreamId("214-1")
    total = provider.get_num_data(rgb_id)
    print(f"Found {total} RGB frames")

    # Get frame size from first frame
    opt = provider.get_default_deliver_queued_options()
    opt.deactivate_stream_all()
    opt.activate_stream(rgb_id)

    # Pipe raw BGR frames directly into ffmpeg
    first_frame = None
    for data in provider.deliver_queued_sensor_data(opt):
        if data.sensor_data_type() == SensorDataType.IMAGE:
            arr = data.image_data_and_record()[0].to_numpy_array()
            img = Image.fromarray(arr)
            w, h = img.size
            new_h = int(args.width * h / w)
            # Ensure even dimensions for h264
            new_h = new_h + (new_h % 2)
            new_w = args.width + (args.width % 2)
            first_frame = img.resize((new_w, new_h), Image.LANCZOS)
            break

    if first_frame is None:
        print("No RGB frames found", file=sys.stderr)
        sys.exit(1)

    new_w, new_h = first_frame.size
    print(f"Output size: {new_w}x{new_h} @ {args.fps} fps → {out_path}")

    # Launch ffmpeg process, feed raw RGB frames via stdin
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{new_w}x{new_h}",
        "-r", str(args.fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # Re-open provider to stream all frames
    provider2 = data_provider.create_vrs_data_provider(args.vrs)
    opt2 = provider2.get_default_deliver_queued_options()
    opt2.deactivate_stream_all()
    opt2.activate_stream(rgb_id)

    timestamps = []
    frame_idx = 0
    try:
        for data in provider2.deliver_queued_sensor_data(opt2):
            if data.sensor_data_type() != SensorDataType.IMAGE:
                continue
            arr = data.image_data_and_record()[0].to_numpy_array()
            img = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
            proc.stdin.write(img.tobytes())
            ts_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
            timestamps.append(ts_ns)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  {frame_idx}/{total}")
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        print(f"ffmpeg exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(1)

    # Write timestamp manifest so viewer can map video time → trajectory frame
    import json
    manifest = {
        "fps": args.fps,
        "width": new_w,
        "height": new_h,
        "frame_count": len(timestamps),
        "timestamps_ns": timestamps,
    }
    manifest_path = out_path.with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\nDone. {frame_idx} frames → {out_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
