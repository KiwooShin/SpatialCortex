#!/usr/bin/env python3
"""
SpatialCortex — COLMAP SfM pipeline for Project Aria VRS recordings.

Extracts RGB frames from a VRS file, undistorts them using the device's
factory calibration (fisheye → pinhole), then runs the full COLMAP
Structure-from-Motion pipeline to produce a sparse point cloud and camera
poses in the format expected by 3D Gaussian Splatting.

Output layout:
    <output>/
    ├── images/          ← undistorted JPEG frames
    ├── database.db      ← COLMAP feature database
    └── sparse/0/
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt  ← sparse point cloud (input for 3DGS)

Usage:
    /path/to/miniconda3/envs/aria/bin/python3 scripts/run_colmap.py \\
        --vrs /path/to/recording.vrs \\
        [--output data/colmap] \\
        [--every-nth 5] \\
        [--width 960] \\
        [--colmap colmap]

Install COLMAP first:
    brew install colmap          # macOS
    sudo apt install colmap      # Ubuntu

COLMAP → ORB-SLAM3 migration note (plan.md):
    ORB-SLAM3 will replace this script for Phase 2. It supports fisheye
    cameras natively (no undistortion needed) and fuses IMU data for
    loop-closed trajectories. The downstream 3DGS pipeline stays the same.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId


# ── Utilities ─────────────────────────────────────────────────────────────────

def check_colmap(colmap_bin: str) -> None:
    if shutil.which(colmap_bin) is None:
        print(f"Error: '{colmap_bin}' not found in PATH.", file=sys.stderr)
        print("Install with:", file=sys.stderr)
        print("  macOS:  brew install colmap", file=sys.stderr)
        print("  Ubuntu: sudo apt install colmap", file=sys.stderr)
        sys.exit(1)


def run(cmd: list, desc: str) -> None:
    print(f"\n[COLMAP] {desc}")
    print("  " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        print(f"Error: COLMAP step '{desc}' failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ done")


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(vrs_path: str, out_dir: Path, every_nth: int, target_width: int) -> dict:
    """
    Extract every Nth RGB frame from the VRS, undistort using Aria factory
    calibration (fisheye → pinhole), resize to target_width, save as JPEG.

    Returns the linear camera calibration parameters for COLMAP.
    """
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    provider = data_provider.create_vrs_data_provider(vrs_path)
    device_calib = provider.get_device_calibration()

    rgb_id = StreamId("214-1")
    rgb_label = provider.get_label_from_stream_id(rgb_id)
    cam_calib = device_calib.get_camera_calib(rgb_label)

    # Original dimensions and focal length
    orig_w, orig_h = cam_calib.get_image_size()
    focal_orig = float(np.mean(cam_calib.get_focal_lengths()))

    # Compute output dimensions (preserve aspect ratio, ensure even)
    scale = target_width / orig_w
    out_w = target_width + (target_width % 2)
    out_h = int(orig_h * scale)
    out_h = out_h + (out_h % 2)
    focal_out = focal_orig * scale

    print(f"  Original:   {orig_w}x{orig_h}, focal={focal_orig:.1f}px")
    print(f"  Undistorted: {out_w}x{out_h}, focal={focal_out:.1f}px (scale={scale:.3f})")

    # Linear (pinhole) calibration at original resolution — used for undistortion
    linear_calib_orig = calibration.get_linear_camera_calibration(
        orig_w, orig_h, focal_orig, rgb_label,
        cam_calib.get_transform_device_camera(),
    )

    total = provider.get_num_data(rgb_id)
    opt = provider.get_default_deliver_queued_options()
    opt.deactivate_stream_all()
    opt.activate_stream(rgb_id)

    saved = []
    frame_idx = 0
    extracted = 0
    for data in provider.deliver_queued_sensor_data(opt):
        if data.sensor_data_type() != SensorDataType.IMAGE:
            continue

        if frame_idx % every_nth == 0:
            raw = data.image_data_and_record()[0].to_numpy_array()

            # Undistort: fisheye → pinhole using factory calibration
            undistorted = calibration.distort_by_calibration(
                raw, linear_calib_orig, cam_calib
            )

            # Resize to target resolution
            img = Image.fromarray(undistorted)
            if (out_w, out_h) != (orig_w, orig_h):
                img = img.resize((out_w, out_h), Image.LANCZOS)

            fname = f"frame_{frame_idx:05d}.jpg"
            img.save(img_dir / fname, quality=90)
            ts_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
            saved.append({"file": fname, "frame": frame_idx, "timestamp_ns": ts_ns})
            extracted += 1

            if extracted % 20 == 0 or extracted == 1:
                print(f"  {extracted} frames extracted (vrs frame {frame_idx}/{total})")

        frame_idx += 1

    print(f"  Extracted {extracted} frames → {img_dir}")

    # Write manifest for downstream use
    manifest = {
        "width": out_w,
        "height": out_h,
        "focal_length_px": focal_out,
        "cx": out_w / 2.0,
        "cy": out_h / 2.0,
        "every_nth": every_nth,
        "frames": saved,
    }
    manifest_path = out_dir / "frames_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ── COLMAP pipeline ───────────────────────────────────────────────────────────

def run_colmap(colmap_bin: str, out_dir: Path, manifest: dict) -> None:
    db = out_dir / "database.db"
    img_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    w = manifest["width"]
    h = manifest["height"]
    f = manifest["focal_length_px"]
    cx = manifest["cx"]
    cy = manifest["cy"]

    # PINHOLE model: fx, fy, cx, cy  (undistorted images → no distortion params)
    camera_params = f"{f},{f},{cx},{cy}"

    # 1. Feature extraction with known intrinsics
    run(
        [
            colmap_bin, "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(img_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.camera_params", camera_params,
        ],
        f"Feature extraction (PINHOLE {w}x{h}, f={f:.1f})",
    )

    # 2. Sequential matcher — best for video (matches temporally nearby frames)
    run(
        [
            colmap_bin, "sequential_matcher",
            "--database_path", str(db),
            "--SequentialMatching.overlap", "15",
            "--SequentialMatching.loop_detection", "1",
        ],
        "Sequential feature matching (overlap=15, loop detection on)",
    )

    # 3. Sparse reconstruction
    run(
        [
            colmap_bin, "mapper",
            "--database_path", str(db),
            "--image_path", str(img_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.num_threads", "8",
        ],
        "Sparse reconstruction (mapper)",
    )

    # Find the largest reconstructed model (highest image count)
    model_dirs = sorted(sparse_dir.iterdir(), key=lambda p: p.name)
    if not model_dirs:
        print("Error: COLMAP mapper produced no models.", file=sys.stderr)
        print("  Too few feature matches — try a lower --every-nth value.", file=sys.stderr)
        sys.exit(1)

    best_model = model_dirs[0]  # mapper outputs 0, 1, 2... in registration order

    # 4. Export to TXT format (required by gaussian-splatting)
    run(
        [
            colmap_bin, "model_converter",
            "--input_path", str(best_model),
            "--output_path", str(best_model),
            "--output_type", "TXT",
        ],
        f"Export to TXT format → {best_model}",
    )

    print(f"\n  Sparse model: {best_model}")
    for f_name in ["cameras.txt", "images.txt", "points3D.txt"]:
        p = best_model / f_name
        size = p.stat().st_size if p.exists() else 0
        print(f"    {f_name}: {size:,} bytes")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(out_dir: Path) -> None:
    sparse_dir = out_dir / "sparse" / "0"
    print("\n" + "="*60)
    print("COLMAP reconstruction complete.")
    print(f"Output: {out_dir}/sparse/0/")
    print()
    print("Next steps:")
    print("  3D Gaussian Splatting:")
    print(f"    python train.py -s {out_dir}")
    print()
    print("  ORB-SLAM3 (future — Phase 2 SLAM):")
    print("    See plan.md → SLAM Point Cloud Strategy")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract Aria VRS frames and run COLMAP SfM pipeline."
    )
    parser.add_argument("--vrs", required=True, help="Path to .vrs recording")
    parser.add_argument("--output", default="data/colmap", help="Output directory")
    parser.add_argument("--every-nth", type=int, default=5,
                        help="Extract every Nth RGB frame (default: 5 → ~80 frames from 400)")
    parser.add_argument("--width", type=int, default=960,
                        help="Output image width in pixels (default: 960)")
    parser.add_argument("--colmap", default="colmap",
                        help="Path to colmap binary (default: colmap)")
    args = parser.parse_args()

    if not Path(args.vrs).exists():
        print(f"Error: VRS file not found: {args.vrs}", file=sys.stderr)
        sys.exit(1)

    check_colmap(args.colmap)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"VRS:       {args.vrs}")
    print(f"Output:    {out_dir}")
    print(f"Every Nth: {args.every_nth} (≈{400 // args.every_nth} frames)")
    print(f"Width:     {args.width}px")

    print("\n── Step 1: Extract & undistort frames ──────────────────────")
    manifest = extract_frames(args.vrs, out_dir, args.every_nth, args.width)

    print("\n── Step 2: COLMAP SfM ──────────────────────────────────────")
    run_colmap(args.colmap, out_dir, manifest)

    print_summary(out_dir)


if __name__ == "__main__":
    main()
