#!/usr/bin/env python3
"""
SpatialCortex — Rerun.io visualization for Project Aria VRS recordings.

Streams in a single timeline:
  - 3D world: VIO trajectory, device pose, glasses outline, SLAM point cloud
  - 2D RGB camera image (live per frame)
  - 2D SLAM front-left / front-right images
  - Placeholder entity paths for future phases:
      world/objects/<label>        <- 3D bounding boxes (Phase 2)
      world/navigation/path        <- navigation waypoints (Phase 3)
      world/navigation/arrow       <- AR direction arrow (Phase 3)

Usage:
    /path/to/miniconda3/envs/aria/bin/python3 scripts/visualize_vrs.py \\
        --vrs /path/to/recording.vrs \\
        [--rrd data/session.rrd] \\
        [--downsample 4] \\
        [--jpeg-quality 75]

Requires: projectaria-tools + rerun-sdk  (conda env: aria)
Adapted from: projectaria_tools/tools/viewer_mps/rerun_viewer_mps.py
              projectaria_tools/utils/rerun_helpers.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from tqdm import tqdm

from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.calibration import DeviceVersion
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId


# ── Helpers adapted from rerun_helpers.py ─────────────────────────────────────

def se3_to_transform3d(pose, axis_length: float = 0.05) -> rr.Transform3D:
    """Convert Sophus SE3 pose to Rerun Transform3D."""
    quat_xyzw = np.roll(pose.rotation().to_quat()[0], -1)  # [w,x,y,z] → [x,y,z,w]
    return rr.Transform3D(
        translation=pose.translation()[0],
        rotation=rr.Quaternion(xyzw=quat_xyzw),
        axis_length=axis_length,
    )


def aria_glasses_outline(device_calibration) -> list:
    """Return point strip tracing the glasses frame outline."""
    version = device_calibration.get_device_version()
    if version == DeviceVersion.Gen2:
        left_label, right_label = "slam-front-left", "slam-front-right"
    else:
        left_label, right_label = "camera-slam-left", "camera-slam-right"

    outline_labels = [
        "mic5", left_label, "mic2", "mic1", "baro0", "mic1",
        left_label, right_label, "mic0", "baro0", right_label, "mic6",
    ]
    pts = []
    for label in outline_labels:
        use_cad = "mic" not in label and "baro" not in label
        T = device_calibration.get_transform_device_sensor(label, get_cad_value=use_cad)
        pts.append(T.translation()[0])
    return pts


# ── Static scene logging ───────────────────────────────────────────────────────

def log_trajectory_static(provider, vio_stream_id) -> None:
    """Log full VIO trajectory as a static line strip."""
    num = provider.get_num_data(vio_stream_id)
    positions = []
    for i in range(num):
        rec = provider.get_vio_data_by_index(vio_stream_id, i)
        t = rec.transform_odometry_bodyimu.translation()[0]
        positions.append(t)

    rr.log(
        "world/trajectory",
        rr.LineStrips3D(positions, radii=0.005, colors=[[100, 180, 255]]),
        static=True,
    )
    # Keyframe markers every 10th pose
    keyframes = [positions[i] for i in range(0, len(positions), 10)]
    rr.log(
        "world/keyframes",
        rr.Points3D(keyframes, radii=0.012, colors=[[255, 170, 68]]),
        static=True,
    )
    print(f"  Trajectory: {len(positions)} poses, {len(keyframes)} keyframes")


def log_glasses_outline_static(device_calibration) -> None:
    """Log glasses wireframe as static geometry."""
    pts = aria_glasses_outline(device_calibration)
    rr.log(
        "world/device/glasses_outline",
        rr.LineStrips3D([pts], colors=[[180, 220, 255]]),
        static=True,
    )


def log_camera_intrinsics_static(device_calibration, stream_labels: list, downsample: int) -> None:
    """Log pinhole camera models for all active streams."""
    for label in stream_labels:
        cam_calib = device_calibration.get_camera_calib(label)
        if cam_calib is None:
            continue
        w, h = cam_calib.get_image_size()
        fx = cam_calib.get_focal_lengths()[0]
        rr.log(
            f"world/device/{label}",
            rr.Pinhole(
                resolution=[w / downsample, h / downsample],
                focal_length=float(fx / downsample),
            ),
            static=True,
        )


# ── Per-frame dynamic logging ──────────────────────────────────────────────────

def log_device_pose(provider, vio_stream_id, device_time_ns: int) -> None:
    """Interpolate VIO pose at device_time_ns and log device transform."""
    num = provider.get_num_data(vio_stream_id)
    # Binary search for nearest VIO record
    lo, hi = 0, num - 1
    while lo < hi:
        mid = (lo + hi) // 2
        t = provider.get_vio_data_by_index(vio_stream_id, mid).capture_timestamp_ns
        if t < device_time_ns:
            lo = mid + 1
        else:
            hi = mid
    rec = provider.get_vio_data_by_index(vio_stream_id, max(0, lo - 1))
    T = rec.transform_odometry_bodyimu
    rr.log("world/device", se3_to_transform3d(T))


def log_image(data, stream_label: str, downsample: int, jpeg_quality: int) -> None:
    """Log a camera image frame."""
    if data.sensor_data_type() != SensorDataType.IMAGE:
        return
    img = data.image_data_and_record()[0].to_numpy_array()
    if downsample > 1:
        img = img[::downsample, ::downsample]
    rr.log(
        f"world/device/{stream_label}",
        rr.Image(img).compress(jpeg_quality=jpeg_quality),
    )


# ── Placeholder entity paths for future phases ────────────────────────────────

def log_placeholder_entities() -> None:
    """
    Log empty placeholder paths so they appear in the blueprint panel.
    Phase 2: object bounding boxes will populate world/objects/<label>
    Phase 3: navigation path/arrow will populate world/navigation/
    """
    rr.log("world/objects", rr.Clear.recursive(), static=True)
    rr.log("world/navigation/path", rr.Clear.flat(), static=True)
    rr.log("world/navigation/arrow", rr.Clear.flat(), static=True)


# ── Blueprint layout ───────────────────────────────────────────────────────────

def make_blueprint(rgb_label: str, slam_labels: list) -> rrb.Blueprint:
    slam_views = [
        rrb.Spatial2DView(name=label, origin=f"world/device/{label}")
        for label in slam_labels[:2]  # front-left and front-right only
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D World",
                origin="world",
                contents=[
                    "+ $origin/**",
                ],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="RGB Camera",
                    origin=f"world/device/{rgb_label}",
                ),
                rrb.Horizontal(*slam_views) if slam_views else rrb.TextDocumentView(name="SLAM"),
                row_shares=[2, 1],
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpatialCortex Rerun.io VRS viewer")
    parser.add_argument("--vrs", required=True, help="Path to .vrs recording")
    parser.add_argument("--rrd", default=None, help="Save .rrd file (omit for live view)")
    parser.add_argument("--downsample", type=int, default=4, help="Image downsample factor")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG compression quality")
    args = parser.parse_args()

    if not Path(args.vrs).exists():
        print(f"Error: VRS file not found: {args.vrs}", file=sys.stderr)
        sys.exit(1)

    # ── Open data provider ────────────────────────────────────────────────────
    provider = data_provider.create_vrs_data_provider(args.vrs)
    device_calibration = provider.get_device_calibration()
    device_version = device_calibration.get_device_version()
    is_gen2 = device_version == DeviceVersion.Gen2

    # Stream IDs
    rgb_stream_id = StreamId("214-1")
    rgb_label = provider.get_label_from_stream_id(rgb_stream_id)

    vio_stream_id = provider.get_stream_id_from_label("vio")

    slam_stream_ids = []
    slam_labels = []
    for label in (["slam-front-left", "slam-front-right"] if is_gen2 else ["camera-slam-left", "camera-slam-right"]):
        sid = provider.get_stream_id_from_label(label)
        if sid is not None:
            slam_stream_ids.append(sid)
            slam_labels.append(label)

    # ── Init Rerun ────────────────────────────────────────────────────────────
    blueprint = make_blueprint(rgb_label, slam_labels)
    if args.rrd:
        rr.init("SpatialCortex", spawn=False)
        rr.save(args.rrd, default_blueprint=blueprint)
        print(f"Saving to {args.rrd} ...")
    else:
        rr.init("SpatialCortex", spawn=True, default_blueprint=blueprint)

    # World coordinate system: right-handed, Z up
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ── Static geometry ───────────────────────────────────────────────────────
    print("Logging static geometry...")
    log_trajectory_static(provider, vio_stream_id)
    log_glasses_outline_static(device_calibration)
    log_camera_intrinsics_static(device_calibration, [rgb_label] + slam_labels, args.downsample)
    log_placeholder_entities()

    # ── Per-frame streaming ───────────────────────────────────────────────────
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(rgb_stream_id)
    for sid in slam_stream_ids:
        deliver_option.activate_stream(sid)

    total_frames = provider.get_num_data(rgb_stream_id)
    print(f"Streaming {total_frames} RGB frames + {len(slam_labels)} SLAM streams...")

    for data in tqdm(provider.deliver_queued_sensor_data(deliver_option), total=total_frames):
        device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
        rr.set_time_nanos("device_time", device_time_ns)
        rr.set_time_sequence("frame", device_time_ns)

        stream_id = data.stream_id()
        if stream_id == rgb_stream_id:
            log_device_pose(provider, vio_stream_id, device_time_ns)
            log_image(data, rgb_label, args.downsample, args.jpeg_quality)
        elif stream_id in slam_stream_ids:
            idx = slam_stream_ids.index(stream_id)
            log_image(data, slam_labels[idx], args.downsample, args.jpeg_quality)

    print("Done.")
    if args.rrd:
        print(f"Saved → {args.rrd}")
    else:
        print("Rerun viewer is open. Close the window to exit.")


if __name__ == "__main__":
    main()
