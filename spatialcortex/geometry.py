"""
Shared 3D geometry helpers for Aria/EFM3D spatial memory pipelines.

Conventions (Aria / EFM3D):
  - World frame: Z-up, gravity-aligned.
  - Device frame: origin at IMU; Z roughly forward, Y roughly up.
  - Camera frame: +Z forward into scene.
  - Aria RGB sensor image is 90° CCW from upright.
    After CW-90° rotation of an N×N square image:
        raw pixel (u, v)  →  rotated pixel (N-1-v, u)

All trajectory CSVs have columns:
    tracking_timestamp_us, tx/ty/tz_world_device, qw/qx/qy/qz_world_device
"""

from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

# ── OBB edge connectivity (from EFM3D BB3D_LINE_ORDERS) ──────────────────────
BB3D_LINE_ORDERS: list[list[int]] = [
    [0, 1], [1, 2], [2, 3], [3, 0],   # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],   # top face
    [0, 4], [1, 5], [2, 6], [3, 7],   # vertical pillars
]


# ── Rotation ──────────────────────────────────────────────────────────────────

def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Unit quaternion [qw, qx, qy, qz] → 3×3 rotation matrix."""
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
    ])


# ── OBB corners ───────────────────────────────────────────────────────────────

def obb_corners_world(tx: float, ty: float, tz: float,
                      qw: float, qx: float, qy: float, qz: float,
                      sx: float, sy: float, sz: float) -> np.ndarray:
    """8 OBB corner points in world frame.

    Parameters
    ----------
    tx, ty, tz : OBB centre (world coordinates)
    qw, qx, qy, qz : rotation quaternion (world ← object)
    sx, sy, sz : **full** side lengths (not half-extents)

    Returns
    -------
    corners : ndarray of shape (8, 3)
    """
    R = quat_to_rotmat(qw, qx, qy, qz)
    center = np.array([tx, ty, tz])
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    xs, ys, zs = [-hx, hx], [-hy, hy], [-hz, hz]
    ids = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
           (0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    local = np.array([[xs[xi], ys[yi], zs[zi]] for xi, yi, zi in ids])
    return center + local @ R.T    # (8, 3)


# ── Trajectory ────────────────────────────────────────────────────────────────

def load_trajectory(traj_csv: str | Path):
    """Load a closed-loop SLAM trajectory CSV.

    Returns
    -------
    times_us : np.ndarray[int64] — tracking_timestamp_us (μs, sorted)
    Rs       : list[np.ndarray] — R_world_device (3×3) per pose
    ts_pos   : list[np.ndarray] — world position (3,) per pose
    """
    df = pd.read_csv(traj_csv)
    times_us = df["tracking_timestamp_us"].values.astype(np.int64)
    Rs, ts_pos = [], []
    for _, row in df.iterrows():
        Rs.append(quat_to_rotmat(
            row["qw_world_device"], row["qx_world_device"],
            row["qy_world_device"], row["qz_world_device"],
        ))
        ts_pos.append(np.array([
            row["tx_world_device"],
            row["ty_world_device"],
            row["tz_world_device"],
        ]))
    return times_us, Rs, ts_pos


def interp_pose(times_us: np.ndarray, Rs: list, ts_pos: list, query_ns: int):
    """Nearest-neighbour pose lookup for a nanosecond timestamp.

    Returns
    -------
    R_wd : np.ndarray (3×3) — R_world_device
    t_wd : np.ndarray (3,)  — world position of device
    """
    idx = min(max(bisect_left(times_us, query_ns // 1000), 0), len(times_us) - 1)
    return Rs[idx], ts_pos[idx]


# ── Fisheye projection ────────────────────────────────────────────────────────

def rotate_cw90(u: float, v: float, N: int):
    """Raw fisheye pixel (u, v) → pixel after 90° CW rotation of an N×N image."""
    return N - 1 - v, u


def project_pt(pw: np.ndarray,
               R_wd: np.ndarray, t_wd: np.ndarray,
               R_dc: np.ndarray, t_dc: np.ndarray,
               cam_calib, N: int):
    """Project a world point onto the CW-90°-rotated fisheye image.

    Parameters
    ----------
    pw       : (3,) world-frame point
    R_wd     : (3×3) R_world_device
    t_wd     : (3,) device position in world
    R_dc     : (3×3) R_device_camera
    t_dc     : (3,) camera position in device frame
    cam_calib: Aria CameraCalibration with .project(pc) → (u, v) or None
    N        : image side length (pixels) for CW-90° transform

    Returns
    -------
    (u_rot, v_rot) pixel tuple, or None if behind camera / outside image.
    """
    p_dev = R_wd.T @ (pw - t_wd)
    pc    = R_dc.T @ (p_dev - t_dc)
    if pc[2] <= 0.05:
        return None
    uv = cam_calib.project(pc)
    if uv is None:
        return None
    return rotate_cw90(uv[0], uv[1], N)
