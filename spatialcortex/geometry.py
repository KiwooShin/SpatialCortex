"""
Shared 3-D geometry helpers for Aria/EFM3D spatial memory pipelines.

Conventions (Aria / EFM3D)
--------------------------
- World frame: Z-up, gravity-aligned.
- Device frame: origin at IMU.
- Camera frame: +Z forward into scene.
- Aria RGB sensor is 90° CCW from upright; after a CW-90° rotation of the
  N×N square image: raw pixel (u, v)  →  rotated pixel (N-1-v, u).
- Trajectory CSVs have columns:
  ``tracking_timestamp_us``, ``tx/ty/tz_world_device``, ``qw/qx/qy/qz_world_device``

Key exports
-----------
``Trajectory``
    Class wrapping a loaded SLAM trajectory. Provides O(log N) pose lookup
    via ``at_ns(query_ns)`` returning a :class:`~spatialcortex.types.Pose`.

``quat_to_rotmat``, ``obb_corners_world``, ``rotate_cw90``, ``project_pt``
    Standalone helpers (also used internally by ``Trajectory`` and ``OBB``).

``load_trajectory``, ``interp_pose``
    Legacy functional API kept for backward compatibility with existing scripts.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from .types import Mat3, Vec3, Pose


# ── OBB edge connectivity (from EFM3D BB3D_LINE_ORDERS) ──────────────────────
BB3D_LINE_ORDERS: list[list[int]] = [
    [0, 1], [1, 2], [2, 3], [3, 0],   # bottom face ring
    [4, 5], [5, 6], [6, 7], [7, 4],   # top face ring
    [0, 4], [1, 5], [2, 6], [3, 7],   # vertical pillars
]


# ── Rotation ──────────────────────────────────────────────────────────────────

def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> Mat3:
    """Convert a (possibly unnormalised) unit quaternion to a 3×3 rotation matrix.

    Parameters
    ----------
    qw, qx, qy, qz : quaternion components (scalar-first convention)

    Returns
    -------
    R : (3, 3) float64 rotation matrix

    Raises
    ------
    ValueError
        If the quaternion magnitude is near zero (< 1e-10).
    """
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if n < 1e-10:
        raise ValueError(
            f"Near-zero quaternion magnitude ({n:.2e}); cannot normalise."
        )
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


# ── OBB corners ───────────────────────────────────────────────────────────────

def obb_corners_world(
    tx: float, ty: float, tz: float,
    qw: float, qx: float, qy: float, qz: float,
    sx: float, sy: float, sz: float,
) -> np.ndarray:
    """Return the 8 corner points of an OBB in the world frame.

    Parameters
    ----------
    tx, ty, tz       : OBB centre (world coordinates, metres)
    qw, qx, qy, qz  : orientation quaternion (world ← object frame)
    sx, sy, sz       : **full** side lengths (not half-extents), metres

    Returns
    -------
    corners : ndarray of shape (8, 3)
        Bottom face corners at indices 0-3 (CCW), top face at 4-7.
    """
    R = quat_to_rotmat(qw, qx, qy, qz)
    center = np.array([tx, ty, tz], dtype=np.float64)
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    xs, ys, zs = [-hx, hx], [-hy, hy], [-hz, hz]
    ids = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]
    local = np.array(
        [[xs[xi], ys[yi], zs[zi]] for xi, yi, zi in ids],
        dtype=np.float64,
    )
    return center + local @ R.T   # (8, 3)


# ── Trajectory class ──────────────────────────────────────────────────────────

class Trajectory:
    """Typed wrapper around a SLAM closed-loop trajectory.

    Stores parallel arrays of timestamps, rotation matrices, and positions.
    Provides fast O(log N) nearest-neighbour pose lookup via ``at_ns``.

    Attributes
    ----------
    times_us : int64 ndarray of ``tracking_timestamp_us`` (monotone, µs)

    Parameters
    ----------
    times_us : sorted int64 array of microsecond timestamps
    Rs       : list of (3×3) ``R_world_device`` rotation matrices
    ts_pos   : list of (3,)  world position vectors

    Examples
    --------
    >>> traj = Trajectory.from_csv("path/to/closed_loop_trajectory.csv")
    >>> pose = traj.at_ns(1_234_567_890_000)
    >>> print(pose)
    Pose(t=(0.123, 0.456, 0.789))
    """

    __slots__ = ("_times_us", "_Rs", "_ts_pos")

    def __init__(
        self,
        times_us: np.ndarray,
        Rs: list[Mat3],
        ts_pos: list[Vec3],
    ) -> None:
        if not (len(times_us) == len(Rs) == len(ts_pos)):
            raise ValueError(
                f"Array length mismatch: times={len(times_us)}, "
                f"Rs={len(Rs)}, ts_pos={len(ts_pos)}"
            )
        if len(times_us) == 0:
            raise ValueError("Trajectory must contain at least one pose.")
        self._times_us: np.ndarray = np.asarray(times_us, dtype=np.int64)
        self._Rs:    list[Mat3] = list(Rs)
        self._ts_pos: list[Vec3] = list(ts_pos)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str | Path) -> "Trajectory":
        """Load a closed-loop SLAM trajectory from a CSV file.

        Expected columns:
        ``tracking_timestamp_us``,
        ``tx/ty/tz_world_device``,
        ``qw/qx/qy/qz_world_device``
        """
        df = pd.read_csv(path)
        times_us = df["tracking_timestamp_us"].values.astype(np.int64)
        Rs: list[Mat3] = []
        ts_pos: list[Vec3] = []
        for _, row in df.iterrows():
            Rs.append(quat_to_rotmat(
                float(row["qw_world_device"]),
                float(row["qx_world_device"]),
                float(row["qy_world_device"]),
                float(row["qz_world_device"]),
            ))
            ts_pos.append(np.array([
                float(row["tx_world_device"]),
                float(row["ty_world_device"]),
                float(row["tz_world_device"]),
            ], dtype=np.float64))
        return cls(times_us, Rs, ts_pos)

    # ── Properties ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._times_us)

    def __repr__(self) -> str:
        dur_s = self.duration_us / 1e6 if len(self) > 1 else 0.0
        return f"Trajectory(n={len(self)}, duration={dur_s:.1f}s)"

    @property
    def times_us(self) -> np.ndarray:
        """Monotone timestamp array in microseconds (read-only view)."""
        return self._times_us

    @property
    def duration_us(self) -> int:
        """Total trajectory duration in microseconds."""
        return int(self._times_us[-1] - self._times_us[0])

    @property
    def positions_xy(self) -> np.ndarray:
        """(N, 2) array of world XY positions — convenient for top-down plots."""
        return np.array([[p[0], p[1]] for p in self._ts_pos], dtype=np.float64)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def at_ns(self, query_ns: int) -> Pose:
        """Return the nearest-neighbour ``Pose`` for a nanosecond timestamp.

        Clamps to the first / last pose when the query is out of range.
        """
        R, t = self._lookup_raw(query_ns)
        return Pose(R=R, t=t)

    def _lookup_raw(self, query_ns: int) -> tuple[Mat3, Vec3]:
        """Binary-search lookup; returns raw (R, t) matrices."""
        query_us = query_ns // 1000
        idx = bisect_left(self._times_us, query_us)
        idx = max(0, min(idx, len(self._times_us) - 1))
        return self._Rs[idx], self._ts_pos[idx]


# ── Legacy functional API (backward-compatible wrappers) ─────────────────────

def load_trajectory(traj_csv: str | Path) -> tuple[np.ndarray, list, list]:
    """Load a SLAM trajectory CSV → ``(times_us, Rs, ts_pos)``.

    .. deprecated::
        Prefer ``Trajectory.from_csv(path)`` for new code.
    """
    traj = Trajectory.from_csv(traj_csv)
    return traj._times_us, traj._Rs, traj._ts_pos


def interp_pose(
    times_us: np.ndarray,
    Rs: list,
    ts_pos: list,
    query_ns: int,
) -> tuple[Mat3, Vec3]:
    """Nearest-neighbour pose lookup for a nanosecond timestamp.

    .. deprecated::
        Prefer ``Trajectory(times_us, Rs, ts_pos).at_ns(query_ns)`` for new code.
    """
    return Trajectory(times_us, Rs, ts_pos)._lookup_raw(query_ns)


# ── Fisheye projection ────────────────────────────────────────────────────────

def rotate_cw90(u: float, v: float, N: int) -> tuple[float, float]:
    """Map a raw fisheye pixel to its position after a 90° CW rotation.

    For a square N×N image, a 90° CW rotation maps::

        (u, v)  →  (N-1-v, u)

    Parameters
    ----------
    u, v : pixel coordinates in the *unrotated* image
    N    : image side length

    Returns
    -------
    (u_rot, v_rot) : pixel coordinates in the *rotated* image
    """
    return N - 1 - v, u


def project_pt(
    pw: Vec3,
    R_wd: Mat3,
    t_wd: Vec3,
    R_dc: Mat3,
    t_dc: Vec3,
    cam_calib,
    N: int,
) -> tuple[float, float] | None:
    """Project a world-frame point onto the CW-90°-rotated fisheye image.

    Parameters
    ----------
    pw         : (3,) world-frame point
    R_wd, t_wd : device pose in world (``R_world_device``, ``t_world_device``)
    R_dc, t_dc : camera-in-device extrinsics
    cam_calib  : object with ``.project(pc) → (u, v) | None``
    N          : image side length (pixels) for the CW-90° transform

    Returns
    -------
    ``(u_rot, v_rot)`` pixel tuple, or ``None`` if behind camera / outside image.
    """
    p_dev = R_wd.T @ (pw - t_wd)
    pc    = R_dc.T @ (p_dev - t_dc)
    if pc[2] <= 0.05:
        return None
    uv = cam_calib.project(pc)
    if uv is None:
        return None
    return rotate_cw90(uv[0], uv[1], N)
