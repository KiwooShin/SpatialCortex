"""
Tests for spatialcortex.geometry

Coverage
--------
quat_to_rotmat      — identity, axis rotations, normalization, orthogonality,
                       determinant, zero-quaternion error.
obb_corners_world   — shape, centroid, axis-aligned extent, rotated extent.
Trajectory          — construction, __len__, duration_us, positions_xy,
                       at_ns boundary clamping, at_ns exact match,
                       from_csv round-trip, mismatched-length error,
                       empty trajectory error.
rotate_cw90         — all four corners of an N×N image, inverse consistency.
project_pt          — behind-camera returns None, invalid cam returns None,
                       forward point projects to finite pixel, CW-90° applied.
load_trajectory     — backward-compat wrapper produces same result as Trajectory.
interp_pose         — backward-compat wrapper: before/after bounds, exact hit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spatialcortex.geometry import (
    BB3D_LINE_ORDERS,
    Trajectory,
    interp_pose,
    load_trajectory,
    obb_corners_world,
    project_pt,
    quat_to_rotmat,
    rotate_cw90,
)
from spatialcortex.types import Pose


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# quat_to_rotmat
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQuatToRotmat:

    def test_identity_quaternion(self):
        R = quat_to_rotmat(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90_degrees_around_z(self):
        """qw=cos(45°), qz=sin(45°)  →  CCW 90° around Z."""
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        R = quat_to_rotmat(c, 0.0, 0.0, s)
        expected = np.array([
            [ 0, -1,  0],
            [ 1,  0,  0],
            [ 0,  0,  1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_180_degrees_around_x(self):
        """qw=0, qx=1  →  180° rotation around X: Y and Z flip sign."""
        R = quat_to_rotmat(0.0, 1.0, 0.0, 0.0)
        expected = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_90_degrees_around_y(self):
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        R = quat_to_rotmat(c, 0.0, s, 0.0)
        expected = np.array([
            [ 0,  0,  1],
            [ 0,  1,  0],
            [-1,  0,  0],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_unnormalised_quaternion_gives_valid_rotation(self):
        """Scale factor should be divided out; result must still be valid."""
        R = quat_to_rotmat(2.0, 0.0, 0.0, 0.0)   # same as identity, scaled
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_orthogonality(self):
        """R @ R.T ≈ I for an arbitrary rotation."""
        R = quat_to_rotmat(0.5, 0.5, 0.5, 0.5)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_determinant_one(self):
        """det(R) must be +1 (proper rotation, not reflection)."""
        R = quat_to_rotmat(0.5, 0.5, 0.5, 0.5)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_output_shape(self):
        R = quat_to_rotmat(1.0, 0.0, 0.0, 0.0)
        assert R.shape == (3, 3)

    def test_output_dtype(self):
        R = quat_to_rotmat(1.0, 0.0, 0.0, 0.0)
        assert R.dtype == np.float64

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError, match="magnitude"):
            quat_to_rotmat(0.0, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize("axis,angle", [
        ((1, 0, 0), math.pi / 3),
        ((0, 1, 0), math.pi / 6),
        ((0, 0, 1), 2 * math.pi / 3),
        ((1, 1, 0), math.pi),   # axis not unit; will be normalised below
    ])
    def test_axis_angle_roundtrip(self, axis, angle):
        """Encode axis-angle as quaternion, decode via rotmat, check a point."""
        ax = np.array(axis, dtype=float)
        ax /= np.linalg.norm(ax)
        qx, qy, qz = ax * math.sin(angle / 2)
        qw = math.cos(angle / 2)
        R = quat_to_rotmat(qw, qx, qy, qz)
        # Rodrigues' formula for the same rotation
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R_ref = (np.eye(3) + math.sin(angle) * K
                 + (1 - math.cos(angle)) * (K @ K))
        np.testing.assert_allclose(R, R_ref, atol=1e-12)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# obb_corners_world
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestObbCornersWorld:

    def test_output_shape(self):
        c = obb_corners_world(0, 0, 0, 1, 0, 0, 0, 1, 1, 1)
        assert c.shape == (8, 3)

    def test_output_dtype(self):
        c = obb_corners_world(0, 0, 0, 1, 0, 0, 0, 1, 1, 1)
        assert c.dtype == np.float64

    def test_centroid_equals_input_centre(self):
        """The mean of all 8 corners must equal the input centre."""
        tx, ty, tz = 1.5, -2.3, 0.7
        c = obb_corners_world(tx, ty, tz, 1, 0, 0, 0, 2, 3, 4)
        np.testing.assert_allclose(c.mean(axis=0), [tx, ty, tz], atol=1e-12)

    def test_axis_aligned_extent_matches_scale(self):
        """For an identity-rotated box, corner spread = sx × sy × sz."""
        c = obb_corners_world(0, 0, 0, 1, 0, 0, 0, 4.0, 6.0, 2.0)
        np.testing.assert_allclose(c[:, 0].max() - c[:, 0].min(), 4.0, atol=1e-12)
        np.testing.assert_allclose(c[:, 1].max() - c[:, 1].min(), 6.0, atol=1e-12)
        np.testing.assert_allclose(c[:, 2].max() - c[:, 2].min(), 2.0, atol=1e-12)

    def test_corners_at_half_extents_from_centre(self):
        """Identity rotation: each corner must be ±half-extent from centre."""
        c = obb_corners_world(0, 0, 0, 1, 0, 0, 0, 2.0, 4.0, 6.0)
        assert set(np.unique(np.round(c[:, 0], 10))) == {-1.0, 1.0}
        assert set(np.unique(np.round(c[:, 1], 10))) == {-2.0, 2.0}
        assert set(np.unique(np.round(c[:, 2], 10))) == {-3.0, 3.0}

    def test_non_zero_centre_shifts_all_corners(self):
        centre = np.array([10.0, 20.0, 30.0])
        c = obb_corners_world(*centre, 1, 0, 0, 0, 1, 1, 1)
        np.testing.assert_allclose(c.mean(axis=0), centre, atol=1e-12)

    def test_90_degree_z_rotation_swaps_x_y(self):
        """Rotating 90° around Z should swap X and Y extents."""
        ang = math.pi / 4
        qw, qz = math.cos(ang), math.sin(ang)
        # 2×4 box; after 90° Z rotation x-span ≈ 4, y-span ≈ 2
        c = obb_corners_world(0, 0, 0, qw, 0, 0, qz, 2.0, 4.0, 1.0)
        np.testing.assert_allclose(c[:, 0].max() - c[:, 0].min(), 4.0, atol=1e-12)
        np.testing.assert_allclose(c[:, 1].max() - c[:, 1].min(), 2.0, atol=1e-12)

    def test_exactly_12_unique_edge_pairs(self):
        """BB3D_LINE_ORDERS must reference valid corner indices."""
        for i, j in BB3D_LINE_ORDERS:
            assert 0 <= i < 8
            assert 0 <= j < 8
            assert i != j
        assert len(BB3D_LINE_ORDERS) == 12


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trajectory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_trajectory(n: int = 5) -> Trajectory:
    """Helper: build a straight-line trajectory with identity rotations."""
    times_us = np.arange(n, dtype=np.int64) * 100_000
    Rs = [np.eye(3)] * n
    ts_pos = [np.array([float(i) * 0.5, 0.0, 0.0]) for i in range(n)]
    return Trajectory(times_us, Rs, ts_pos)


class TestTrajectory:

    def test_len(self):
        assert len(_make_trajectory(7)) == 7

    def test_duration_us(self):
        traj = _make_trajectory(5)
        assert traj.duration_us == 4 * 100_000

    def test_positions_xy_shape(self):
        traj = _make_trajectory(5)
        assert traj.positions_xy.shape == (5, 2)

    def test_positions_xy_values(self):
        traj = _make_trajectory(3)
        expected = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(traj.positions_xy, expected)

    def test_at_ns_returns_pose(self):
        traj = _make_trajectory(5)
        pose = traj.at_ns(0)
        assert isinstance(pose, Pose)

    def test_at_ns_exact_match(self):
        """Exact nanosecond timestamp for the 3rd pose (index 2)."""
        traj = _make_trajectory(5)
        # times_us[2] = 200_000 µs → 200_000_000 ns
        pose = traj.at_ns(200_000_000)
        np.testing.assert_allclose(pose.t, [1.0, 0.0, 0.0])

    def test_at_ns_before_start_clamps_to_first(self):
        traj = _make_trajectory(5)
        pose = traj.at_ns(-1_000_000_000)
        np.testing.assert_allclose(pose.t, [0.0, 0.0, 0.0])

    def test_at_ns_after_end_clamps_to_last(self):
        traj = _make_trajectory(5)
        pose = traj.at_ns(999_999_999_999)
        np.testing.assert_allclose(pose.t, [2.0, 0.0, 0.0])

    def test_at_ns_pose_has_valid_rotation(self):
        traj = _make_trajectory(5)
        pose = traj.at_ns(0)
        assert pose.is_valid_rotation()

    def test_repr_contains_n(self):
        traj = _make_trajectory(5)
        assert "n=5" in repr(traj)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            Trajectory(np.array([1, 2, 3]), [np.eye(3)], [np.zeros(3)])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Trajectory(np.array([], dtype=np.int64), [], [])

    def test_from_csv_roundtrip(self, trajectory_csv):
        traj = Trajectory.from_csv(trajectory_csv)
        assert len(traj) == 5
        np.testing.assert_allclose(traj._ts_pos[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(traj._ts_pos[4], [0.4, 0.0, 0.0])

    def test_from_csv_rotation_matrices_are_valid(self, trajectory_csv):
        traj = Trajectory.from_csv(trajectory_csv)
        for R in traj._Rs:
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_times_us_is_int64(self):
        traj = _make_trajectory(3)
        assert traj.times_us.dtype == np.int64


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rotate_cw90
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRotateCw90:
    N = 100

    def test_top_left_corner(self):
        # (0, 0) → after CW-90°: (N-1, 0)
        assert rotate_cw90(0, 0, self.N) == (self.N - 1, 0)

    def test_top_right_corner(self):
        # (N-1, 0) → (N-1, N-1)
        assert rotate_cw90(self.N - 1, 0, self.N) == (self.N - 1, self.N - 1)

    def test_bottom_left_corner(self):
        # (0, N-1) → (0, 0)
        assert rotate_cw90(0, self.N - 1, self.N) == (0, 0)

    def test_bottom_right_corner(self):
        # (N-1, N-1) → (0, N-1)
        assert rotate_cw90(self.N - 1, self.N - 1, self.N) == (0, self.N - 1)

    def test_centre_maps_to_centre(self):
        """For odd N the centre pixel is exactly (N//2, N//2) both before/after."""
        N = 101
        cx, cy = N // 2, N // 2
        u_r, v_r = rotate_cw90(cx, cy, N)
        # CW 90°: (cx, cy) → (N-1-cy, cx) = (50, 50) for N=101
        assert (u_r, v_r) == (N - 1 - cy, cx)

    def test_four_cw_rotations_return_to_origin(self):
        """Applying CW-90° four times must return to the original pixel."""
        N, u0, v0 = 200, 47, 130
        u, v = u0, v0
        for _ in range(4):
            u, v = rotate_cw90(u, v, N)
        assert (u, v) == (u0, v0)

    @pytest.mark.parametrize("u,v,N", [
        (0, 0, 10),
        (9, 9, 10),
        (5, 3, 10),
        (0, 9, 10),
        (9, 0, 10),
    ])
    def test_output_always_in_bounds(self, u, v, N):
        u_r, v_r = rotate_cw90(u, v, N)
        assert 0 <= u_r < N
        assert 0 <= v_r < N


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# project_pt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProjectPt:
    """project_pt tests use Pose.identity() for both world-device and device-camera
    transforms so the only variable is the point position."""

    def _call(self, pw, cam, N=200, R_wd=None, t_wd=None, R_dc=None, t_dc=None):
        R_wd  = R_wd  if R_wd  is not None else np.eye(3)
        t_wd  = t_wd  if t_wd  is not None else np.zeros(3)
        R_dc  = R_dc  if R_dc  is not None else np.eye(3)
        t_dc  = t_dc  if t_dc  is not None else np.zeros(3)
        return project_pt(np.array(pw), R_wd, t_wd, R_dc, t_dc, cam, N)

    def test_behind_camera_returns_none(self, mock_cam):
        """Point at z <= 0.05 in camera frame → None."""
        result = self._call([0.0, 0.0, 0.04], mock_cam)
        assert result is None

    def test_exactly_at_threshold_returns_none(self, mock_cam):
        result = self._call([0.0, 0.0, 0.05], mock_cam)
        assert result is None

    def test_cam_project_none_returns_none(self, mock_cam_invalid):
        result = self._call([0.0, 0.0, 1.0], mock_cam_invalid)
        assert result is None

    def test_forward_point_returns_tuple(self, mock_cam):
        result = self._call([0.0, 0.0, 1.0], mock_cam)
        assert result is not None
        assert len(result) == 2

    def test_forward_point_returns_floats(self, mock_cam):
        u, v = self._call([0.0, 0.0, 1.0], mock_cam)
        assert isinstance(u, float)
        assert isinstance(v, float)

    def test_on_axis_projects_to_image_centre(self, mock_cam):
        """A point directly on the camera +Z axis should project to image centre."""
        N = 200
        u, v = self._call([0.0, 0.0, 5.0], mock_cam, N=N)
        # MockCam: u = 0/5*50 + 100 = 100, v = 100 → after CW-90°: (N-1-v, u)
        # = (99, 100) for N=200  ← rotate_cw90 is applied
        assert u == pytest.approx(N - 1 - 100, abs=1e-6)
        assert v == pytest.approx(100.0, abs=1e-6)

    def test_translation_offsets_device_frame(self, mock_cam):
        """World-to-device translation moves the point in camera space."""
        N = 200
        t_wd = np.array([1.0, 0.0, 0.0])   # device is 1 m to the right
        result = self._call([0.0, 0.0, 2.0], mock_cam, N=N, t_wd=t_wd)
        # Without translation: p_dev = [0, 0, 2]; with t_wd=[1,0,0]: p_dev = [-1, 0, 2]
        result_no_t = self._call([0.0, 0.0, 2.0], mock_cam, N=N)
        assert result != result_no_t

    def test_cw90_transform_applied(self, mock_cam):
        """The CW-90° rotation must change coordinates relative to raw projection."""
        N = 200
        focal = mock_cam._focal
        pw = np.array([0.5, 0.0, 1.0])   # off-centre right
        raw_u = 0.5 / 1.0 * focal + N / 2   # 125.0
        raw_v = 0.0 / 1.0 * focal + N / 2   # 100.0
        expected_u = N - 1 - raw_v           # 99.0
        expected_v = raw_u                   # 125.0
        u, v = self._call(pw.tolist(), mock_cam, N=N)
        assert u == pytest.approx(expected_u, abs=1e-9)
        assert v == pytest.approx(expected_v, abs=1e-9)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Legacy backward-compat wrappers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLegacyAPI:

    def test_load_trajectory_returns_triple(self, trajectory_csv):
        result = load_trajectory(trajectory_csv)
        assert len(result) == 3
        times_us, Rs, ts_pos = result
        assert len(times_us) == 5
        assert len(Rs) == 5
        assert len(ts_pos) == 5

    def test_load_trajectory_matches_from_csv(self, trajectory_csv):
        times_us, Rs, ts_pos = load_trajectory(trajectory_csv)
        traj = Trajectory.from_csv(trajectory_csv)
        np.testing.assert_array_equal(times_us, traj.times_us)
        for R1, R2 in zip(Rs, traj._Rs):
            np.testing.assert_allclose(R1, R2)

    def test_interp_pose_before_start_clamps(self, trajectory_csv):
        times_us, Rs, ts_pos = load_trajectory(trajectory_csv)
        R, t = interp_pose(times_us, Rs, ts_pos, query_ns=-1)
        np.testing.assert_allclose(t, ts_pos[0])

    def test_interp_pose_after_end_clamps(self, trajectory_csv):
        times_us, Rs, ts_pos = load_trajectory(trajectory_csv)
        R, t = interp_pose(times_us, Rs, ts_pos, query_ns=10**18)
        np.testing.assert_allclose(t, ts_pos[-1])

    def test_interp_pose_exact_match(self, trajectory_csv):
        times_us, Rs, ts_pos = load_trajectory(trajectory_csv)
        # Index 2: times_us[2]=200_000 µs → 200_000_000 ns
        R, t = interp_pose(times_us, Rs, ts_pos, query_ns=200_000_000)
        np.testing.assert_allclose(t, ts_pos[2])
