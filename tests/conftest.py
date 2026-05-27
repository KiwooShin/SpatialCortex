"""
Shared pytest fixtures for the SpatialCortex test suite.

Conventions
-----------
- All fixtures that produce numpy arrays use ``pytest.approx`` at assertion sites.
- ``MockCameraCalibration`` satisfies the ``CameraCalibration`` protocol and can
  be parameterised for valid / always-None / behind-camera scenarios.
- ``trajectory_csv`` writes a minimal 5-pose CSV to a tmp_path so ``Trajectory.from_csv``
  can be tested without touching real data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from spatialcortex.types import OBB, Pose, ScenePaths, QueryResult


# ── Camera calibration mock ───────────────────────────────────────────────────

class MockCameraCalibration:
    """Minimal pinhole-model mock satisfying the ``CameraCalibration`` protocol.

    Parameters
    ----------
    N           : image side length in pixels
    always_valid: if False, ``project()`` always returns None
    focal       : focal length used for the perspective projection
    """

    def __init__(
        self,
        N: int = 200,
        always_valid: bool = True,
        focal: float = 50.0,
    ) -> None:
        self.N = N
        self._valid = always_valid
        self._focal = focal

    def project(self, pc: np.ndarray) -> tuple[float, float] | None:
        if not self._valid or pc[2] <= 0:
            return None
        u = pc[0] / pc[2] * self._focal + self.N / 2
        v = pc[1] / pc[2] * self._focal + self.N / 2
        return (float(u), float(v))

    def get_image_size(self) -> tuple[int, int]:
        return (self.N, self.N)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_cam() -> MockCameraCalibration:
    """Standard 200×200 mock camera that always produces valid projections."""
    return MockCameraCalibration(N=200, always_valid=True)


@pytest.fixture
def mock_cam_invalid() -> MockCameraCalibration:
    """Mock camera whose ``project()`` always returns None."""
    return MockCameraCalibration(N=200, always_valid=False)


@pytest.fixture
def identity_pose() -> Pose:
    """Identity Pose (R=I, t=0)."""
    return Pose.identity()


@pytest.fixture
def unit_obb() -> OBB:
    """Axis-aligned 1×1×1 m box centred at the origin, class 'test'."""
    return OBB(
        cx=0.0, cy=0.0, cz=0.0,
        qw=1.0, qx=0.0, qy=0.0, qz=0.0,
        sx=1.0, sy=1.0, sz=1.0,
        name="test",
        prob=0.9,
    )


@pytest.fixture
def trajectory_csv(tmp_path) -> str:
    """Write a 5-pose trajectory CSV to *tmp_path* and return its path string."""
    rows = []
    for i in range(5):
        rows.append({
            "tracking_timestamp_us": i * 100_000,          # 0 … 400 µs
            "tx_world_device": float(i) * 0.1,
            "ty_world_device": 0.0,
            "tz_world_device": 0.0,
            "qw_world_device": 1.0,
            "qx_world_device": 0.0,
            "qy_world_device": 0.0,
            "qz_world_device": 0.0,
        })
    path = tmp_path / "trajectory.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def blank_bgr_image() -> np.ndarray:
    """200×200 black BGR image for drawing tests."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def scene_paths_invalid(tmp_path) -> ScenePaths:
    """ScenePaths whose file paths all point to non-existent files."""
    return ScenePaths(
        name="fake",
        vrs=tmp_path / "main.vrs",
        traj=tmp_path / "traj.csv",
        snippet_obbs=tmp_path / "snippet.csv",
        scene_obbs=tmp_path / "scene.csv",
    )


@pytest.fixture
def scene_paths_valid(tmp_path) -> ScenePaths:
    """ScenePaths whose file paths all exist (empty touch files)."""
    vrs          = tmp_path / "main.vrs";          vrs.touch()
    traj         = tmp_path / "traj.csv";          traj.touch()
    snippet_obbs = tmp_path / "snippet.csv";       snippet_obbs.touch()
    scene_obbs   = tmp_path / "scene.csv";         scene_obbs.touch()
    return ScenePaths(
        name="real",
        vrs=vrs, traj=traj,
        snippet_obbs=snippet_obbs, scene_obbs=scene_obbs,
    )
