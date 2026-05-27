"""SpatialCortex — shared library for Aria/EFM3D spatial memory pipelines.

Public API
----------
Types (dataclasses)::

    from spatialcortex import Pose, OBB, Trajectory, ScenePaths, QueryResult

Geometry helpers::

    from spatialcortex.geometry import (
        quat_to_rotmat, obb_corners_world, project_pt,
        rotate_cw90, load_trajectory, interp_pose,
        BB3D_LINE_ORDERS,
    )

Drawing helpers::

    from spatialcortex.drawing import (
        ClassColorRegistry, get_color, draw_obb, draw_hud_arrow,
    )

Project config::

    from spatialcortex.config import BASE, RGB_SID, SCENES, CLIP_MODEL
"""

from .types import (
    CameraCalibration,
    OBB,
    Pose,
    QueryResult,
    ScenePaths,
)
from .geometry import Trajectory

__all__ = [
    # Types
    "CameraCalibration",
    "OBB",
    "Pose",
    "QueryResult",
    "ScenePaths",
    # Geometry
    "Trajectory",
]
