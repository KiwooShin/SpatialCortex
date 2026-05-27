"""
Core domain types for SpatialCortex.

Provides typed, immutable value objects for the key domain concepts:
  - ``Pose``        — rigid-body SE(3) transform with compose / inverse
  - ``OBB``         — oriented 3-D bounding box with geometry helpers
  - ``ScenePaths``  — typed filesystem paths for one AEO scene
  - ``QueryResult`` — structured output from a spatial memory query

Design notes
------------
- All spatial types are ``frozen=True`` dataclasses → hashable, safe as dict keys,
  no accidental mutation.
- Numpy arrays inside frozen dataclasses are copied in ``__post_init__`` so the
  immutability guarantee is meaningful.
- ``OBB.corners()`` and ``Pose.from_quaternion()`` use *lazy* imports to keep the
  import graph acyclic:
      types.py  ←imports—  geometry.py  ←imports—  drawing.py
- ``ScenePaths`` implements ``__getitem__`` so legacy code using ``paths["vrs"]``
  continues to work without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


# ── Type aliases (documentation-only; not enforced at runtime) ───────────────
Mat3 = np.ndarray   # shape (3, 3) float64
Vec3 = np.ndarray   # shape (3,)  float64


# ── Camera calibration protocol ───────────────────────────────────────────────

@runtime_checkable
class CameraCalibration(Protocol):
    """Structural interface for Aria / mock camera calibrations.

    Any object that provides these two methods satisfies the protocol,
    enabling unit tests to inject lightweight fakes without needing a
    real VRS recording.
    """

    def project(self, point_in_camera: Vec3) -> tuple[float, float] | None:
        """Project a 3-D camera-frame point → image pixel, or None if invisible."""
        ...

    def get_image_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` in pixels."""
        ...


# ── Pose ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Pose:
    """Rigid body pose (element of SE(3)).

    Represents transform T_{B←A}: maps a point expressed in frame A to frame B::

        p_B = R @ p_A + t

    Attributes
    ----------
    R : (3, 3) rotation matrix
    t : (3,)  translation vector
    """

    R: Mat3
    t: Vec3

    def __post_init__(self) -> None:
        # Ensure owned float64 copies even when callers pass lists or views.
        object.__setattr__(self, "R", np.asarray(self.R, dtype=np.float64).copy())
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64).copy())

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def identity(cls) -> "Pose":
        """Identity pose — zero rotation, zero translation."""
        return cls(R=np.eye(3), t=np.zeros(3))

    @classmethod
    def from_quaternion(
        cls,
        qw: float, qx: float, qy: float, qz: float,
        tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
    ) -> "Pose":
        """Construct from a (possibly unnormalised) quaternion + translation."""
        from .geometry import quat_to_rotmat  # lazy import — avoids circular dep
        return cls(R=quat_to_rotmat(qw, qx, qy, qz), t=np.array([tx, ty, tz]))

    # ── Operations ────────────────────────────────────────────────────────────

    def transform(self, point: Vec3) -> Vec3:
        """Apply the pose to a point: ``R @ point + t``."""
        return self.R @ np.asarray(point, dtype=np.float64) + self.t

    def inverse(self) -> "Pose":
        """Return the inverse pose T^{-1} = (R^T, −R^T @ t)."""
        Rt = self.R.T
        return Pose(R=Rt, t=-(Rt @ self.t))

    def compose(self, other: "Pose") -> "Pose":
        """Chain poses: T_self ∘ T_other (apply *other* first, then *self*)."""
        return Pose(R=self.R @ other.R, t=self.R @ other.t + self.t)

    # ── Validation ────────────────────────────────────────────────────────────

    def is_valid_rotation(self, tol: float = 1e-6) -> bool:
        """Return True if R is a valid rotation matrix (orthogonal, det ≈ +1)."""
        orth_err = float(np.linalg.norm(self.R @ self.R.T - np.eye(3)))
        det_err  = abs(float(np.linalg.det(self.R)) - 1.0)
        return orth_err < tol and det_err < tol

    def __repr__(self) -> str:
        tx, ty, tz = self.t
        return f"Pose(t=({tx:.3f}, {ty:.3f}, {tz:.3f}))"


# ── OBB ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OBB:
    """Oriented 3-D bounding box in the world frame.

    Attributes
    ----------
    cx, cy, cz      : box centre in world coordinates (metres)
    qw, qx, qy, qz  : orientation quaternion (world ← object frame)
    sx, sy, sz       : **full** side lengths (not half-extents), metres
    name             : EFM3D class label, e.g. ``"chair"``
    prob             : detection confidence in [0, 1]
    """

    cx: float; cy: float; cz: float          # noqa: E702
    qw: float; qx: float; qy: float; qz: float  # noqa: E702
    sx: float; sy: float; sz: float          # noqa: E702
    name: str   = ""
    prob: float = 1.0

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_row(cls, row) -> "OBB":
        """Construct from a pandas Series or dict row.

        Supports two column naming conventions automatically:

        - **snippet_obbs.csv style**: ``tx_world_object``, ``ty_world_object``, …
        - **scene_obbs / SQLite style**: ``tx``, ``ty``, ``tz``, ``qw``, …
        """
        if "tx_world_object" in row:
            return cls(
                cx=float(row["tx_world_object"]),
                cy=float(row["ty_world_object"]),
                cz=float(row["tz_world_object"]),
                qw=float(row["qw_world_object"]),
                qx=float(row["qx_world_object"]),
                qy=float(row["qy_world_object"]),
                qz=float(row["qz_world_object"]),
                sx=float(row["scale_x"]),
                sy=float(row["scale_y"]),
                sz=float(row["scale_z"]),
                name=str(row.get("name", "")),
                prob=float(row.get("prob", 1.0)),
            )
        # scene_obbs / SQLite columns: tx, ty, tz, qw, …
        return cls(
            cx=float(row["tx"]), cy=float(row["ty"]), cz=float(row["tz"]),
            qw=float(row["qw"]), qx=float(row["qx"]),
            qy=float(row["qy"]), qz=float(row["qz"]),
            sx=float(row["scale_x"]),
            sy=float(row["scale_y"]),
            sz=float(row["scale_z"]),
            name=str(row.get("name", "")),
            prob=float(row.get("prob", 1.0)),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def center(self) -> Vec3:
        """Box centre as a (3,) float64 array."""
        return np.array([self.cx, self.cy, self.cz], dtype=np.float64)

    @property
    def extents(self) -> Vec3:
        """Full side-lengths as a (3,) float64 array."""
        return np.array([self.sx, self.sy, self.sz], dtype=np.float64)

    @property
    def volume(self) -> float:
        """Geometric volume in m³."""
        return float(self.sx * self.sy * self.sz)

    # ── Geometry methods ──────────────────────────────────────────────────────

    def corners(self) -> np.ndarray:
        """Return (8, 3) corner positions in world frame."""
        from .geometry import obb_corners_world  # lazy import
        return obb_corners_world(
            self.cx, self.cy, self.cz,
            self.qw, self.qx, self.qy, self.qz,
            self.sx, self.sy, self.sz,
        )

    def distance_to(self, other: "OBB") -> float:
        """Euclidean distance between box centres (metres)."""
        return float(np.linalg.norm(self.center - other.center))

    def __repr__(self) -> str:
        return (
            f"OBB(name={self.name!r}, "
            f"center=({self.cx:.2f},{self.cy:.2f},{self.cz:.2f}), "
            f"size={self.sx:.2f}×{self.sy:.2f}×{self.sz:.2f}, "
            f"prob={self.prob:.3f})"
        )


# ── ScenePaths ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScenePaths:
    """Typed filesystem paths for one AEO scene.

    Implements ``__getitem__`` so legacy code that treats a ``ScenePaths``
    as a plain dict (``paths["vrs"]``) continues to work unchanged.
    """

    name:         str
    vrs:          Path
    traj:         Path
    snippet_obbs: Path
    scene_obbs:   Path

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ScenePaths":
        """Construct from the legacy ``dict[str, Path]`` representation."""
        return cls(
            name=name,
            vrs=Path(d["vrs"]),
            traj=Path(d["traj"]),
            snippet_obbs=Path(d["snippet_obbs"]),
            scene_obbs=Path(d["scene_obbs"]),
        )

    # ── Subscript compatibility ───────────────────────────────────────────────

    def __getitem__(self, key: str) -> Path:
        """``paths["vrs"]`` → ``paths.vrs`` (backward-compat with dict API)."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return error strings for every path that does not exist on disk.

        An empty list means all paths are present.
        """
        missing: list[str] = []
        for attr in ("vrs", "traj", "snippet_obbs", "scene_obbs"):
            p: Path = getattr(self, attr)
            if not p.exists():
                missing.append(f"{self.name}.{attr}: {p}")
        return missing

    def __repr__(self) -> str:
        return f"ScenePaths(name={self.name!r}, vrs={self.vrs.name!r})"


# ── QueryResult ───────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Structured output from a spatial memory query.

    Attributes
    ----------
    name        : matched object class label
    scene       : scene identifier (e.g. ``"seq01"``)
    position    : (3,) world XYZ in metres
    similarity  : CLIP cosine similarity in [0, 1]
    best_ts_ns  : nanosecond timestamp of the best-view frame
    nearby      : class names of objects within 2 m
    image_path  : path to the saved annotated frame (empty if not rendered)
    """

    name:       str
    scene:      str
    position:   Vec3
    similarity: float
    best_ts_ns: int
    nearby:     list[str] = field(default_factory=list)
    image_path: str = ""

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)

    @property
    def distance_from_origin(self) -> float:
        """Euclidean distance of the object from the world origin (metres)."""
        return float(np.linalg.norm(self.position))

    def __repr__(self) -> str:
        x, y, z = self.position
        return (
            f"QueryResult(name={self.name!r}, scene={self.scene!r}, "
            f"pos=({x:.2f},{y:.2f},{z:.2f}), sim={self.similarity:.3f})"
        )
