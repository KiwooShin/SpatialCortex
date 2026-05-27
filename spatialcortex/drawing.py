"""
Shared OpenCV drawing utilities for Aria/EFM3D visualization pipelines.

Key exports
-----------
``ClassColorRegistry``
    Manages class-name → BGR-color mappings.  Case-insensitive lookup,
    configurable fallback, ``register()`` for custom entries.

``get_color(name)``
    Module-level convenience wrapper around the default registry singleton.

``draw_obb(...)``
    Project and draw an oriented 3-D bounding box onto a fisheye image.
    Returns ``True`` if at least one edge segment was drawn.

``draw_hud_arrow(...)``
    Render a compass HUD in the bottom-right corner pointing toward a target.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import BB3D_LINE_ORDERS, project_pt


# ── Class color registry ──────────────────────────────────────────────────────

class ClassColorRegistry:
    """Manages class-name → BGR-255 draw-color mappings.

    - Case-insensitive key lookup.
    - Falls back to a configurable default color for unknown classes.
    - Supports runtime addition of new classes via ``register()``.
    - Seeded with the canonical EFM3D ``SSI_SEM_COLORS`` palette.

    Examples
    --------
    >>> reg = ClassColorRegistry()
    >>> reg["lamp"]
    (63, 204, 255)
    >>> "sofa" in reg
    True
    >>> reg.register("robot", r=0.2, g=0.8, b=0.5)
    """

    # EFM3D SSI_SEM_COLORS palette (RGB 0–1)
    _SSI_DEFAULTS: dict[str, tuple[float, float, float]] = {
        "chair":          (0.20, 0.60, 1.00),
        "sofa":           (0.10, 0.50, 0.10),
        "table":          (1.00, 1.00, 0.00),
        "shelf":          (0.50, 0.00, 0.50),
        "lamp":           (1.00, 0.80, 0.25),
        "bed":            (0.90, 0.40, 0.60),
        "monitor":        (0.00, 0.80, 0.80),
        "cabinet":        (0.80, 0.60, 0.20),
        "tv":             (0.30, 0.90, 0.70),
        "window":         (0.70, 0.90, 1.00),
        "door":           (0.80, 0.70, 0.60),
        "picture_frame":  (1.00, 0.60, 0.20),
        "dresser":        (0.85, 0.55, 0.15),
        "mirror":         (0.60, 0.70, 1.00),
        "plant":          (0.20, 0.80, 0.20),
        "pillow":         (0.80, 0.60, 0.80),
        "floor_mat":      (0.60, 0.40, 0.20),
        "cart":           (0.60, 0.80, 0.40),
        "container":      (0.80, 0.50, 0.20),
        "ladder":         (0.50, 0.80, 0.30),
        "microwave":      (1.00, 0.40, 0.40),
        "refrigerator":   (0.40, 0.40, 1.00),
        "oven":           (0.70, 0.60, 0.30),
        "whiteboard":     (0.90, 0.90, 0.70),
        "curtain":        (0.70, 0.50, 1.00),
        "trash_can":      (0.50, 0.50, 0.50),
        "book":           (0.90, 0.70, 0.40),
        "bottle":         (0.40, 0.80, 0.60),
        "flower_pot":     (0.30, 0.70, 0.30),
        "exercise_weight":(0.60, 0.60, 0.90),
    }
    _DEFAULT_RGB: tuple[float, float, float] = (0.70, 0.70, 0.70)

    def __init__(
        self,
        extra: dict[str, tuple[float, float, float]] | None = None,
    ) -> None:
        """Initialise with EFM3D defaults, optionally extended by *extra*.

        Parameters
        ----------
        extra : optional ``{name: (r, g, b)}`` dict of RGB 0–1 overrides.
                Keys in *extra* take precedence over the built-in defaults.
        """
        self._table: dict[str, tuple[int, int, int]] = {
            name: self._rgb_to_bgr(*rgb)
            for name, rgb in self._SSI_DEFAULTS.items()
        }
        self._default_bgr: tuple[int, int, int] = self._rgb_to_bgr(*self._DEFAULT_RGB)
        if extra:
            for name, rgb in extra.items():
                self._table[name.lower()] = self._rgb_to_bgr(*rgb)

    # ── Core interface ────────────────────────────────────────────────────────

    @staticmethod
    def _rgb_to_bgr(r: float, g: float, b: float) -> tuple[int, int, int]:
        return (int(b * 255), int(g * 255), int(r * 255))

    def __getitem__(self, name: str) -> tuple[int, int, int]:
        """Return BGR color; falls back to default for unknown classes."""
        return self._table.get(name.lower(), self._default_bgr)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._table

    def __len__(self) -> int:
        return len(self._table)

    def __repr__(self) -> str:
        return f"ClassColorRegistry({len(self._table)} classes)"

    # ── Mutation ──────────────────────────────────────────────────────────────

    def register(self, name: str, r: float, g: float, b: float) -> None:
        """Add or overwrite a class color (RGB 0–1 components).

        Parameters
        ----------
        name : class label (stored case-insensitively)
        r, g, b : red, green, blue in [0.0, 1.0]
        """
        self._table[name.lower()] = self._rgb_to_bgr(r, g, b)

    # ── Introspection ─────────────────────────────────────────────────────────

    def known_classes(self) -> list[str]:
        """Return sorted list of registered class names."""
        return sorted(self._table)


# Module-level singleton — used by the backward-compatible ``get_color()``
_REGISTRY = ClassColorRegistry()


def get_color(name: str) -> tuple[int, int, int]:
    """Return the BGR draw color for an EFM3D object class (case-insensitive)."""
    return _REGISTRY[name]


# ── Oriented bounding box ─────────────────────────────────────────────────────

def draw_obb(
    img: np.ndarray,
    corners_world: np.ndarray,
    R_wd: np.ndarray,
    t_wd: np.ndarray,
    R_dc: np.ndarray,
    t_dc: np.ndarray,
    cam_calib,
    color: tuple[int, int, int],
    label: str,
    N: int,
    *,
    lw: int = 2,
    n_samples: int = 12,
    highlight: bool = False,
    fill_alpha: float = 0.28,
) -> bool:
    """Project and draw an oriented 3-D bounding box onto a fisheye image.

    Parameters
    ----------
    img           : BGR uint8 ndarray (modified in-place)
    corners_world : (8, 3) OBB corner positions in world frame
    R_wd, t_wd   : device pose in world frame (``R_world_device``, translation)
    R_dc, t_dc   : device-to-camera extrinsics
    cam_calib     : Aria CameraCalibration with ``.project()``
    color         : BGR draw color ``(B, G, R)``
    label         : text label placed at the projected box centroid
    N             : image side length in pixels (for CW-90° coordinate mapping)
    lw            : line width in pixels
    n_samples     : sample points per edge for fisheye curve correction
    highlight     : if True, flood-fill the projected convex hull
    fill_alpha    : translucency weight for the highlight fill (0 = transparent)

    Returns
    -------
    bool
        ``True`` if at least one edge segment was successfully drawn,
        ``False`` if the box is entirely behind the camera or outside the image.
    """
    H, W = img.shape[:2]

    corners_px: list[tuple[float, float] | None] = [
        project_pt(c, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
        for c in corners_world
    ]

    # Optional translucent fill for the highlighted target box
    if highlight and all(p is not None for p in corners_px):
        pts = np.array(corners_px, dtype=np.int32)
        if (
            pts[:, 0].min() >= 0 and pts[:, 0].max() < W
            and pts[:, 1].min() >= 0 and pts[:, 1].max() < H
        ):
            ov = img.copy()
            cv2.fillConvexPoly(ov, cv2.convexHull(pts), color)
            cv2.addWeighted(ov, fill_alpha, img, 1.0 - fill_alpha, 0, img)

    # Draw edges with fisheye-corrected sampling (each edge → n_samples segments)
    any_drawn = False
    for i, j in BB3D_LINE_ORDERS:
        ts_edge = np.linspace(0, 1, n_samples)
        pts3d = [
            corners_world[i] + t * (corners_world[j] - corners_world[i])
            for t in ts_edge
        ]
        pts2d = [
            project_pt(p, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
            for p in pts3d
        ]
        for k in range(len(pts2d) - 1):
            p0, p1 = pts2d[k], pts2d[k + 1]
            if p0 is None or p1 is None:
                continue
            x0, y0 = int(round(p0[0])), int(round(p0[1]))
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            # Small slack so edges entering/leaving the frame are drawn fully
            in_slack = lambda x, y: -60 <= x < W + 60 and -60 <= y < H + 60
            if not (in_slack(x0, y0) or in_slack(x1, y1)):
                continue
            cv2.line(img, (x0, y0), (x1, y1), color, lw, cv2.LINE_AA)
            any_drawn = True

    # Text label at the centroid of all visible projected corners
    if any_drawn and label:
        valid = [p for p in corners_px if p is not None]
        if valid:
            cx = int(np.mean([p[0] for p in valid]))
            cy = int(np.mean([p[1] for p in valid]))
            cx = max(2, min(cx, W - 120))
            cy = max(16, min(cy, H - 4))
            sc = 0.60 if highlight else 0.45
            th = 2 if highlight else 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, tht), _ = cv2.getTextSize(label, font, sc, th)
            cv2.rectangle(img, (cx - 3, cy - tht - 4),
                          (cx + tw + 3, cy + 3), (10, 10, 10), -1)
            cv2.putText(img, label, (cx, cy), font, sc, color, th, cv2.LINE_AA)

    return any_drawn


# ── HUD compass arrow ─────────────────────────────────────────────────────────

def draw_hud_arrow(
    img: np.ndarray,
    R_wd: np.ndarray,
    t_wd: np.ndarray,
    R_dc: np.ndarray,
    t_dc: np.ndarray,
    target_pos: np.ndarray,
) -> None:
    """Render a compass HUD in the bottom-right corner pointing toward a target.

    When the device is within 0.3 m of the target, the HUD displays ``"HERE"``
    instead of an arrow.

    Parameters
    ----------
    img        : BGR uint8 ndarray (modified in-place)
    R_wd, t_wd : device pose in world from ``Trajectory.at_ns()``
    R_dc, t_dc : device-to-camera extrinsics
    target_pos : (3,) world position of the navigation target
    """
    H, W = img.shape[:2]
    cx, cy = W - 70, H - 70
    r = 48

    dist_m = float(np.linalg.norm(np.asarray(target_pos) - t_wd))

    # Semi-transparent dark circle background
    ov = img.copy()
    cv2.circle(ov, (cx, cy), r + 4, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
    cv2.circle(img, (cx, cy), r + 4, (160, 160, 160), 1, cv2.LINE_AA)

    if dist_m < 0.3:
        cv2.putText(
            img, "HERE", (cx - 22, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA,
        )
    else:
        # Direction from device to target projected onto the world XY plane
        delta_xy = np.array(target_pos[:2]) - t_wd[:2]
        delta_xy /= max(float(np.linalg.norm(delta_xy)), 1e-6)

        # Camera optical-axis forward projected onto world XY
        fwd = R_wd @ (R_dc @ np.array([0.0, 0.0, 1.0]))
        fwd_xy = fwd[:2]
        fwd_xy /= max(float(np.linalg.norm(fwd_xy)), 1e-4)

        # Signed angle (positive = target is clockwise from camera forward)
        angle = -np.arctan2(
            fwd_xy[0] * delta_xy[1] - fwd_xy[1] * delta_xy[0],
            fwd_xy[0] * delta_xy[0] + fwd_xy[1] * delta_xy[1],
        )

        tip  = (int(cx + (r - 8) * np.sin(angle)),
                int(cy - (r - 8) * np.cos(angle)))
        tail = (int(cx - (r - 20) * np.sin(angle)),
                int(cy + (r - 20) * np.cos(angle)))
        cv2.arrowedLine(img, tail, tip, (0, 200, 255), 3,
                        cv2.LINE_AA, tipLength=0.32)

        # Cardinal ticks at N / E / S / W
        for a in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            cv2.line(
                img,
                (int(cx + r * np.sin(a)),       int(cy - r * np.cos(a))),
                (int(cx + (r - 7) * np.sin(a)), int(cy - (r - 7) * np.cos(a))),
                (120, 120, 120), 1, cv2.LINE_AA,
            )

    # Distance readout below the compass circle
    dist_str = f"{dist_m:.1f} m"
    (tw, _), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(
        img, dist_str, (cx - tw // 2, cy + r + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA,
    )
