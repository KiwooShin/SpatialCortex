"""
Shared OpenCV drawing utilities for Aria/EFM3D visualization pipelines.

All draw_* functions modify the given BGR uint8 ndarray in-place.
"""

import cv2
import numpy as np

from .geometry import BB3D_LINE_ORDERS, project_pt

# ── Class colors (EFM3D SSI_SEM_COLORS palette) ───────────────────────────────
_SSI_RGB: dict[str, tuple[float, float, float]] = {
    "chair":         (0.20, 0.60, 1.00),
    "sofa":          (0.10, 0.50, 0.10),
    "table":         (1.00, 1.00, 0.00),
    "shelf":         (0.50, 0.00, 0.50),
    "lamp":          (1.00, 0.80, 0.25),
    "bed":           (0.90, 0.40, 0.60),
    "monitor":       (0.00, 0.80, 0.80),
    "cabinet":       (0.80, 0.60, 0.20),
    "tv":            (0.30, 0.90, 0.70),
    "window":        (0.70, 0.90, 1.00),
    "door":          (0.80, 0.70, 0.60),
    "picture_frame": (1.00, 0.60, 0.20),
    "dresser":       (0.85, 0.55, 0.15),
    "mirror":        (0.60, 0.70, 1.00),
    "plant":         (0.20, 0.80, 0.20),
    "pillow":        (0.80, 0.60, 0.80),
    "floor_mat":     (0.60, 0.40, 0.20),
    "cart":          (0.60, 0.80, 0.40),
    "container":     (0.80, 0.50, 0.20),
    "ladder":        (0.50, 0.80, 0.30),
    "microwave":     (1.00, 0.40, 0.40),
    "refrigerator":  (0.40, 0.40, 1.00),
    "oven":          (0.70, 0.60, 0.30),
    "whiteboard":    (0.90, 0.90, 0.70),
    "curtain":       (0.70, 0.50, 1.00),
    "trash_can":     (0.50, 0.50, 0.50),
    "book":          (0.90, 0.70, 0.40),
    "bottle":        (0.40, 0.80, 0.60),
    "flower_pot":    (0.30, 0.70, 0.30),
    "exercise_weight":(0.60, 0.60, 0.90),
}
_DEFAULT_RGB = (0.70, 0.70, 0.70)


def _bgr255(r: float, g: float, b: float) -> tuple[int, int, int]:
    return (int(b * 255), int(g * 255), int(r * 255))


def get_color(name: str) -> tuple[int, int, int]:
    """BGR color for an EFM3D object class name."""
    return _bgr255(*_SSI_RGB.get(name.lower(), _DEFAULT_RGB))


# ── Oriented bounding box ─────────────────────────────────────────────────────

def draw_obb(img: np.ndarray,
             corners_world: np.ndarray,
             R_wd: np.ndarray, t_wd: np.ndarray,
             R_dc: np.ndarray, t_dc: np.ndarray,
             cam_calib,
             color: tuple[int, int, int],
             label: str,
             N: int,
             lw: int = 2,
             n_samples: int = 12,
             highlight: bool = False,
             fill_alpha: float = 0.28) -> None:
    """Project and draw an oriented 3D bounding box onto a fisheye image.

    Parameters
    ----------
    img           : BGR image (modified in-place)
    corners_world : (8, 3) OBB corners in world frame
    R_wd, t_wd   : world-to-device pose
    R_dc, t_dc   : device-to-camera transform
    cam_calib     : Aria CameraCalibration
    color         : BGR draw color
    label         : text label drawn at the box centroid
    N             : image side length (for CW-90° pixel mapping)
    lw            : line width
    n_samples     : points sampled along each edge (fisheye correction)
    highlight     : if True, fill the convex hull and use bolder text
    fill_alpha    : blending weight for the highlight fill
    """
    H, W = img.shape[:2]

    # Project all 8 corners
    corners_px = [
        project_pt(c, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
        for c in corners_world
    ]

    # Translucent fill when highlighting
    if highlight and all(p is not None for p in corners_px):
        pts = np.array(corners_px, dtype=np.int32)
        if (pts[:, 0].min() >= 0 and pts[:, 0].max() < W and
                pts[:, 1].min() >= 0 and pts[:, 1].max() < H):
            ov = img.copy()
            cv2.fillConvexPoly(ov, cv2.convexHull(pts), color)
            cv2.addWeighted(ov, fill_alpha, img, 1.0 - fill_alpha, 0, img)

    # Draw edges with fisheye-corrected sampling
    any_drawn = False
    for i, j in BB3D_LINE_ORDERS:
        ts_edge = np.linspace(0, 1, n_samples)
        pts3d = [corners_world[i] + t * (corners_world[j] - corners_world[i])
                 for t in ts_edge]
        pts2d = [project_pt(p, R_wd, t_wd, R_dc, t_dc, cam_calib, N)
                 for p in pts3d]
        for k in range(len(pts2d) - 1):
            p0, p1 = pts2d[k], pts2d[k + 1]
            if p0 is None or p1 is None:
                continue
            x0, y0 = int(round(p0[0])), int(round(p0[1]))
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            in_slack = lambda x, y: -60 <= x < W + 60 and -60 <= y < H + 60
            if not (in_slack(x0, y0) or in_slack(x1, y1)):
                continue
            cv2.line(img, (x0, y0), (x1, y1), color, lw, cv2.LINE_AA)
            any_drawn = True

    # Label at centroid of visible corners
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
            cv2.rectangle(img, (cx - 3, cy - tht - 4), (cx + tw + 3, cy + 3),
                          (10, 10, 10), -1)
            cv2.putText(img, label, (cx, cy), font, sc, color, th, cv2.LINE_AA)


# ── HUD compass arrow ─────────────────────────────────────────────────────────

def draw_hud_arrow(img: np.ndarray,
                   R_wd: np.ndarray, t_wd: np.ndarray,
                   R_dc: np.ndarray, t_dc: np.ndarray,
                   target_pos: np.ndarray) -> None:
    """Compass HUD in the bottom-right corner, always pointing toward target.

    Parameters
    ----------
    img        : BGR image (modified in-place)
    R_wd, t_wd: camera pose in world (from interp_pose)
    R_dc, t_dc: device-to-camera extrinsics
    target_pos : (3,) target world position
    """
    H, W = img.shape[:2]
    cx, cy = W - 70, H - 70
    r = 48

    dist_m = float(np.linalg.norm(np.array(target_pos) - t_wd))

    # Semi-transparent dark circle background
    ov = img.copy()
    cv2.circle(ov, (cx, cy), r + 4, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
    cv2.circle(img, (cx, cy), r + 4, (160, 160, 160), 1, cv2.LINE_AA)

    if dist_m < 0.3:
        cv2.putText(img, "HERE", (cx - 22, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA)
    else:
        # Direction from camera to target (world XY plane)
        delta_xy = np.array(target_pos[:2]) - t_wd[:2]
        delta_xy /= max(np.linalg.norm(delta_xy), 1e-6)

        # Camera optical-axis forward projected onto world XY
        fwd = R_wd @ (R_dc @ np.array([0.0, 0.0, 1.0]))
        fwd_xy = fwd[:2]
        fwd_xy /= max(np.linalg.norm(fwd_xy), 1e-4)

        # Signed angle: positive = target is clockwise from camera forward
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

        # Cardinal ticks
        for a in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            cv2.line(img,
                     (int(cx + r * np.sin(a)),       int(cy - r * np.cos(a))),
                     (int(cx + (r - 7) * np.sin(a)), int(cy - (r - 7) * np.cos(a))),
                     (120, 120, 120), 1, cv2.LINE_AA)

    # Distance text below compass
    dist_str = f"{dist_m:.1f} m"
    (tw, _), _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(img, dist_str, (cx - tw // 2, cy + r + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
