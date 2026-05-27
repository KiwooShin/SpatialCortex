"""
Tests for spatialcortex.drawing

Coverage
--------
ClassColorRegistry — construction, __getitem__ (known/unknown/case),
                     __contains__, __len__, register custom color,
                     known_classes, repr, extra overrides constructor.
get_color          — delegates to module singleton; returns BGR tuple.
draw_obb           — returns bool; False when all behind camera;
                     True when a point projects; image modified in-place;
                     highlight fill does not crash; label rendering path.
draw_hud_arrow     — runs without error; HERE rendered when dist < 0.3;
                     arrow rendered when dist > 0.3; modifies image in-place.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialcortex.drawing import (
    ClassColorRegistry,
    draw_hud_arrow,
    draw_obb,
    get_color,
)
from spatialcortex.geometry import obb_corners_world


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _front_corners(center=(0.0, 0.0, 3.0), size=(1.0, 1.0, 1.0)) -> np.ndarray:
    """OBB centred 3 m in front of an identity camera — all corners visible."""
    return obb_corners_world(*center, 1.0, 0.0, 0.0, 0.0, *size)


def _behind_corners() -> np.ndarray:
    """OBB 3 m behind the camera — all corners invisible."""
    return obb_corners_world(0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def _draw_obb_kwargs(cam, N=200):
    """Minimal common kwargs for draw_obb with identity poses."""
    return dict(
        R_wd=np.eye(3),
        t_wd=np.zeros(3),
        R_dc=np.eye(3),
        t_dc=np.zeros(3),
        cam_calib=cam,
        color=(0, 255, 0),
        label="test",
        N=N,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ClassColorRegistry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClassColorRegistry:

    def test_known_class_returns_bgr_tuple(self):
        reg = ClassColorRegistry()
        color = reg["lamp"]
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_lamp_color_bgr_values(self):
        """lamp = RGB (1.00, 0.80, 0.25) → BGR (63, 204, 255)."""
        reg = ClassColorRegistry()
        b, g, r = reg["lamp"]
        assert b == int(0.25 * 255)
        assert g == int(0.80 * 255)
        assert r == int(1.00 * 255)

    def test_unknown_class_returns_default_gray(self):
        reg = ClassColorRegistry()
        color = reg["definitely_not_a_real_class"]
        # Default: RGB (0.70, 0.70, 0.70) → BGR (178, 178, 178)
        expected = int(0.70 * 255)
        assert all(c == expected for c in color)

    def test_case_insensitive_lookup(self):
        reg = ClassColorRegistry()
        assert reg["LAMP"] == reg["lamp"]
        assert reg["Sofa"] == reg["sofa"]
        assert reg["TABLE"] == reg["table"]

    def test_contains_known_class(self):
        reg = ClassColorRegistry()
        assert "chair" in reg
        assert "lamp"  in reg

    def test_contains_case_insensitive(self):
        reg = ClassColorRegistry()
        assert "CHAIR" in reg

    def test_not_contains_unknown(self):
        reg = ClassColorRegistry()
        assert "spaceship" not in reg

    def test_len_at_least_default_count(self):
        reg = ClassColorRegistry()
        assert len(reg) >= 30   # at least all SSI defaults

    def test_register_new_class(self):
        reg = ClassColorRegistry()
        reg.register("robot", r=0.2, g=0.8, b=0.5)
        assert "robot" in reg
        b, g, r = reg["robot"]
        assert b == int(0.5 * 255)
        assert g == int(0.8 * 255)
        assert r == int(0.2 * 255)

    def test_register_overrides_existing(self):
        reg = ClassColorRegistry()
        original = reg["lamp"]
        reg.register("lamp", r=0.0, g=0.0, b=0.0)
        assert reg["lamp"] == (0, 0, 0)
        assert reg["lamp"] != original

    def test_register_case_insensitive_storage(self):
        reg = ClassColorRegistry()
        reg.register("MyClass", r=1.0, g=0.0, b=0.0)
        assert "myclass" in reg
        assert reg["MYCLASS"] == reg["myclass"]

    def test_extra_overrides_in_constructor(self):
        reg = ClassColorRegistry(extra={"lamp": (0.0, 1.0, 0.0)})
        b, g, r = reg["lamp"]
        assert r == 0
        assert g == 255
        assert b == 0

    def test_extra_adds_new_class(self):
        reg = ClassColorRegistry(extra={"alien": (1.0, 0.0, 0.5)})
        assert "alien" in reg

    def test_known_classes_returns_sorted_list(self):
        reg = ClassColorRegistry()
        classes = reg.known_classes()
        assert classes == sorted(classes)
        assert "lamp" in classes

    def test_repr(self):
        reg = ClassColorRegistry()
        r = repr(reg)
        assert "ClassColorRegistry" in r
        assert str(len(reg)) in r

    def test_color_components_in_0_255_range(self):
        reg = ClassColorRegistry()
        for name in reg.known_classes():
            for component in reg[name]:
                assert 0 <= component <= 255, f"{name}: {reg[name]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# get_color (module-level convenience)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetColor:

    def test_returns_bgr_tuple(self):
        color = get_color("lamp")
        assert isinstance(color, tuple) and len(color) == 3

    def test_matches_registry_singleton(self):
        reg = ClassColorRegistry()
        assert get_color("sofa") == reg["sofa"]

    def test_unknown_returns_default(self):
        color = get_color("invisible_class")
        assert isinstance(color, tuple) and len(color) == 3

    @pytest.mark.parametrize("name", [
        "chair", "sofa", "table", "lamp", "bed", "mirror", "plant",
    ])
    def test_all_standard_classes_have_color(self, name):
        color = get_color(name)
        assert color != get_color("definitely_not_a_class"), (
            f"{name} should not return default color"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# draw_obb
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDrawObb:

    def test_returns_false_when_all_behind_camera(self, blank_bgr_image, mock_cam):
        corners = _behind_corners()
        result = draw_obb(blank_bgr_image, corners, **_draw_obb_kwargs(mock_cam))
        assert result is False

    def test_returns_false_with_invalid_camera(self, blank_bgr_image, mock_cam_invalid):
        corners = _front_corners()
        result = draw_obb(blank_bgr_image, corners, **_draw_obb_kwargs(mock_cam_invalid))
        assert result is False

    def test_returns_true_when_visible(self, blank_bgr_image, mock_cam):
        corners = _front_corners()
        result = draw_obb(blank_bgr_image, corners, **_draw_obb_kwargs(mock_cam))
        assert result is True

    def test_modifies_image_when_visible(self, mock_cam):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        corners = _front_corners()
        draw_obb(img, corners, **_draw_obb_kwargs(mock_cam))
        assert img.sum() > 0, "Image should have non-zero pixels after drawing"

    def test_does_not_modify_image_when_behind(self, mock_cam):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        corners = _behind_corners()
        draw_obb(img, corners, **_draw_obb_kwargs(mock_cam))
        assert img.sum() == 0, "Image should remain black when nothing is visible"

    def test_highlight_does_not_raise(self, blank_bgr_image, mock_cam):
        corners = _front_corners()
        draw_obb(
            blank_bgr_image, corners,
            **_draw_obb_kwargs(mock_cam),
            highlight=True,
        )

    def test_empty_label_does_not_raise(self, blank_bgr_image, mock_cam):
        corners = _front_corners()
        kw = _draw_obb_kwargs(mock_cam)
        kw["label"] = ""
        draw_obb(blank_bgr_image, corners, **kw)

    def test_custom_line_width_does_not_raise(self, blank_bgr_image, mock_cam):
        corners = _front_corners()
        draw_obb(blank_bgr_image, corners, **_draw_obb_kwargs(mock_cam), lw=5)

    def test_n_samples_one_does_not_raise(self, blank_bgr_image, mock_cam):
        """n_samples=1 means no edge segments can be drawn; must return False."""
        corners = _front_corners()
        result = draw_obb(
            blank_bgr_image, corners,
            **_draw_obb_kwargs(mock_cam),
            n_samples=1,
        )
        # linspace(0,1,1) → [0.0] → len(pts2d)-1 = 0 iterations
        assert result is False

    def test_image_not_modified_when_behind(self, mock_cam):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        corners = _behind_corners()
        original = img.copy()
        draw_obb(img, corners, **_draw_obb_kwargs(mock_cam))
        np.testing.assert_array_equal(img, original)

    def test_fill_alpha_zero_still_draws_edges(self, mock_cam):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        corners = _front_corners()
        result = draw_obb(
            img, corners, **_draw_obb_kwargs(mock_cam),
            highlight=True, fill_alpha=0.0,
        )
        assert result is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# draw_hud_arrow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDrawHudArrow:
    """draw_hud_arrow modifies the image in-place; we test that it runs
    without error and that the image is visibly modified."""

    def _base_kwargs(self):
        return dict(
            R_wd=np.eye(3),
            t_wd=np.zeros(3),
            R_dc=np.eye(3),
            t_dc=np.zeros(3),
        )

    def test_runs_without_error_far_target(self, blank_bgr_image):
        target = np.array([5.0, 0.0, 0.0])
        draw_hud_arrow(blank_bgr_image, target_pos=target, **self._base_kwargs())

    def test_runs_without_error_near_target(self, blank_bgr_image):
        """Distance < 0.3 → "HERE" text branch."""
        target = np.array([0.1, 0.0, 0.0])
        draw_hud_arrow(blank_bgr_image, target_pos=target, **self._base_kwargs())

    def test_modifies_image(self, blank_bgr_image):
        target = np.array([5.0, 0.0, 0.0])
        draw_hud_arrow(blank_bgr_image, target_pos=target, **self._base_kwargs())
        assert blank_bgr_image.sum() > 0

    def test_here_branch_modifies_image(self):
        """target at device position → dist=0 < 0.3 → HERE text drawn."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        draw_hud_arrow(
            img,
            R_wd=np.eye(3), t_wd=np.zeros(3),
            R_dc=np.eye(3), t_dc=np.zeros(3),
            target_pos=np.array([0.0, 0.0, 0.0]),
        )
        assert img.sum() > 0

    def test_returns_none(self, blank_bgr_image):
        target = np.array([3.0, 0.0, 0.0])
        result = draw_hud_arrow(blank_bgr_image, target_pos=target, **self._base_kwargs())
        assert result is None

    def test_various_target_directions_do_not_raise(self, blank_bgr_image):
        """Arrow direction rotates as target moves around the camera."""
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
            img = blank_bgr_image.copy()
            draw_hud_arrow(
                img,
                target_pos=np.array([float(dx) * 5, float(dy) * 5, 0.0]),
                **self._base_kwargs(),
            )

    def test_very_large_distance_does_not_raise(self, blank_bgr_image):
        draw_hud_arrow(
            blank_bgr_image,
            target_pos=np.array([1e6, 0.0, 0.0]),
            **self._base_kwargs(),
        )

    def test_target_directly_overhead_does_not_raise(self, blank_bgr_image):
        """delta_xy ≈ 0 tests the normalisation clamp."""
        draw_hud_arrow(
            blank_bgr_image,
            target_pos=np.array([0.0, 0.0, 100.0]),
            **self._base_kwargs(),
        )
