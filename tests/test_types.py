"""
Tests for spatialcortex.types

Coverage
--------
Pose            — identity, from_quaternion, transform, inverse, compose,
                  is_valid_rotation, frozen immutability, repr.
OBB             — from_row (both column conventions), center, extents,
                  volume, corners shape/centroid, distance_to, repr.
ScenePaths      — from_dict, subscript access, validate (missing / all-present),
                  frozen, repr.
QueryResult     — construction, distance_from_origin, repr, position coercion.
CameraCalibration Protocol — conformance check with MockCameraCalibration.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from spatialcortex.types import (
    CameraCalibration,
    OBB,
    Pose,
    QueryResult,
    ScenePaths,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pose
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPose:

    def test_identity_r(self, identity_pose):
        np.testing.assert_allclose(identity_pose.R, np.eye(3))

    def test_identity_t(self, identity_pose):
        np.testing.assert_allclose(identity_pose.t, np.zeros(3))

    def test_identity_is_valid_rotation(self, identity_pose):
        assert identity_pose.is_valid_rotation()

    def test_from_quaternion_identity(self):
        pose = Pose.from_quaternion(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(pose.R, np.eye(3), atol=1e-12)

    def test_from_quaternion_with_translation(self):
        pose = Pose.from_quaternion(1.0, 0.0, 0.0, 0.0, tx=3.0, ty=4.0, tz=5.0)
        np.testing.assert_allclose(pose.t, [3.0, 4.0, 5.0])

    def test_from_quaternion_90z_rotation(self):
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        pose = Pose.from_quaternion(c, 0.0, 0.0, s)
        assert pose.is_valid_rotation()

    def test_transform_identity_is_noop(self, identity_pose):
        p = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(identity_pose.transform(p), p)

    def test_transform_applies_rotation_and_translation(self):
        # 90° around Z: x→y, y→-x; then translate by [1, 0, 0]
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        pose = Pose.from_quaternion(c, 0.0, 0.0, s, tx=1.0)
        p = np.array([1.0, 0.0, 0.0])
        result = pose.transform(p)
        # After 90° CCW Z rotation: [1,0,0] → [0,1,0]; + [1,0,0] = [1,1,0]
        np.testing.assert_allclose(result, [1.0, 1.0, 0.0], atol=1e-12)

    def test_inverse_of_identity_is_identity(self, identity_pose):
        inv = identity_pose.inverse()
        np.testing.assert_allclose(inv.R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(inv.t, np.zeros(3), atol=1e-12)

    def test_inverse_undoes_transform(self):
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        pose = Pose.from_quaternion(c, 0.0, 0.0, s, tx=5.0, ty=-3.0, tz=1.0)
        p = np.array([2.0, 1.0, -1.0])
        p_transformed = pose.transform(p)
        p_recovered   = pose.inverse().transform(p_transformed)
        np.testing.assert_allclose(p_recovered, p, atol=1e-12)

    def test_compose_identity_is_noop(self, identity_pose):
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        pose = Pose.from_quaternion(c, 0.0, 0.0, s, tx=1.0, ty=2.0)
        composed = pose.compose(identity_pose)
        np.testing.assert_allclose(composed.R, pose.R, atol=1e-12)
        np.testing.assert_allclose(composed.t, pose.t, atol=1e-12)

    def test_compose_then_inverse_is_identity(self):
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        pose = Pose.from_quaternion(c, 0.0, 0.0, s, tx=1.0, ty=2.0)
        result = pose.compose(pose.inverse())
        np.testing.assert_allclose(result.R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(result.t, np.zeros(3), atol=1e-12)

    def test_arrays_are_float64(self):
        pose = Pose(R=np.eye(3, dtype=np.float32), t=np.zeros(3, dtype=np.float32))
        assert pose.R.dtype == np.float64
        assert pose.t.dtype == np.float64

    def test_frozen_r_attribute_cannot_be_reassigned(self, identity_pose):
        """frozen=True prevents attribute *reassignment* (not in-place array mutation)."""
        with pytest.raises((TypeError, AttributeError)):
            identity_pose.R = np.eye(3)  # type: ignore[misc]

    def test_frozen_t_attribute_cannot_be_reassigned(self, identity_pose):
        with pytest.raises((TypeError, AttributeError)):
            identity_pose.t = np.zeros(3)  # type: ignore[misc]

    def test_repr_contains_translation(self):
        pose = Pose(R=np.eye(3), t=np.array([1.5, -2.0, 3.0]))
        r = repr(pose)
        assert "1.500" in r
        assert "-2.000" in r

    def test_is_valid_rotation_rejects_bad_matrix(self):
        bad_R = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        pose  = Pose.__new__(Pose)
        object.__setattr__(pose, "R", bad_R)
        object.__setattr__(pose, "t", np.zeros(3))
        assert not pose.is_valid_rotation()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SNIPPET_ROW = {
    "tx_world_object": 1.0, "ty_world_object": 2.0, "tz_world_object": 3.0,
    "qw_world_object": 1.0, "qx_world_object": 0.0,
    "qy_world_object": 0.0, "qz_world_object": 0.0,
    "scale_x": 0.5, "scale_y": 0.6, "scale_z": 0.7,
    "name": "chair", "prob": 0.95,
}

_SCENE_ROW = {
    "tx": 1.0, "ty": 2.0, "tz": 3.0,
    "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
    "scale_x": 0.5, "scale_y": 0.6, "scale_z": 0.7,
    "name": "sofa", "prob": 0.80,
}


class TestOBB:

    def test_from_row_snippet_style_position(self):
        obb = OBB.from_row(_SNIPPET_ROW)
        assert obb.cx == pytest.approx(1.0)
        assert obb.cy == pytest.approx(2.0)
        assert obb.cz == pytest.approx(3.0)

    def test_from_row_snippet_style_orientation(self):
        obb = OBB.from_row(_SNIPPET_ROW)
        assert obb.qw == pytest.approx(1.0)
        assert obb.qx == pytest.approx(0.0)

    def test_from_row_snippet_style_scale(self):
        obb = OBB.from_row(_SNIPPET_ROW)
        assert obb.sx == pytest.approx(0.5)
        assert obb.sy == pytest.approx(0.6)
        assert obb.sz == pytest.approx(0.7)

    def test_from_row_snippet_style_name_prob(self):
        obb = OBB.from_row(_SNIPPET_ROW)
        assert obb.name == "chair"
        assert obb.prob == pytest.approx(0.95)

    def test_from_row_scene_style(self):
        obb = OBB.from_row(_SCENE_ROW)
        assert obb.name == "sofa"
        assert obb.cx == pytest.approx(1.0)
        assert obb.sz == pytest.approx(0.7)

    def test_center_shape_and_values(self, unit_obb):
        c = unit_obb.center
        assert c.shape == (3,)
        np.testing.assert_allclose(c, [0.0, 0.0, 0.0])

    def test_extents_shape_and_values(self, unit_obb):
        e = unit_obb.extents
        assert e.shape == (3,)
        np.testing.assert_allclose(e, [1.0, 1.0, 1.0])

    def test_volume_unit_cube(self, unit_obb):
        assert unit_obb.volume == pytest.approx(1.0)

    def test_volume_rectangular_box(self):
        obb = OBB(0, 0, 0, 1, 0, 0, 0, 2.0, 3.0, 4.0)
        assert obb.volume == pytest.approx(24.0)

    def test_corners_shape(self, unit_obb):
        assert unit_obb.corners().shape == (8, 3)

    def test_corners_centroid_equals_center(self):
        obb = OBB(1.0, 2.0, 3.0, 1, 0, 0, 0, 2.0, 4.0, 6.0)
        np.testing.assert_allclose(obb.corners().mean(axis=0), [1.0, 2.0, 3.0])

    def test_corners_span_equals_extents(self):
        obb = OBB(0, 0, 0, 1, 0, 0, 0, 2.0, 4.0, 6.0)
        c = obb.corners()
        np.testing.assert_allclose(c[:, 0].max() - c[:, 0].min(), 2.0)
        np.testing.assert_allclose(c[:, 1].max() - c[:, 1].min(), 4.0)
        np.testing.assert_allclose(c[:, 2].max() - c[:, 2].min(), 6.0)

    def test_distance_to_self_is_zero(self, unit_obb):
        assert unit_obb.distance_to(unit_obb) == pytest.approx(0.0)

    def test_distance_to_offset_obb(self, unit_obb):
        other = OBB(3.0, 4.0, 0.0, 1, 0, 0, 0, 1, 1, 1)
        assert unit_obb.distance_to(other) == pytest.approx(5.0)

    def test_frozen_cannot_set_attribute(self, unit_obb):
        with pytest.raises((TypeError, AttributeError)):
            unit_obb.cx = 99.0  # type: ignore[misc]

    def test_repr_contains_name(self, unit_obb):
        assert "'test'" in repr(unit_obb)

    def test_repr_contains_size(self, unit_obb):
        assert "1.00×1.00×1.00" in repr(unit_obb)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScenePaths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestScenePaths:

    def test_from_dict_constructs_paths(self, tmp_path):
        d = {
            "vrs":          str(tmp_path / "main.vrs"),
            "traj":         str(tmp_path / "traj.csv"),
            "snippet_obbs": str(tmp_path / "snippet.csv"),
            "scene_obbs":   str(tmp_path / "scene.csv"),
        }
        sp = ScenePaths.from_dict("seq_test", d)
        assert sp.name == "seq_test"
        assert isinstance(sp.vrs, Path)
        assert sp.vrs.name == "main.vrs"

    def test_subscript_access_vrs(self, scene_paths_invalid):
        assert scene_paths_invalid["vrs"] == scene_paths_invalid.vrs

    def test_subscript_access_traj(self, scene_paths_invalid):
        assert scene_paths_invalid["traj"] == scene_paths_invalid.traj

    def test_subscript_access_snippet_obbs(self, scene_paths_invalid):
        assert scene_paths_invalid["snippet_obbs"] == scene_paths_invalid.snippet_obbs

    def test_subscript_access_unknown_key_raises(self, scene_paths_invalid):
        with pytest.raises(KeyError):
            _ = scene_paths_invalid["nonexistent"]

    def test_validate_all_missing(self, scene_paths_invalid):
        errors = scene_paths_invalid.validate()
        assert len(errors) == 4
        assert all("fake" in e for e in errors)

    def test_validate_all_present(self, scene_paths_valid):
        errors = scene_paths_valid.validate()
        assert errors == []

    def test_validate_partial_missing(self, tmp_path):
        vrs = tmp_path / "main.vrs"; vrs.touch()
        sp = ScenePaths(
            name="partial",
            vrs=vrs,
            traj=tmp_path / "missing_traj.csv",
            snippet_obbs=tmp_path / "missing_snip.csv",
            scene_obbs=tmp_path / "missing_scene.csv",
        )
        errors = sp.validate()
        assert len(errors) == 3
        assert not any("vrs" in e for e in errors)

    def test_frozen_cannot_mutate(self, scene_paths_invalid):
        with pytest.raises((TypeError, AttributeError)):
            scene_paths_invalid.name = "new_name"  # type: ignore[misc]

    def test_repr_contains_name(self, scene_paths_invalid):
        assert "'fake'" in repr(scene_paths_invalid)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QueryResult
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQueryResult:

    def test_construction(self):
        qr = QueryResult(
            name="lamp", scene="seq01",
            position=[1.0, 0.0, 0.0],
            similarity=0.85, best_ts_ns=12345,
        )
        assert qr.name == "lamp"
        assert qr.scene == "seq01"
        assert qr.similarity == pytest.approx(0.85)

    def test_position_converted_to_float64_array(self):
        qr = QueryResult("a", "s", [1, 2, 3], 0.5, 0)
        assert isinstance(qr.position, np.ndarray)
        assert qr.position.dtype == np.float64

    def test_distance_from_origin_3_4_0(self):
        qr = QueryResult("a", "s", [3.0, 4.0, 0.0], 0.5, 0)
        assert qr.distance_from_origin == pytest.approx(5.0)

    def test_distance_from_origin_zero(self):
        qr = QueryResult("a", "s", [0.0, 0.0, 0.0], 0.5, 0)
        assert qr.distance_from_origin == pytest.approx(0.0)

    def test_default_nearby_is_empty_list(self):
        qr = QueryResult("a", "s", [0, 0, 0], 0.5, 0)
        assert qr.nearby == []

    def test_default_image_path_is_empty(self):
        qr = QueryResult("a", "s", [0, 0, 0], 0.5, 0)
        assert qr.image_path == ""

    def test_nearby_list_mutability(self):
        qr = QueryResult("a", "s", [0, 0, 0], 0.5, 0, nearby=["chair"])
        qr.nearby.append("table")
        assert "table" in qr.nearby

    def test_repr_contains_name_and_scene(self):
        qr = QueryResult("lamp", "seq01", [1, 2, 3], 0.9, 0)
        r = repr(qr)
        assert "'lamp'" in r
        assert "'seq01'" in r

    def test_repr_contains_similarity(self):
        qr = QueryResult("lamp", "seq01", [1, 2, 3], 0.987, 0)
        assert "0.987" in repr(qr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CameraCalibration Protocol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCameraCalibrationProtocol:

    def test_mock_satisfies_protocol(self, mock_cam):
        """MockCameraCalibration from conftest must satisfy the Protocol."""
        assert isinstance(mock_cam, CameraCalibration)

    def test_invalid_mock_satisfies_protocol(self, mock_cam_invalid):
        assert isinstance(mock_cam_invalid, CameraCalibration)

    def test_object_without_project_does_not_satisfy(self):
        class NoProject:
            def get_image_size(self): return (100, 100)
        assert not isinstance(NoProject(), CameraCalibration)

    def test_object_without_get_image_size_does_not_satisfy(self):
        class NoImageSize:
            def project(self, pc): return (0.0, 0.0)
        assert not isinstance(NoImageSize(), CameraCalibration)
