"""
Project-wide paths, scene registry, and model configuration.

All scripts import constants from here so there is exactly one place to update
when data or model paths change.

``SCENES``
    ``dict[str, ScenePaths]`` — the main scene registry.
    ``ScenePaths`` supports subscript access (``paths["vrs"]``) so all existing
    scripts that treat it as a ``dict[str, Path]`` continue to work unchanged.
"""

from pathlib import Path

from projectaria_tools.core.stream_id import StreamId

from .types import ScenePaths


# ── Project root ──────────────────────────────────────────────────────────────
# spatialcortex/ is one level below the repo root.
BASE: Path = Path(__file__).resolve().parent.parent

# ── Aria RGB stream ───────────────────────────────────────────────────────────
RGB_SID = StreamId(214, 1)

# ── AEO scene registry ────────────────────────────────────────────────────────
SCENES: dict[str, ScenePaths] = {
    "seq00": ScenePaths(
        name="seq00",
        vrs=BASE / "data/aeo/aeo_seq00_173376298563204/main.vrs",
        traj=BASE / "data/aeo/aeo_seq00_173376298563204/mps/mps/slam/closed_loop_trajectory.csv",
        snippet_obbs=BASE / "output/efm3d_aeo_seq00/model_release/aeo_seq00_173376298563204/snippet_obbs.csv",
        scene_obbs=BASE / "output/efm3d_aeo_seq00/model_release/aeo_seq00_173376298563204/scene_obbs.csv",
    ),
    "seq01": ScenePaths(
        name="seq01",
        vrs=BASE / "data/aeo/aeo_seq01_208838848508107/main.vrs",
        traj=BASE / "data/aeo/aeo_seq01_208838848508107/mps/slam/closed_loop_trajectory.csv",
        snippet_obbs=BASE / "output/efm3d_aeo_seq01/model_release/aeo_seq01_208838848508107/snippet_obbs.csv",
        scene_obbs=BASE / "output/efm3d_aeo_seq01/model_release/aeo_seq01_208838848508107/scene_obbs.csv",
    ),
    "seq02": ScenePaths(
        name="seq02",
        vrs=BASE / "data/aeo/aeo_seq02_181771578105956/main.vrs",
        traj=BASE / "data/aeo/aeo_seq02_181771578105956/mps/slam/closed_loop_trajectory.csv",
        snippet_obbs=BASE / "output/efm3d_aeo_seq02/model_release/aeo_seq02_181771578105956/snippet_obbs.csv",
        scene_obbs=BASE / "output/efm3d_aeo_seq02/model_release/aeo_seq02_181771578105956/scene_obbs.csv",
    ),
    "seq07": ScenePaths(
        name="seq07",
        vrs=BASE / "data/aeo/aeo_seq07_622483472741639/main.vrs",
        traj=BASE / "data/aeo/aeo_seq07_622483472741639/mps/slam/closed_loop_trajectory.csv",
        snippet_obbs=BASE / "output/efm3d_aeo_seq07/model_release/aeo_seq07_622483472741639/snippet_obbs.csv",
        scene_obbs=BASE / "output/efm3d_aeo_seq07/model_release/aeo_seq07_622483472741639/scene_obbs.csv",
    ),
}

# ── CLIP model ────────────────────────────────────────────────────────────────
CLIP_MODEL:      str = "ViT-L-14-quickgelu"
CLIP_PRETRAINED: str = "openai"

# ── Default artifact paths ────────────────────────────────────────────────────
DB_PATH:        Path = BASE / "data" / "scene_db.sqlite"
SCENE_FAISS:    Path = BASE / "data" / "scene.faiss"
KEYFRAME_FAISS: Path = BASE / "data" / "keyframe_index.faiss"
KEYFRAME_CSV:   Path = BASE / "data" / "keyframe_index.csv"
OUTPUT_DIR:     Path = BASE / "output"
