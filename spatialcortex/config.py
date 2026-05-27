"""
Project-wide paths, scene registry, and model configuration.

All scripts import BASE, RGB_SID, SCENES and file-path constants from here
so there is exactly one place to update when data moves.
"""

from pathlib import Path

from projectaria_tools.core.stream_id import StreamId

# ── Project root ──────────────────────────────────────────────────────────────
# spatialcortex/ lives one level below the repo root.
BASE = Path(__file__).resolve().parent.parent

# ── Aria RGB stream ───────────────────────────────────────────────────────────
RGB_SID = StreamId(214, 1)

# ── AEO scene registry ────────────────────────────────────────────────────────
# Keys used consistently across all scripts:
#   vrs          – main VRS recording
#   traj         – closed-loop SLAM trajectory CSV
#   snippet_obbs – per-frame EFM3D detections (correct orientations, ~2 s apart)
#   scene_obbs   – fused scene-level OBBs (use only for top-down maps / DB build)
SCENES: dict[str, dict[str, Path]] = {
    "seq00": {
        "vrs":          BASE / "data/aeo/aeo_seq00_173376298563204/main.vrs",
        "traj":         BASE / "data/aeo/aeo_seq00_173376298563204/mps/mps/slam/closed_loop_trajectory.csv",
        "snippet_obbs": BASE / "output/efm3d_aeo_seq00/model_release/aeo_seq00_173376298563204/snippet_obbs.csv",
        "scene_obbs":   BASE / "output/efm3d_aeo_seq00/model_release/aeo_seq00_173376298563204/scene_obbs.csv",
    },
    "seq01": {
        "vrs":          BASE / "data/aeo/aeo_seq01_208838848508107/main.vrs",
        "traj":         BASE / "data/aeo/aeo_seq01_208838848508107/mps/slam/closed_loop_trajectory.csv",
        "snippet_obbs": BASE / "output/efm3d_aeo_seq01/model_release/aeo_seq01_208838848508107/snippet_obbs.csv",
        "scene_obbs":   BASE / "output/efm3d_aeo_seq01/model_release/aeo_seq01_208838848508107/scene_obbs.csv",
    },
    "seq02": {
        "vrs":          BASE / "data/aeo/aeo_seq02_181771578105956/main.vrs",
        "traj":         BASE / "data/aeo/aeo_seq02_181771578105956/mps/slam/closed_loop_trajectory.csv",
        "snippet_obbs": BASE / "output/efm3d_aeo_seq02/model_release/aeo_seq02_181771578105956/snippet_obbs.csv",
        "scene_obbs":   BASE / "output/efm3d_aeo_seq02/model_release/aeo_seq02_181771578105956/scene_obbs.csv",
    },
}

# ── CLIP model ────────────────────────────────────────────────────────────────
CLIP_MODEL      = "ViT-L-14-quickgelu"
CLIP_PRETRAINED = "openai"

# ── Default file paths ────────────────────────────────────────────────────────
DB_PATH        = BASE / "data" / "scene_db.sqlite"
SCENE_FAISS    = BASE / "data" / "scene.faiss"
KEYFRAME_FAISS = BASE / "data" / "keyframe_index.faiss"
KEYFRAME_CSV   = BASE / "data" / "keyframe_index.csv"
OUTPUT_DIR     = BASE / "output"
