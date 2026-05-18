# SpatialCortex

**Spatial memory system for AR/VR — walk into a room, remember it forever, query it in natural language.**

> Built on 3D Gaussian Splatting · Grounded-SAM 2 · CLIP · Gemma 3 27B · Three.js · Rerun.io

---

## What is this?

SpatialCortex gives AR/VR devices (or robots) persistent spatial memory. On first entry, it maps an environment and detects every object in 3D. Later, you can ask *"Where did I leave the hammer?"* in natural language and get an AR navigation path back to it — even after leaving and returning.

**Four-stage pipeline:**

```
[Entry]  VRS recording  →  SLAM trajectory + 3D Gaussian Splatting
                        →  Grounded-SAM 2 object detection (2D → 3D)
                        →  Scene graph: 3D bounding boxes + CLIP embeddings → FAISS

[Query]  "Where is the chair?"
                        →  CLIP retrieval → Gemma 3 27B visual confirmation
                        →  Re-localization in stored map
                        →  Dijkstra navigation path → AR overlay
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Project Aria .vrs recording                             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   SLAM + 3D Gaussian      │  projectaria_tools (VIO poses)
          │   Splatting               │  gaussian-splatting (3DGS)
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   Open-Vocab 3D Detection │  Grounded-SAM 2 (2D masks)
          │                           │  Depth unprojection → 3D bbox
          └─────────────┬─────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   Spatial Memory DB       │  SQLite (objects + poses)
          │                           │  FAISS (CLIP ViT-L/14 vectors)
          └──────┬──────────┬─────────┘
                 │          │
    ┌────────────▼──┐  ┌────▼────────────────┐
    │  VLM Query    │  │  Navigation          │
    │  Gemma 3 27B  │  │  Dijkstra on         │
    │  (on-device)  │  │  waypoint graph      │
    └────────────┬──┘  └────┬────────────────┘
                 │          │
          ┌──────▼──────────▼──────────┐
          │   Visualization             │
          │   Three.js (web map)        │
          │   Rerun.io (real-time)      │
          └─────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Scene reconstruction | 3D Gaussian Splatting (Kerbl et al., ICCV 2023) |
| Object detection | Grounded-SAM 2 (Meta AI, 2024) |
| Semantic embeddings | CLIP ViT-L/14 + FAISS |
| Spatial VLM | Gemma 3 27B multimodal (on-device, DGX Spark) |
| Spatial database | SQLite + FAISS index |
| Visualization | Three.js (interactive web map) + Rerun.io (real-time dashboard) |
| Navigation | Dijkstra on SLAM waypoint graph |
| Input data | Project Aria Gen 2 `.vrs` recordings |

---

## Current Status

| Milestone | Status |
|---|---|
| SLAM trajectory + point cloud visualization (Three.js) | ✅ Done |
| Aria VRS → JSON converter (`vrs_to_json.py`) | ✅ Done |
| 3D Gaussian Splatting reconstruction | 🔲 In progress |
| Grounded-SAM 2 → 3D object detection | 🔲 In progress |
| CLIP + FAISS spatial memory DB | 🔲 Planned |
| Gemma 3 27B spatial query pipeline | 🔲 Planned |
| Re-localization + navigation | 🔲 Planned |
| Three.js interactive map upgrade | 🔲 Planned |
| Rerun.io real-time dashboard | 🔲 Planned |
| End-to-end demo video | 🔲 Planned |

---

## Quick Start

### Visualize the SLAM viewer (current)

```bash
git clone https://github.com/KiwooShin/SpatialCortex.git
cd SpatialCortex

# Generate data from a Project Aria .vrs file
/path/to/miniconda3/envs/aria/bin/python3 scripts/vrs_to_json.py \
  --vrs /path/to/recording.vrs \
  --output data/aria_vrs.json

# Serve locally
python3 -m http.server 8000
# Open http://localhost:8000
```

### Environment setup (DGX Spark)

```bash
conda create -n spatialcortex python=3.10 -y
conda activate spatialcortex
pip install projectaria-tools torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything-2
pip install open_clip_torch faiss-gpu rerun-sdk
```

---

## Viewer Controls

| Input | Action |
|---|---|
| Left drag | Orbit |
| Right drag | Pan |
| Scroll | Zoom |
| `Space` | Play / pause trajectory |
| `←` `→` | Step frames |
| `P` | Toggle point cloud |
| `T` | Toggle trajectory |
| `F` | Toggle camera frustum |
| `C` | Cycle color mode (height / confidence / uniform) |

---

## Project Structure

```
SpatialCortex/
├── index.html              # Three.js SLAM viewer
├── plan.md                 # 2-week build plan
├── scripts/
│   └── vrs_to_json.py      # Aria .vrs → viewer JSON (VIO trajectory + point cloud)
└── data/                   # gitignored — generated files go here
    ├── aria_vrs.json        # extracted from .vrs
    ├── scene_db.sqlite      # object detections + metadata
    ├── scene.faiss          # CLIP embedding index
    └── splat.ply            # 3DGS output
```

---

## References

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., ICCV 2023
- [Grounded-SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) — IDEA Research / Meta AI
- [Project Aria Tools](https://github.com/facebookresearch/projectaria_tools) — Meta Reality Labs
- [Gemma 3](https://ai.google.dev/gemma) — Google DeepMind
- [Rerun.io](https://rerun.io) — multimodal data visualization

---

## License

MIT
