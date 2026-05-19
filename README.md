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
| Rerun.io VRS viewer (`visualize_vrs.py`) | ✅ Done |
| Three.js interactive map — bounding boxes, nav path, query sidebar | ✅ Done |
| 3D Gaussian Splatting reconstruction | 🔲 Planned |
| Grounded-SAM 2 → 3D object detection | 🔲 Planned |
| CLIP + FAISS spatial memory DB | 🔲 Planned |
| Gemma 3 27B spatial query pipeline | 🔲 Planned |
| Re-localization + navigation | 🔲 Planned |
| End-to-end demo video | 🔲 Planned |

---

## How to Run

### 1. Clone & setup environment

```bash
git clone https://github.com/KiwooShin/SpatialCortex.git
cd SpatialCortex

# Create conda environment with projectaria-tools + rerun
conda create -n aria python=3.10 -y
conda activate aria
pip install projectaria-tools rerun-sdk
```

### 2. Extract SLAM data from a VRS recording

Converts VIO trajectory and synthesizes a point cloud from a raw Project Aria `.vrs` file into the JSON format consumed by the web viewer.

```bash
conda activate aria
python3 scripts/vrs_to_json.py \
  --vrs /path/to/recording.vrs \
  --output data/aria_vrs.json \
  --points 15000          # number of synthetic environment points
  --subsample 1           # take every Nth VIO pose (1 = all)
```

### 3. Extract RGB video for live camera feed sync

Exports all RGB frames as a compressed MP4. The web viewer syncs this video with the 3D trajectory playback.

```bash
conda activate aria
python3 scripts/extract_video.py \
  --vrs /path/to/recording.vrs \
  --output data/rgb_video.mp4\
  --width 640 \
  --fps 10
```

### 4. Launch the Three.js web viewer

Interactive 3D map with sidebar query panel, object bounding boxes, and navigation path.

```bash
python3 -m http.server 8000
# Open http://localhost:8000
```

### 5. Launch the Rerun.io real-time dashboard

Streams RGB camera, SLAM cameras, VIO trajectory, and device pose into a Rerun timeline. Saves a `.rrd` replay file for offline demo.

```bash
conda activate aria

# Live interactive viewer
python3 scripts/visualize_vrs.py \
  --vrs /path/to/recording.vrs

# Save .rrd for offline replay
python3 scripts/visualize_vrs.py \
  --vrs /path/to/recording.vrs \
  --rrd data/session.rrd \
  --downsample 4 \
  --jpeg-quality 75

# Replay saved session
rerun data/session.rrd
```

---

## Viewer Controls

### Three.js web viewer

| Input | Action |
|---|---|
| Left drag | Orbit |
| Right drag | Pan |
| Scroll | Zoom |
| `Space` | Play / pause trajectory |
| `←` `→` | Step frames |
| `P` | Toggle point cloud |
| `T` | Toggle trajectory |
| `B` | Toggle bounding boxes |
| `N` | Toggle navigation path |
| `F` | Toggle camera frustum |
| `C` | Cycle color mode (height / confidence / uniform) |

Click an object in the sidebar → camera focuses on it, draws navigation path from current position.

---

## Project Structure

```
SpatialCortex/
├── index.html                  # Three.js interactive viewer (video sync + trajectory)
├── plan.md                     # 2-week build plan
├── scripts/
│   ├── vrs_to_json.py          # Aria .vrs → viewer JSON (VIO trajectory + point cloud)
│   ├── visualize_vrs.py        # Rerun.io VRS dashboard (RGB + SLAM + trajectory)
│   ├── extract_keyframes.py    # Extract N evenly-spaced JPEG keyframes for filmstrip
│   └── extract_video.py        # Extract all RGB frames → MP4 for live video sync
└── data/                       # gitignored — generated files go here
    ├── aria_vrs.json            # extracted from .vrs (vrs_to_json.py output)
    ├── rgb_video.mp4            # RGB camera feed (extract_video.py output)
    ├── session.rrd              # Rerun recording (visualize_vrs.py output)
    ├── scene_db.sqlite          # object detections + metadata  (Phase 2)
    ├── scene.faiss              # CLIP embedding index           (Phase 2)
    └── splat.ply                # 3DGS reconstruction            (Phase 2)
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
