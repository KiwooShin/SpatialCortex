# SpatialCortex

**Spatial memory system for AR/VR — walk into a room, remember it forever, query it in natural language.**

> Built on 3D Gaussian Splatting · Grounded-SAM 2 · Depth Anything V2 · CLIP · Gemma 3 27B · Three.js · Rerun.io

---

## What is this?

SpatialCortex gives AR/VR devices (or robots) persistent spatial memory. On first entry, it maps an environment and detects every object in 3D. Later, you can ask *"Where did I leave the hammer?"* in natural language and get an AR navigation path back to it — even after leaving and returning.

**Four-stage pipeline:**

```
[Entry]  VRS recording  →  COLMAP SfM → SLAM trajectory + 3D Gaussian Splatting
                        →  Grounded-SAM 2: open-vocab 2D detection + SAM 2 masks
                        →  Depth Anything V2 (metric, COLMAP-calibrated) → dense depth
                        →  Depth unproject → 3D AABBs → visualized as projected cuboids
                        →  Scene graph: 3D bboxes + CLIP ViT-L/14 embeddings → FAISS

[Query]  "Where is the toaster?"
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
| 2D detection + segmentation | Grounded-SAM 2 = Grounding DINO + SAM 2 (Meta AI / IDEA Research, 2024) |
| Depth estimation | Depth Anything V2 metric-indoor (COLMAP scale-calibrated) |
| 3D lifting | Depth unprojection → axis-aligned 3D bboxes, visualized as projected cuboids |
| Semantic embeddings | CLIP ViT-L/14 + FAISS |
| Spatial VLM | Gemma 3 27B multimodal (on-device, DGX Spark) |
| Spatial database | SQLite + FAISS index |
| Visualization | Three.js (interactive web map) + Rerun.io (real-time dashboard) |
| Navigation | Dijkstra on SLAM waypoint graph |
| Input data | Project Aria Gen 2 `.vrs` recordings |
| *(Stretch)* Language 3D field | LangSplat / LangSplatV2 — CLIP features embedded in 3D Gaussians |

---

## Current Status

| Milestone | Status |
|---|---|
| Aria VRS → JSON converter (`vrs_to_json.py`) | ✅ Done |
| Rerun.io VRS viewer (`visualize_vrs.py`) | ✅ Done |
| Three.js interactive map — bounding boxes, nav path, query sidebar | ✅ Done |
| COLMAP SfM — undistorted frames + camera poses | ✅ Done |
| **EFM3D inference on AEO seq00/01/02** | ✅ Done |
| **Per-snippet OBB visualization (fisheye + top-down, `visualize_efm3d_obbs.py`)** | ✅ Done |
| **Scene-level OBB fusion (`fuse_scene_obbs.py`)** | ✅ Done |
| **Scene top-down map (`scene_topdown.py`)** | ✅ Done |
| 3D Gaussian Splatting reconstruction | 🔲 In progress |
| Grounded-SAM 2 on registered frames (`run_gsam2.py`) | 🔲 In progress |
| Depth Anything V2 + COLMAP scale calibration (`estimate_depth.py`) | 🔲 In progress |
| 3D bbox lifting + projected cuboid visualization (`lift_to_3d.py`) | 🔲 In progress |
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

### 4. Run COLMAP SfM (already done on this dataset)

Extracts undistorted RGB frames and runs COLMAP to produce camera poses and a sparse point cloud.

```bash
conda activate aria
python scripts/run_colmap.py \
  --vrs /path/to/recording.vrs \
  --output data/colmap \
  --every-nth 2
# Output: data/colmap/images/ + data/colmap/sparse/0/
```

### 5. Run the 3D detection pipeline

Single script: runs Grounded-SAM 2, Depth Anything V2, and 3D lifting in one pass.
Each frame is fully processed before moving to the next — no intermediate files required.

```bash
conda activate gsam2
python scripts/detect_objects_3d.py \
  --colmap  data/colmap \
  --output  data/detections_3d.json \
  --vis-dir data/visualizations
# Output:
#   data/detections_3d.json      ← [{object_id, label, bbox_3d, …}]
#   data/visualizations/*.jpg    ← frames annotated with projected 3D cuboids
```

Each visualization frame shows per-class coloured 3D cuboids projected onto the original image, with light transparent fills and text labels (`label confidence`).

To also save intermediate masks and depth maps for debugging:
```bash
python scripts/detect_objects_3d.py \
  --colmap      data/colmap \
  --output      data/detections_3d.json \
  --vis-dir     data/visualizations \
  --save-masks  data/debug/masks \
  --save-depth  data/debug/depth
```

### 8. Launch the Three.js web viewer

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

## EFM3D 3D Object Detection

SpatialCortex integrates [EFM3D](https://github.com/facebookresearch/efm3d) (Meta Reality Labs) as the primary 3D object detection backbone. It runs directly on Project Aria `.vrs` recordings using gravity-aligned voxel lifting of DINOv2 features.

### Quick start — EFM3D inference

```bash
conda activate efm3d
cd ~/efm3d

# Run inference on an AEO sequence (one snippet every 2 s = full coverage)
python infer.py \
  --input /path/to/aeo_seq/main.vrs \
  --output-dir ~/SpatialCortex/output/efm3d_seq \
  --snip_stride 2.0 \
  --obb_only

# Fuse per-snippet OBBs into one consistent scene map (no pytorch3d required)
cd ~/SpatialCortex
conda run -n efm3d python scripts/fuse_scene_obbs.py \
  --obbs  output/efm3d_seq/.../snippet_obbs.csv \
  --prob-thresh 0.20 --dist-thresh 0.80 --min-obs 2

# Render top-down scene map
conda run -n efm3d python scripts/scene_topdown.py \
  --obbs  output/efm3d_seq/.../scene_obbs.csv \
  --traj  data/aeo/.../mps/slam/closed_loop_trajectory.csv \
  --output output/scene_topdown/seq_topdown.jpg \
  --title "My Scene"

# Per-frame fisheye + top-down visualization
conda run -n efm3d python scripts/visualize_efm3d_obbs.py \
  --obbs  output/efm3d_seq/.../snippet_obbs.csv \
  --vrs   data/aeo/.../main.vrs \
  --traj  data/aeo/.../mps/slam/closed_loop_trajectory.csv \
  --output-dir output/efm3d_viz/
```

**Key implementation notes:**
- Aria RGB sensor image is 90° CCW from upright — the visualizer applies a CW 90° rotation to both the image array and all projected pixel coordinates.
- `scale_x/y/z` in `snippet_obbs.csv` are **full** dimensions; half-extents = scale/2.
- EFM3D's built-in `track_obbs()` requires pytorch3d (unavailable on aarch64/CUDA 13). `fuse_scene_obbs.py` is a drop-in replacement: greedy proximity clustering + confidence-weighted position/log-scale averaging + Markley quaternion mean + accumulated evidence confidence.

---

## Project Structure

```
SpatialCortex/
├── index.html                         # Three.js interactive viewer
├── plan.md                            # Build plan
├── research.md                        # Related research reference
├── scripts/
│   ├── vrs_to_json.py                 # Aria .vrs → viewer JSON
│   ├── visualize_vrs.py               # Rerun.io VRS dashboard
│   ├── extract_keyframes.py           # Extract N evenly-spaced keyframes
│   ├── extract_video.py               # RGB frames → MP4
│   ├── run_colmap.py                  # VRS → undistorted frames + COLMAP SfM
│   ├── visualize_colmap_points.py     # Overlay COLMAP sparse points on frames
│   ├── detect_objects_3d.py           # Full pipeline: SAM 2 + depth + 3D cuboid viz
│   ├── visualize_efm3d_obbs.py        # EFM3D: fisheye + top-down per-frame viz
│   ├── fuse_scene_obbs.py             # EFM3D: fuse snippet OBBs → scene_obbs.csv
│   └── scene_topdown.py               # EFM3D: render single scene top-down map
└── data/                              # gitignored — generated files go here
    ├── colmap/                         # COLMAP output (images + sparse/0/)
    ├── gaussian_output/                # 3DGS training output
    ├── detections/                     # run_gsam2.py output (JSON + mask PNGs)
    ├── depth/                          # estimate_depth.py output (*_depth.npy)
    ├── detections_3d.json              # lift_to_3d.py — 3D bbox list
    ├── visualizations/                 # lift_to_3d.py — frames with projected cuboids
    ├── scene_db.sqlite                 # object detections + metadata  (Phase 2)
    ├── scene.faiss                     # CLIP embedding index           (Phase 2)
    └── splat.ply                       # 3DGS reconstruction            (Phase 2)
```

---

## References

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., ICCV 2023
- [EFM3D / EVL](https://github.com/facebookresearch/efm3d) — Meta Reality Labs, 2024; egocentric voxel lifting for 3D OBB detection
- [DUSt3R](https://arxiv.org/abs/2312.14132) — CVPR 2024; feed-forward multi-view 3D reconstruction without calibration
- [MASt3R](https://arxiv.org/abs/2406.09756) — NAVER Labs, 2024; DUSt3R + dense matching head
- [OpenMask3D](https://openmask3d.github.io/) — NeurIPS 2023; open-vocabulary 3D instance segmentation
- [ConceptFusion](https://concept-fusion.github.io/) — RSS 2023; open-set multimodal 3D mapping via CLIP
- [OpenScene](https://pengsongyou.github.io/openscene) — CVPR 2023; CLIP feature distillation into 3D point clouds
- [LangSplat](https://langsplat.github.io/) — CVPR 2024; language-embedded 3D Gaussians (stretch goal)
- [Grounded-SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) — IDEA Research / Meta AI, 2024
- [Depth Anything V2](https://depth-anything-v2.github.io/) — metric monocular depth estimation
- [Project Aria Tools](https://github.com/facebookresearch/projectaria_tools) — Meta Reality Labs
- [Gemma 3](https://ai.google.dev/gemma) — Google DeepMind
- [Rerun.io](https://rerun.io) — multimodal data visualization

See [research.md](research.md) for detailed summaries of all related works.

---

## License

MIT
