# SpatialCortex — 2-Week Build Plan

**Goal**: End-to-end spatial memory system for AR/VR — a recruiter-ready demo that shows perception, language understanding, and navigation in one coherent pipeline.

**Hardware**: DGX Spark (128 GB unified memory)
**VLM**: Gemma 3 27B (multimodal, on-device)
**Data**: Project Aria `aria_gen2_sample_data_1.vrs` + optional iPhone LiDAR scan

---

## SLAM Point Cloud Strategy

### Phase 1 — COLMAP (local, no cloud dependency)
Run COLMAP Structure-from-Motion on extracted Aria RGB frames with known intrinsics from device calibration.
Produces sparse point cloud + refined camera poses in the exact format 3DGS expects.

```
VRS → extract frames (undistorted, pinhole) → COLMAP feature extraction
   → sequential matcher → mapper → sparse/0/{cameras,images,points3D}.txt
   → input for gaussian-splatting
```

**Why COLMAP first:** no cloud account needed, output directly feeds 3DGS, well-tested.

### Phase 2 — ORB-SLAM3 (future, loop-closed trajectory)

Replace COLMAP with ORB-SLAM3 for loop-closed trajectory + denser point tracking.
ORB-SLAM3 supports fisheye cameras natively (KannalaBrandt8) and fuses IMU for
loop-closed trajectories comparable to MPS quality — fully local, no cloud needed.

#### What needs to be done

**Step 1 — Build ORB-SLAM3 from source (~1 hour, mostly compile time)**
```bash
brew install cmake eigen opencv pangolin   # macOS deps
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3
cd ORB_SLAM3 && chmod +x build.sh && ./build.sh
# g2o, DBoW2, Sophus are bundled — no separate install
```

**Step 2 — Export VRS → EuRoC format (`scripts/export_euroc.py`, ~2 hours)**

ORB-SLAM3 expects this directory layout:
```
euroc_export/
├── cam0/data/     ← slam-front-left frames, filename = timestamp_ns.png
├── cam1/data/     ← slam-front-right frames (stereo mode)
└── imu0/data.csv  ← timestamp, wx, wy, wz, ax, ay, az
```
Use Aria SLAM cameras (grayscale 512×512, ~20Hz) not RGB — higher frequency,
better stereo overlap, designed for SLAM. Time-sync to IMU from VRS.

**Step 3 — Write Aria Gen2 camera YAML config (`configs/aria_gen2_slam.yaml`, ~1 hour)**

ORB-SLAM3 uses KannalaBrandt8 (KB8) fisheye model. Aria uses FISHEYE624 —
the first 4 distortion coefficients map approximately to KB8's k1,k2,k3,k4.
Read exact values from VRS calibration via projectaria_tools.

```yaml
Camera.type: "KannalaBrandt8"
Camera.fx: 241.0      # from Aria SLAM camera calibration
Camera.fy: 241.0
Camera.cx: 256.0
Camera.cy: 256.0
Camera.k1: ...        # FISHEYE624[0..3] ≈ KB8 coefficients
Camera.k2: ...
Camera.k3: ...
Camera.k4: ...
IMU.NoiseGyro: 0.002
IMU.NoiseAcc: 0.02
# T_cam_imu: extrinsics from device calibration
```

**Step 4 — Choose run mode and test (~30 min)**

| Mode | Cameras | Loop closure | Notes |
|---|---|---|---|
| Monocular-Inertial | slam-front-left + IMU | Yes | Easiest to start |
| **Stereo-Inertial** | slam-front-left + right + IMU | Yes | **Best — matches MPS quality** |
| Monocular | slam-front-left only | Yes | No IMU, slower convergence |

Target: **Stereo-Inertial** — same setup MPS uses internally.

#### Migration path from COLMAP

| | COLMAP (Phase 1) | ORB-SLAM3 (Phase 2) |
|---|---|---|
| Input cameras | RGB (10fps, fisheye→pinhole) | SLAM cams (20Hz, fisheye native) |
| IMU | No | Yes — fused |
| Trajectory | SfM keyframes only | Continuous 1kHz, loop-closed |
| Points | Sparse SfM | Semi-dense ORB map |
| Camera model | Pinhole (undistorted) | KannalaBrandt8 |
| Output | cameras/images/points3D.txt | TUM trajectory + map points |
| 3DGS compatibility | Direct | Needs format converter |

Output format changes but the downstream 3DGS pipeline stays the same —
swap COLMAP poses for ORB-SLAM3 poses, keep everything else unchanged.

---

## Tech Stack

| Layer | Technology | Why it's impressive |
|---|---|---|
| Scene reconstruction | **3D Gaussian Splatting** (3DGS) | State-of-the-art novel view synthesis; visually stunning |
| 2D segmentation | **Grounded-SAM 2** (Grounding DINO + SAM 2) | Open-vocabulary, zero-shot object masking; still SOTA for 2D-first detection as of 2025 |
| 3D object lifting | **Depth unprojection + mask fusion** | Converts 2D masks → 3D bounding volumes in SLAM frame |
| Semantic indexing | **CLIP ViT-L/14 embeddings + FAISS** | Sub-millisecond vector search over object crops |
| *(Stretch)* Language 3D field | **LangSplat / LangSplatV2** | Embeds CLIP features directly into 3D Gaussians — replaces CLIP+FAISS with native language-queryable 3D space; 199× faster than LERF |
| Spatial VLM | **Gemma 3 27B** (vision-language, on DGX Spark) | On-device, no API, full control; latest Google model |
| Spatial DB | **SQLite + FAISS index** | Lightweight, portable, no infra overhead |
| Visualization | **Three.js** (web) + **Rerun.io** (real-time) | Interactive 3D map for demo; Rerun used by FAIR/DeepMind |
| Navigation | **Dijkstra on waypoint graph** over SLAM map | Clean path from user position to target object |

---

## Week 1 — Perception & Spatial Memory

### Day 1 — Environment Setup
- Install CUDA stack, PyTorch, Gemma 3 27B via `ollama` or HuggingFace `transformers` on DGX Spark
- Clone and verify: `gaussian-splatting`, `Grounded-SAM-2`, `projectaria_tools`
- Confirm Aria VRS pipeline end-to-end (`vrs_to_json.py` → Three.js viewer)

### Day 2 — 3D Gaussian Splatting Reconstruction
- Run `scripts/run_colmap.py` to extract undistorted frames + camera poses via COLMAP SfM
- Run **3DGS training** on COLMAP output (~30 min on DGX Spark)
- Render a flythrough video of the reconstructed scene
- Export Gaussian splat centers as `.ply` for web viewer
- *(Future)* Swap COLMAP poses for ORB-SLAM3 loop-closed trajectory (Phase 2 SLAM)

### Day 3 — Open-Vocabulary 3D Object Detection
- Run **Grounded-SAM 2** on all SLAM keyframes with kitchen-specific text prompt
  (e.g. "dishes, toaster, sink, faucet, cooker, mug, bowl, knife, bottle")
- Per keyframe: produce per-object masks + class labels + confidence scores
- Unproject each 2D mask into 3D using depth map + camera intrinsics/extrinsics from Aria calibration
- Output: list of 3D bounding boxes with class labels, one per detected object
- *(Stretch — if 3DGS is trained by end of Day 2)* Run **LangSplat** on the Gaussian output:
  encodes CLIP features into each Gaussian so objects can be queried natively in 3D space,
  replacing the separate CLIP+FAISS step below with a single language-queryable 3D field

### Day 4 — Spatial Scene Graph
- Merge duplicate detections across keyframes (IoU in 3D + NMS)
- For each unique object: store 3D centroid, bounding box, best-view keyframe crop, class label
- Build **CLIP ViT-L/14 embeddings** for all object crops
- Persist to SQLite + FAISS index: `scene_db.sqlite` + `scene.faiss`

### Day 5 — Gemma 3 27B Integration
- Serve Gemma 3 27B multimodal on DGX Spark (HuggingFace `transformers` or `ollama`)
- Build query pipeline:
  1. User text query → CLIP text embedding → FAISS top-k retrieval → candidate object crops
  2. Feed top-3 candidate crops + query to Gemma 3 → confirm match, return object ID + reasoning
- Test with 10 natural language queries against the stored scene

### Day 6 — Re-localization
- Given a new camera frame, match against stored SLAM keyframes using CLIP cosine similarity
- Recover user's 6-DoF pose in the stored map frame
- Fall back to manual room selection if re-localization confidence is low

### Day 7 — Navigation Path Planning
- Build a **waypoint graph** by sampling navigable nodes along the SLAM trajectory (every 0.5 m)
- Run **Dijkstra** from re-localized user position to target object centroid
- Output: ordered list of 3D waypoints from user → object

---

## Week 2 — Visualization, Polish & Demo

### Day 8 — Three.js Interactive 3D Map
- Upgrade existing viewer to render 3DGS splat centers as colored point cloud
- Overlay 3D bounding boxes for detected objects (colored by class)
- Highlight queried object in red; draw navigation path as an animated dashed line
- Show user position as a camera frustum, updated on re-localization

### Day 9 — Rerun.io Real-Time Dashboard
- Instrument the full pipeline with **Rerun.io** logging:
  - SLAM trajectory streaming in real time
  - 3D bounding boxes as they are detected
  - CLIP similarity scores per object
  - VLM reasoning text alongside the matched crop
- Export a `.rrd` replay file for offline demo playback

### Day 10 — Web Demo UI
- Single-page app (vanilla JS or React):
  - Left: Three.js 3D scene (splats + objects + navigation path)
  - Right: chat-style query interface (type query → object highlighted in map)
  - Bottom: timeline scrubbing through the SLAM session
- Host locally via `python -m http.server`; record demo screencast

### Day 11 — Robustness & Edge Cases
- Handle query with no match (Gemma response: "I didn't see X in this room")
- Handle partial occlusion in 3D bounding box merging
- Add confidence threshold filtering for low-quality detections
- Stress-test with 50 diverse natural language queries

### Day 12 — Second Environment
- Record or download a second scene (different room; use Record3D on iPhone if Aria unavailable)
- Run full pipeline end-to-end without manual tuning
- Demonstrates generalization beyond a single scene

### Day 13 — Demo Video Production
- Record a 2-minute screen capture:
  1. Raw Aria data → 3DGS reconstruction (timelapse)
  2. Object detection bounding boxes populating the 3D map
  3. Natural language query: *"Where is the chair?"* → object highlighted + path drawn
  4. Rerun.io dashboard showing the real-time pipeline
- Add captions; export as MP4 and GIF thumbnail

### Day 14 — README, GitHub Polish & Writeup
- `README.md`: project overview, architecture diagram (Mermaid), quick-start, demo GIF
- `RESULTS.md`: retrieval accuracy (precision@1, precision@3) on a hand-labeled 50-query eval set
- Clean code: remove debug prints, add docstrings to public functions
- Pin all dependencies: `requirements.txt` + `environment.yml`
- Tag release `v0.1.0` on GitHub

---

## Deliverables by End of Week 2

| Deliverable | Format |
|---|---|
| 3DGS scene reconstruction | `.ply` + rendered flythrough video |
| Object detection + scene graph | SQLite DB + FAISS index |
| Gemma 3 spatial query demo | Running locally on DGX Spark |
| Interactive 3D web map | Browser Three.js app |
| Rerun.io real-time dashboard | `.rrd` replay file |
| 2-minute demo video | MP4 + GIF |
| Public GitHub repo | Clean code + README + results |

---

## DGX Spark Setup — Migration from macOS

### Step 1 — Verify the DGX environment
```bash
ssh user@dgx-spark-ip
nvidia-smi                  # confirm GPU visible, note CUDA version
nvcc --version              # confirm CUDA toolkit installed
cat /etc/os-release         # should be Ubuntu 22.04
python3 --version
```
DGX Spark ships with CUDA pre-installed. GB10 (Grace Blackwell) = **sm_100**.
Run `python -c "import torch; print(torch.cuda.get_device_capability())"` after
PyTorch install to confirm arch — use that value for `TORCH_CUDA_ARCH_LIST`.

### Step 2 — Transfer data from Mac
```bash
# From Mac — transfer VRS (256MB) and COLMAP output (45MB)
rsync -avh --progress \
  /Users/kiwooshin/work/aria_dataset/aria_gen2_sample_data_1.vrs \
  user@dgx-spark-ip:~/data/

rsync -avh --progress \
  /Users/kiwooshin/work/SpatialCortex/data/colmap/ \
  user@dgx-spark-ip:~/SpatialCortex/data/colmap/
```
COLMAP is already done on Mac — no need to re-run unless more frames are needed.

### Step 3 — Clone the repo
```bash
git clone https://github.com/KiwooShin/SpatialCortex.git ~/SpatialCortex
cd ~/SpatialCortex
```

### Step 4 — Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash && source ~/.bashrc
```

### Step 5 — Recreate the aria conda environment
```bash
conda create -n aria python=3.10 -y
conda activate aria
pip install projectaria-tools rerun-sdk pillow tqdm numpy

# Verify VRS pipeline works
python3 scripts/vrs_to_json.py \
  --vrs ~/data/aria_gen2_sample_data_1.vrs \
  --output data/aria_vrs.json
```

### Step 6 — Install 3D Gaussian Splatting
```bash
conda create -n gaussian_splatting python=3.10 -y
conda activate gaussian_splatting

# PyTorch with CUDA (match nvcc --version output)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Clone 3DGS with submodules
git clone https://github.com/graphdeco-inria/gaussian-splatting \
  --recursive ~/gaussian-splatting
cd ~/gaussian-splatting

pip install plyfile tqdm numpy Pillow scipy

# Build CUDA submodules — set arch to match GPU
export TORCH_CUDA_ARCH_LIST="10.0"   # Blackwell sm_100 (check nvidia-smi)
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### Step 7 — Run 3DGS training
```bash
conda activate gaussian_splatting
cd ~/gaussian-splatting

python train.py \
  -s ~/SpatialCortex/data/colmap \
  -m ~/SpatialCortex/data/gaussian_output \
  --iterations 30000
```

Input layout expected under `-s` (already produced by `run_colmap.py`):
```
data/colmap/
├── images/          ← undistorted JPEG frames (960x720)
└── sparse/0/
    ├── cameras.txt  ← PINHOLE 960 720 fx fy cx cy
    ├── images.txt
    └── points3D.txt ← 4,266 sparse points
```
Training ~30 min on DGX Spark. Output → `data/gaussian_output/`.

### Step 8 — Render and verify
```bash
python render.py -m ~/SpatialCortex/data/gaussian_output
python metrics.py -m ~/SpatialCortex/data/gaussian_output
# Renders → data/gaussian_output/train/ours_30000/renders/
```

### Environment map

| Item | Mac | DGX Spark |
|---|---|---|
| Repo | `/Users/kiwooshin/work/SpatialCortex` | `~/SpatialCortex` |
| VRS file | `/Users/kiwooshin/work/aria_dataset/` | `~/data/` |
| COLMAP output | `data/colmap/` | `data/colmap/` (rsync from Mac) |
| 3DGS repo | not needed | `~/gaussian-splatting/` |
| 3DGS output | not needed | `data/gaussian_output/` |
| conda env (scripts) | `aria` | `aria` |
| conda env (training) | not applicable | `gaussian_splatting` |

### Potential issues

| Issue | Fix |
|---|---|
| `TORCH_CUDA_ARCH_LIST` mismatch — crash at runtime | Check `torch.cuda.get_device_capability()`, set flag to match |
| Hopper chip instead of Blackwell | Use `export TORCH_CUDA_ARCH_LIST="9.0"` |
| Low 3DGS quality (46 frames marginal) | Re-run `run_colmap.py --every-nth 2` (200 frames), retrain |
| OOM (unlikely on 128GB) | Add `--resolution 2` to halve image size |

---

## Key Research Concepts Featured

- **3D Gaussian Splatting** (Kerbl et al., ICCV 2023 best paper) — scene representation
- **EFM3D / EVL** (Meta Reality Labs, 2024) — egocentric voxel lifting for 3D OBB detection; current primary detection backbone
- **Grounded-SAM 2** (Meta AI, 2024) — open-vocabulary 2D segmentation + video tracking; chosen over 2025 alternatives (Mosaic3D, OpenYOLO3D, SceneSplat) for lower pipeline complexity and faster time-to-demo
- **LangSplat / LangSplatV2** (CVPR 2024 / 2025) — stretch goal; embeds language into 3D Gaussians for native language-queryable scene representation
- **DUSt3R / MASt3R** (CVPR 2024 / NAVER Labs 2024) — feed-forward multi-view 3D reconstruction without calibration; potential COLMAP replacement
- **OpenMask3D / ConceptFusion / OpenScene** — open-vocabulary 3D scene understanding via multi-view CLIP fusion; next step after EFM3D
- **Embodied VLM querying** — spatial grounding with Gemma 3 27B multimodal
- **Scene graph memory** — structured persistent spatial representation
- **Lifelong localization** — reuse of a stored map across sessions
- **FAISS vector retrieval** — scalable semantic search over visual memory

---

## Query Pipeline Plan

**Goal**: Natural language → 3D object location, powered by CLIP retrieval + Gemma 3 27B visual confirmation.

```
scene_obbs.csv  →  [1] Crop extraction  →  crops/{scene}/{id}.jpg
                →  [2] CLIP encoding    →  512/768-dim vectors
                →  [3] DB build         →  scene_db.sqlite + scene.faiss
                →  [4] Query CLI        →  text → CLIP → FAISS → Gemma 3 → 3D location
```

### Step 1 — Crop Extraction (`scripts/extract_crops.py`)

For each fused object in `scene_obbs.csv`, find the single best RGB crop:
- Match fused object → raw snippet detection by class + Euclidean dist < 0.8 m; pick highest-prob match
- Project 3D OBB corners into fisheye image at that timestamp using the same pipeline as `visualize_efm3d_obbs.py`
- Take axis-aligned bounding rect of valid projected corners + 20% padding
- Apply CW 90° rotation before cropping; clamp to image bounds
- Fall back to next-best timestamp if fewer than 4 corners project

Inputs: `scene_obbs.csv`, `snippet_obbs.csv`, `main.vrs`, `closed_loop_trajectory.csv`
Output: `output/crops/{scene}/{obj_id}_{class}.jpg` + `crop_path`/`best_ts_ns` columns appended to scene_obbs

### Step 2 — CLIP Encoding + DB Build (`scripts/build_scene_db.py`)

- Model: `open_clip` ViT-L-14 / openai weights → 768-dim normalized vectors
- FAISS: `IndexFlatIP` (cosine similarity, exact search; ~100 objects total → flat is fine)
- SQLite schema: id, scene, name, prob, count, tx/ty/tz, quaternion, scale_x/y/z, crop_path, clip_idx

```sql
CREATE TABLE objects (
    id INTEGER PRIMARY KEY, scene TEXT, name TEXT,
    prob REAL, count INTEGER,
    tx REAL, ty REAL, tz REAL,
    qw REAL, qx REAL, qy REAL, qz REAL,
    scale_x REAL, scale_y REAL, scale_z REAL,
    crop_path TEXT, clip_idx INTEGER
);
```

Output: `data/scene_db.sqlite`, `data/scene.faiss`

### Step 3 — Query CLI (`scripts/query_scene.py`)

```
$ python scripts/query_scene.py --query "where is the chair"
Top matches:
  [1] chair  seq01  prob=0.99  pos=(-2.11, -0.81, -1.19)  sim=0.82
Gemma 3: "The office chair is 2.3m to your left, next to the sofa."
```

Flow: CLIP text encode → FAISS top-k → load crops + metadata → Gemma 3 27B visual confirmation → print 3D location
- `--no-vlm` flag to skip Gemma and use CLIP-only retrieval
- Gemma 3 access: `transformers` with `google/gemma-3-27b-it` or `ollama run gemma3:27b`

### Dependencies

```bash
# Add to efm3d env (already has projectaria_tools)
pip install open_clip_torch faiss-gpu

# New env for Gemma 3
conda create -n spatialcortex python=3.10 -y
pip install torch torchvision transformers accelerate open_clip_torch faiss-gpu pillow pandas numpy
```

### Build order

1. `extract_crops.py` — validate crops visually first
2. `build_scene_db.py` — embed validated crops → FAISS + SQLite
3. `query_scene.py --no-vlm` — verify CLIP retrieval quality
4. Wire in Gemma 3 for VLM confirmation

---

## Progress Log

### 2026-05-26 (continued — navigation + video render)

**Re-localization (`scripts/build_keyframe_index.py` + `scripts/navigate.py`)**

- Built CLIP ViT-L/14 image embedding index of all VRS keyframes across seq00/01/02 (stride=10 → 303 keyframes). Stored in `data/keyframe_index.faiss` (303 × 768d, IndexFlatIP) + `data/keyframe_index.csv` (faiss_idx, scene, ts_ns, pose columns).
- `navigate.py`: takes a query image (or `--scene` flag) + object name → CLIP re-localization → FAISS keyframe search → Dijkstra path on trajectory waypoints (step=0.5 m, kNN-5) → renders annotated fisheye frame with floor-projected waypoints + HUD compass arrow.
- Key geometry: trajectory Z≈−0.05 m (eye level). Floor Z computed per target: `floor_z = target_tz − scale_z/2 − 0.05`. Each waypoint is snapped to `floor_z` before fisheye projection so dots appear on the floor, not the ceiling.
- HUD compass (bottom-right, r=48): angle = `−arctan2(cross, dot)` between camera forward XY and target direction XY. Shows arrow + distance; "HERE" when within 0.3 m. Works for targets behind camera.
- Cross-scene handling: for same-scene targets renders from user's re-localized frame (natural AR view); for cross-scene targets shows target frame + note.
- Verified static navigation results: `output/nav_lamp.jpg`, `output/nav_bed.jpg`, `output/nav_sofa.jpg`.

**Full-sequence navigation video (`scripts/render_nav_video.py`)**

- Renders every RGB frame of a VRS sequence (up to 998 frames @ 10 fps) with: all scene OBBs projected, target highlighted white `>>> NAME <<<`, live HUD compass + distance, frame-counter banner.
- First run used `scene_obbs.csv` for OBBs — produced tilted, misaligned boxes (especially sofa).
- **Bug found and fixed**: `scene_obbs.csv` fuses quaternions by naive averaging across observations that alternate between two 90°-ambiguous orientations (sofa flips between 160° and 72° Z-rotation → fused average is 116° → box tilted 45° wrong). Fix: switched to `snippet_obbs.csv` (per-frame detections). For each video frame, binary-search to nearest snippet timestamp (~2 s spacing) and use those OBBs — same approach as `visualize_efm3d_obbs.py`, which was already correct.
- After fix: sofa, picture frames, chairs all align perfectly with the image content.
- Output: `output/nav_video_seq01_lamp_full_fixed.mp4` (998 frames @ 10 fps, ~100 s).
- `nearest_snip(ts_ns)` helper: bisect over 49 sorted snippet timestamps → O(log N) per frame.

### 2026-05-26

**3D Gaussian Splatting — training complete**

- 3DGS training ran on COLMAP data (200 images, PINHOLE 960×720, 4266 sparse points) for 30 000 iterations on DGX Spark.
- Output: `data/gaussian_output/point_cloud/iteration_30000/point_cloud.ply` (210 MB), `point_cloud.ply.splat` (27 MB for WebGL), `flythrough.mp4` (2.8 MB).
- Next: export splat centers to Three.js viewer overlay; run `render.py` for per-camera renders.

**VLM — switched from Gemma 3 to LLaVA 1.5 7B**

- Gemma 3 27B and 4B are gated on HuggingFace and require an access token not present on this machine.
- Switched VLM to `llava-hf/llava-1.5-7b-hf` (not gated, no token required, 13 GB).
- `query_scene.py` updated: `load_vlm()` uses `pipeline('image-to-text', device_map='auto', torch_dtype=float16)`; `vlm_confirm()` queries each top-3 candidate individually with a YES/NO prompt and picks the highest-scoring match.
- LLaVA download running in background (~7 GB received of ~13 GB as of this writing).

**Query pipeline — fully operational (Steps 1–3 complete)**

**Step 1 — Crop extraction (`scripts/extract_crops.py`)**
- For each fused object in `scene_obbs.csv`, find the highest-confidence raw snippet detection of the same class within 0.8 m, load that RGB frame from VRS, project the 3D OBB corners into the fisheye image (with CW-90° rotation), crop the axis-aligned bounding rect + 25% padding.
- 78 of 86 objects got valid crops across seq00 (15/17), seq01 (35/39), seq02 (28/30). 8 failures were objects partially off-screen at their best detection frame.
- Output: `output/crops/{seq}/{id}_{class}.jpg` + `scene_obbs_crops.csv` with `crop_path` and `best_ts_ns` columns.

**Step 2 — CLIP encoding + DB build (`scripts/build_scene_db.py`)**
- Model: CLIP ViT-L/14-quickgelu (OpenAI weights, correct QuickGELU activation — using plain ViT-L-14 caused an activation mismatch that degraded embeddings).
- FAISS strategy: image embeddings only in the index. Text embeddings (class name) for the 8 cropless objects are excluded from FAISS — mixing text and image embeddings in the same cosine index causes text-text similarity (~0.78) to dominate over image-text similarity (~0.20), breaking ranking.
- 8 cropless objects stored in SQLite only (has_image=0, clip_idx=-1); surfaced via class-name string match during query.
- Output: `data/scene_db.sqlite` (86 objects), `data/scene.faiss` (78 vectors × 768d, IndexFlatIP).
- Both scripts set `HF_HUB_OFFLINE=1` — all weights cached locally, no network calls during inference.

**Step 3 — Natural language query CLI (`scripts/query_scene.py`)**
- CLIP text embed → FAISS search (fetch ≥50 to ensure class-matched objects are found even if ranked low by raw CLIP sim) → two-bucket re-ranking:
  - Bucket A: objects whose class name words appear in the query → ranked by CLIP similarity
  - Bucket B: everything else → ranked by CLIP similarity
- Synonym expansion: sit→chair/sofa, sleep→bed/sofa, light→lamp, couch→sofa, screen→monitor/tv, storage→cabinet/shelf/dresser.
- SQLite fallback: cropless objects (has_image=0) appended if class name matches query words.
- Nearby-object context: for each result, queries SQLite for objects within 1.5 m to show spatial context ("Nearby: sofa, window, lamp").
- Flags: `--scene` (restrict to one scene), `--interactive` (loop), `--vlm` (Gemma 3 visual confirmation), `--no-vlm` (CLIP only).
- Gemma 3 path: multimodal `AutoModelForImageTextToText`, shows top-3 crops as images, asks which matches the query and where it is. Run `--download-vlm` once to cache model locally.

**Verified query results (CLIP-only, fully local):**

| Query | Rank 1 result | Correct? |
|---|---|---|
| "where is the sofa" | SOFA (seq00) | ✓ |
| "where is the bed" | BED (seq02) | ✓ |
| "find me a lamp" | LAMP (seq02) | ✓ |
| "I need to sleep" | BED (seq02) | ✓ |
| "where can I sit down" | CHAIR + SOFA | ✓ |
| "show me a chair near a window" | WINDOW + CHAIR (seq01) | ✓ |

**Visual query pipeline — `scripts/query_visual.py` (Step 4 complete)**

Full end-to-end: natural language → CLIP retrieval → full VRS frame with 3D OBBs drawn → LLaVA visual description.

- Flow: CLIP text embed → FAISS top-k → SQLite best match → load full RGB frame from VRS at `best_ts_ns` → project 3D OBBs of target (white highlight, thick border, `>>> NAME <<<` label + arrow) + context objects (class colors, thin) → feed annotated frame to LLaVA 1.5 7B → structured answer.
- LLaVA download confirmed complete: 14 GB cached at `~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/`.
- Fixed ranking bug: original code used substring matching (`"a" in "ladder"` → True), causing "find me a lamp" to return LADDER. Fixed to exact word-level matching (`name_words & qwords`).
- Output format (always printed, VLM appended when `--vlm`):
  ```
  ── Answer ─────────────────────────────────────────────────────
    Object   : BED
    Scene    : seq02
    Frame ts : 245288370176  (245.288 s into recording)
    Position : (-1.48, -1.09, -1.07) m
    Nearby   : container, table, window, lamp, chair
    Image    : output/query_bed.jpg

    LLaVA: "The bed is in the center of the room, surrounded by a container,
             table, window, lamp, and chair."
  ```
- Annotated frames saved to `output/query_result.jpg` (or `--out` path).

**Verified visual query results (LLaVA + annotated frame):**

| Query | Found | LLaVA answer |
|---|---|---|
| "where is the sofa" | SOFA seq00 | "The sofa is located in the center of the room, surrounded by a chair, a lamp, a ladder, and a picture frame." |
| "where is the bed" | BED seq02 | "The bed is in the center of the room, surrounded by a container, table, window, lamp, and chair." |
| "find me a lamp" | LAMP seq02 | "The lamp is located in the corner of the room, surrounded by a chair, table, and a flower pot." |

---

### 2026-05-25

**EFM3D 3D Object Detection — fully integrated**

- Downloaded AEO dataset sequences seq01 (living room) and seq02 (bedroom) in addition to previously available seq00.
- Ran EFM3D inference with `--snip_stride 2.0` for full sequence coverage (~50 snippets each) on all three sequences. Previous seq00 run used only 30 snippets (2.7% coverage); new runs cover the full sequence.
- Investigated model inputs: EFM3D uses **RGB only** (`video_streams: [rgb]`). SLAM cameras are loaded for timestamp intersection only. Semidense SLAM points are used as occupancy + freespace voxel channels. Fixed voxel extent `[-2,2,0,4,-2,2]` m; vol_min/max from semidense quantiles are computed but not consumed by the model.
- Identified that `track_obbs()` is blocked by `pytorch3d` (unavailable on aarch64/CUDA 13.0). Written `scripts/fuse_scene_obbs.py` as a drop-in replacement: greedy center-distance clustering per class → confidence-weighted position/log-scale averaging → Markley quaternion mean → accumulated evidence confidence → `scene_obbs.csv`.
- Fused scene maps: seq00 = 17 objects, seq01 = 39 objects, seq02 = 30 objects.

**Visualization improvements**

- Fixed Aria RGB 90° rotation: raw sensor image is 90° CCW from upright. Applied CW 90° rotation to both image array and projected pixel coordinates via `(N-1-v, u)` transform. Both fisheye projection and corner drawing are now geometrically correct.
- Added top-down bird's-eye view alongside fisheye overlay (side-by-side output per frame) in `visualize_efm3d_obbs.py`.
- Written `scripts/scene_topdown.py`: renders a single top-down scene map from `scene_obbs.csv`, auto-scaled to cover all objects + full camera trajectory. Generated maps for seq00/01/02.

**Research review**

- Reviewed cross-frame consistency approaches: StreamPETR (temporal memory queue), OpenMask3D (multi-view CLIP fusion on 3D masks), ConceptFusion (per-point CLIP accumulation), LangSplat/LERF (language-embedded scene representations).
- Identified DUSt3R (CVPR 2024, arXiv 2312.14132) and MASt3R (arXiv 2406.09756) as direct replacements for COLMAP that work from uncalibrated images; MASt3R adds a dense matching head on top of DUSt3R.
- Added `research.md` with detailed summaries of all related works.
