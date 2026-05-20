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
| 2D segmentation | **Grounded-SAM 2** (Grounding DINO + SAM 2) | Open-vocabulary, zero-shot object masking |
| 3D object lifting | **Depth unprojection + mask fusion** | Converts 2D masks → 3D bounding volumes in SLAM frame |
| Semantic indexing | **CLIP ViT-L/14 embeddings + FAISS** | Sub-millisecond vector search over object crops |
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
- Run **Grounded-SAM 2** on all SLAM keyframes with a broad text prompt ("all objects")
- Per keyframe: produce per-object masks + class labels + confidence scores
- Unproject each 2D mask into 3D using depth map + camera intrinsics/extrinsics from Aria calibration
- Output: list of 3D bounding boxes with class labels, one per detected object

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
- **Grounded-SAM 2** (Meta AI, 2024) — open-vocabulary 2D/3D segmentation
- **Embodied VLM querying** — spatial grounding with Gemma 3 27B multimodal
- **Scene graph memory** — structured persistent spatial representation
- **Lifelong localization** — reuse of a stored map across sessions
- **FAISS vector retrieval** — scalable semantic search over visual memory
