# SpatialCortex — 2-Week Build Plan

**Goal**: End-to-end spatial memory system for AR/VR — a recruiter-ready demo that shows perception, language understanding, and navigation in one coherent pipeline.

**Hardware**: DGX Spark (128 GB unified memory)
**VLM**: Gemma 3 27B (multimodal, on-device)
**Data**: Project Aria `aria_gen2_sample_data_1.vrs` + optional iPhone LiDAR scan

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
- Extract SLAM keyframes + camera poses from Aria VRS using `projectaria_tools`
- Run **3DGS training** on the keyframes (~30 min on DGX Spark)
- Render a flythrough video of the reconstructed scene
- Export Gaussian splat centers as `.ply` for web viewer

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

## Key Research Concepts Featured

- **3D Gaussian Splatting** (Kerbl et al., ICCV 2023 best paper) — scene representation
- **Grounded-SAM 2** (Meta AI, 2024) — open-vocabulary 2D/3D segmentation
- **Embodied VLM querying** — spatial grounding with Gemma 3 27B multimodal
- **Scene graph memory** — structured persistent spatial representation
- **Lifelong localization** — reuse of a stored map across sessions
- **FAISS vector retrieval** — scalable semantic search over visual memory
