# Spatial Memory AR/VR Navigation System

## Overview

A wearable AR/VR system (or robot-mounted) that builds a persistent spatial memory of any environment on first visit, then lets the user query it in natural language to retrieve object locations and navigate back to them.

---

## System Pipeline

### Phase 1 — Environment Mapping (On Entry)

When the user enters a new room or environment, the system automatically runs:

1. **SLAM** — constructs a metric 3D map and localizes the device in real time (trajectory + semi-dense point cloud)
2. **3D Object Detection** — runs a detector (e.g., EFM3D, DETIC-3D, or GroundingDINO + depth) on keyframes to detect, classify, and localize objects as 3D bounding boxes within the SLAM map
3. **Scene Graph Construction** — associates each detected object with its 3D position, orientation, and a representative image crop, then stores the bundle in a local spatial database

Output: a compact scene record — SLAM map + object graph + image crops — stored persistently per environment.

---

### Phase 2 — Spatial Memory Storage

The database entry for each environment contains:

| Field | Content |
|---|---|
| `env_id` | Unique identifier for the space |
| `map` | Sparse 3D point cloud + keyframe poses |
| `objects[]` | List of detected objects: class label, 3D centroid, bounding box, keyframe crop |
| `image_crops[]` | Cropped RGB patches of each object for VLM retrieval |

Because only keyframes and object bounding boxes are stored (not full video), memory footprint per environment stays small (tens of MB).

---

### Phase 3 — Natural Language Retrieval (VLM Query)

When the user later asks *"Where can I find the hammer?"*:

1. The query is passed to a **VLM** (e.g., GPT-4V, LLaVA, or a CLIP-based retriever)
2. The VLM matches the query against stored image crops and class labels in the spatial database
3. The top-matching object record is retrieved, returning its 3D position and associated image crop
4. The system re-localizes the user in the stored map using the current camera feed

---

### Phase 4 — AR Navigation

Once the target object location is known and the user is re-localized:

1. **Arrow overlay** — a directional AR arrow is rendered in the headset/glasses pointing toward the target object, updating in real time as the user moves
2. **Interactive 3D map** — an optional bird's-eye map shows the full room layout, the user's current position, and the target object highlighted, with a computed path overlaid
3. Navigation terminates when the user is within a threshold distance of the target

---

## Target Hardware

- **AR/VR glasses**: Meta Quest 3, Apple Vision Pro, or Project Aria (dev kit)
- **Robot**: any mobile platform with an RGB-D or stereo camera + IMU
- **Compute**: on-device (edge) for SLAM and detection; VLM query can be on-device (small model) or offloaded to API

---

## Tech Stack (Planned)

| Component | Candidate |
|---|---|
| SLAM | Project Aria MPS, ORB-SLAM3, or on-device VIO |
| 3D Detection | EFM3D, GroundingDINO + depth unprojection |
| VLM Retrieval | CLIP embeddings + GPT-4V or LLaVA |
| Spatial DB | SQLite + numpy arrays (local), or Qdrant for vector search |
| Visualization | Three.js (web), Unity (AR overlay), or RViz (robot) |

---

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | SLAM trajectory + point cloud visualization (this repo) | ✅ Done |
| 2 | Integrate real Aria VRS data via `vrs_to_json.py` | ✅ Done |
| 3 | 3D object detection on Aria keyframes | 🔲 Next |
| 4 | Spatial database: store objects with 3D poses + image crops | 🔲 Planned |
| 5 | VLM query interface (natural language → object retrieval) | 🔲 Planned |
| 6 | Re-localization against stored map | 🔲 Planned |
| 7 | AR navigation overlay (arrow + interactive map) | 🔲 Planned |
| 8 | End-to-end demo on Project Aria or Quest 3 | 🔲 Planned |
