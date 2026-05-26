# SpatialCortex — Related Research

Reference document for research works directly relevant to the SpatialCortex pipeline.
Updated: 2026-05-25.

---

## 3D Scene Reconstruction

### DUSt3R — Geometric 3D Vision Made Easy
- **Authors**: Shuzhe Wang et al., NAVER LABS Europe
- **Venue**: CVPR 2024
- **arXiv**: 2312.14132
- **What it does**: Feed-forward vision transformer that takes an arbitrary collection of uncalibrated RGB images and outputs dense 3D pointmaps, camera poses, and pixel-level correspondences in a single pass. No calibration file, no SfM pipeline, no iterative optimization needed.
- **Why it matters**: End-to-end replacement for COLMAP. If scene texture is poor or image overlap is low, DUSt3R is more robust than feature-based SfM.

### MASt3R — Grounding Image Matching in 3D with MASt3R
- **Authors**: Vincent Leroy et al., NAVER LABS Europe
- **Venue**: 2024
- **arXiv**: 2406.09756
- **What it does**: Extends DUSt3R with a second head that jointly predicts dense local feature descriptors alongside 3D pointmaps. Enables robust image matching + 3D reconstruction from unordered collections. Achieves ~80% reduction in median rotation error vs DUSt3R and scales to thousands of images.
- **Why it matters**: Drop-in upgrade over DUSt3R when precise image matching matters. Directly applicable to Aria RGB frames for scene reconstruction without COLMAP.

### 3D Gaussian Splatting
- **Authors**: Kerbl et al.
- **Venue**: ICCV 2023 (Best Paper)
- **What it does**: Represents scenes as a collection of 3D Gaussians with learnable position, covariance, opacity, and spherical harmonics color. Renders in real-time via differentiable splatting.
- **Why it matters**: Foundation for the SpatialCortex scene representation. Downstream of COLMAP/DUSt3R; trains in ~30 min on DGX Spark.

---

## 3D Object Detection and Tracking

### EFM3D / EVL (Egocentric Voxel Lifting)
- **Authors**: Meta Reality Labs (Stefan Wiedemann et al.)
- **Venue**: 2024
- **What it does**: DINOv2 2D features lifted into a gravity-aligned 3D voxel grid (96×96×96 at 4 cm/voxel). Predicts per-voxel oriented bounding boxes (OBBs) via centerness + bbox + classification heads. Trained on AEO (Aria Everyday Objects) dataset with 29 semantic classes.
- **Inputs**: RGB only (despite loading SLAM cameras); semidense SLAM points for occupancy + freespace voxel channels.
- **Limitation**: `track_obbs()` depends on pytorch3d which is unavailable on aarch64/CUDA 13.0. Fixed with custom `fuse_scene_obbs.py`.
- **Why it matters**: Current primary 3D detection backbone in SpatialCortex. Produces world-frame OBBs per snippet.

### StreamPETR — Streaming Perception with Temporal Context
- **Authors**: Wang et al.
- **Venue**: ICCV 2023
- **What it does**: Maintains a memory queue of 3D reference points (object queries) propagated across frames using cross-attention. Each new frame attends to the growing temporal context rather than being processed independently. SOTA on nuScenes 3D detection.
- **Why it matters**: Addresses cross-frame inconsistency at the architecture level — detections are temporally coherent by design, not post-hoc fused. Alternative direction to `fuse_scene_obbs.py`.

### ObbTracker (EFM3D built-in)
- **What it does**: Hungarian matching between per-snippet OBBs (cost = 8×class match + 1×center dist + 4×2D-GIoU; 3D IoU disabled). Running-average update for position/rotation/scale (window cap w_max=30). Outputs scene objects after w_min=5 observations.
- **Limitation**: Blocked by `pytorch3d.ops.iou_box3d` import in `obb_utils.py`; not available on aarch64/CUDA 13.0.
- **Replacement**: `scripts/fuse_scene_obbs.py` implements offline equivalent using greedy center-distance clustering per class, confidence-weighted position averaging (scale in log space), Markley quaternion mean, and accumulated evidence confidence.

---

## Open-Vocabulary 3D Scene Understanding

### OpenMask3D — Open-Vocabulary 3D Instance Segmentation
- **Authors**: Takmaz et al.
- **Venue**: NeurIPS 2023
- **What it does**: Computes 3D instance masks from point clouds (via Mask3D backbone), then aggregates per-view CLIP embeddings for each mask across all camera views. Produces class-agnostic 3D instance segmentation queryable by natural language at inference time.
- **Why it matters**: Directly applicable to Aria data: take MPS semidense point cloud + all RGB frames, run OpenMask3D to get consistent language-queryable 3D instances without per-frame detection inconsistency.

### ConceptFusion — Open-Set Multimodal 3D Mapping
- **Authors**: Jatavallabhula et al.
- **Venue**: RSS 2023
- **What it does**: Fuses per-pixel CLIP + SAM segment features into a globally consistent 3D point cloud. Each 3D point accumulates CLIP embeddings from all views in which it appears. Queryable by image, text, audio, or sketch.
- **Why it matters**: Multi-modal scene memory. Consistency from geometric fusion — a 3D point always has the same embedding regardless of which frame contributed it. Simpler than OpenMask3D (no learned 3D backbone).

### OpenScene — Open Vocabulary 3D Scene Understanding
- **Authors**: Peng et al.
- **Venue**: CVPR 2023
- **What it does**: Distills 2D CLIP/LSeg features into 3D point clouds by nearest-neighbor projection from all views. One semantic embedding per 3D point, computed offline. Queries "which 3D points match this text?" at test time.
- **Why it matters**: Simpler, faster alternative to OpenMask3D. Good baseline for converting Aria semidense points into a queryable semantic 3D map.

---

## Language-Embedded Scene Representations

### LERF — Language Embedded Radiance Fields
- **Authors**: Kerr et al.
- **Venue**: ICCV 2023
- **What it does**: Trains a NeRF that also outputs a CLIP embedding at every 3D point. Multi-scale CLIP features embedded alongside RGB/density. Query by text → relevancy volume → 3D localization of language concepts.
- **Why it matters**: Spatially consistent semantic field — consistency is guaranteed because NeRF is a continuous function. Slower to train than 3DGS-based approaches.

### LangSplat — Language 3D Gaussian Splatting
- **Authors**: Qin et al.
- **Venue**: CVPR 2024
- **What it does**: Trains a small autoencoder to compress CLIP features, then attaches compressed CLIP embeddings to each 3D Gaussian. Language queries rendered at 199× faster than LERF via Gaussian splatting. Three-scale hierarchy captures object/part/scene semantics.
- **Why it matters**: Stretch goal for SpatialCortex — replaces CLIP+FAISS vector DB with a native language-queryable 3D representation. Directly downstream of 3DGS training.

---

## Egocentric / Embodied 3D Perception

### EmbodiedScan — Egocentric Multi-View 3D Perception
- **Authors**: Wang et al.
- **Venue**: CVPR 2024
- **What it does**: Dataset + model for holistic 3D scene understanding from egocentric multi-view RGB-D frames. Covers 3D detection, VQA, and grounding in a unified benchmark. Uses a transformer that aggregates features across all views.
- **Why it matters**: Directly addresses the same setting as SpatialCortex — egocentric video of a room → 3D object understanding.

---

## Cross-Frame Consistency Methods (Summary)

| Approach | Consistency Mechanism | Applicable to Aria? |
|---|---|---|
| `fuse_scene_obbs.py` (ours) | Greedy clustering + weighted averaging offline | Yes — done |
| EFM3D ObbTracker | Hungarian match + running average online | Yes — blocked by pytorch3d |
| StreamPETR | Temporal memory queue in transformer | Needs retraining |
| OpenMask3D | 3D instance masks + multi-view CLIP fusion | Yes — no retraining |
| ConceptFusion | Per-point CLIP accumulation from all views | Yes — no retraining |
| LangSplat | Language embedded in 3DGS | Yes — after 3DGS training |
| DUSt3R / MASt3R | Global 3D reconstruction from all frames | Yes — replaces COLMAP |
