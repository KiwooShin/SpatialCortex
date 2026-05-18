# SLAM AR Visualizer

**Interactive 3D visualization of SLAM trajectories and point clouds in the browser.**

> Explore how AR/VR devices perceive and map the world — no install required.

![Demo](docs/demo-placeholder.gif)

🔗 **[Live Demo →](https://YOUR_USERNAME.github.io/slam-ar-visualizer/)**

---

## What is this?

An interactive web-based tool that visualizes the internal state of SLAM (Simultaneous Localization and Mapping) systems used in AR/VR devices. Load a SLAM session and explore:

- **6DoF camera trajectory** — the path the device traveled through space
- **Semi-dense point cloud** — the 3D map the SLAM system built
- **Camera frustum playback** — step through the session frame-by-frame to see what the device "saw" at each moment
- **Keyframe highlighting** — see which frames the SLAM system selected as keyframes

Built with [Three.js](https://threejs.org/) and deployable on GitHub Pages. Designed to work with data from **Meta Project Aria** (AR glasses), **TUM RGB-D**, and **EuRoC MAV** datasets.

## Why this project?

SLAM is the core technology behind AR/VR spatial tracking — ARKit, ARCore, Meta Quest's Insight tracking all run SLAM internally. This visualizer makes the invisible visible: you can see exactly how a SLAM system builds its understanding of a 3D environment from camera and IMU data.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Browser (Three.js + WebGL)                     │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │PointCloud│  │Trajectory│  │Camera Frustum│  │
│  │ Renderer │  │ Renderer │  │   Playback   │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       └──────────────┼───────────────┘          │
│                      │                          │
│              ┌───────┴────────┐                 │
│              │  Data Loader   │                 │
│              │  (JSON / PLY)  │                 │
│              └───────┬────────┘                 │
└──────────────────────┼──────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   Aria MPS CSV   TUM RGB-D    EuRoC MAV
   (trajectory +  (groundtruth  (state_
    pointcloud)    + ORB-SLAM)   groundtruth)
```

## Supported Datasets

| Dataset | Source | Type | What you get |
|---------|--------|------|-------------|
| **Aria Everyday Activities** | Meta Project Aria glasses | AR glasses (egocentric) | 6DoF trajectory + semi-dense point cloud + eye gaze |
| **TUM RGB-D** | Handheld RGB-D camera | Indoor | Ground truth trajectory + ORB-SLAM output |
| **EuRoC MAV** | Drone (stereo + IMU) | Indoor flight | Ground truth + VINS-Mono/ORB-SLAM3 output |

## Quick Start

### View the live demo
Visit the [GitHub Pages demo](https://YOUR_USERNAME.github.io/slam-ar-visualizer/) — sample data is preloaded.

### Run locally
```bash
git clone https://github.com/YOUR_USERNAME/slam-ar-visualizer.git
cd slam-ar-visualizer
# Serve with any static server
npx serve .
# or
python3 -m http.server 8000
```

### Use your own data

#### From Aria MPS:
```bash
# Convert Aria CSV to viewer JSON format
python3 scripts/aria_to_json.py \
  --trajectory path/to/closed_loop_trajectory.csv \
  --points path/to/semidense_points.csv.gz \
  --output data/my_session.json
```

#### From TUM RGB-D:
```bash
python3 scripts/tum_to_json.py \
  --groundtruth path/to/groundtruth.txt \
  --pointcloud path/to/pointcloud.ply \
  --output data/my_session.json
```

## Controls

| Input | Action |
|-------|--------|
| Left drag | Orbit camera |
| Right drag | Pan |
| Scroll | Zoom |
| `Space` | Play / pause trajectory |
| `←` `→` | Step through frames |
| `P` | Toggle point cloud |
| `T` | Toggle trajectory line |
| `F` | Toggle camera frustum |
| `C` | Toggle point cloud coloring (height / confidence / uniform) |

## Data Format

The viewer consumes a single JSON file:

```json
{
  "metadata": {
    "dataset": "aria_aea",
    "sequence": "loc1_script1_seq1",
    "num_frames": 1200,
    "num_points": 45000
  },
  "trajectory": [
    {
      "timestamp_ns": 1000000,
      "position": [0.0, 0.0, 0.0],
      "orientation": [1.0, 0.0, 0.0, 0.0],
      "is_keyframe": true
    }
  ],
  "point_cloud": [
    {
      "position": [1.2, 0.5, -3.1],
      "confidence": 0.95
    }
  ]
}
```

## Project Structure

```
slam-ar-visualizer/
├── index.html          # Main viewer (Three.js)
├── src/
│   ├── viewer.js       # Core 3D viewer
│   ├── data-loader.js  # JSON/PLY loading
│   ├── trajectory.js   # Trajectory rendering + playback
│   ├── pointcloud.js   # Point cloud rendering
│   └── controls.js     # UI controls + keyboard shortcuts
├── scripts/
│   ├── aria_to_json.py # Convert Aria MPS CSV → viewer JSON
│   └── tum_to_json.py  # Convert TUM RGB-D → viewer JSON
├── data/
│   └── sample.json     # Preloaded sample for demo
└── docs/
    └── demo.gif        # Demo recording
```

## Roadmap

- [x] Project setup + Three.js viewer skeleton
- [ ] Point cloud rendering with height-based coloring
- [ ] Trajectory line rendering with keyframe markers
- [ ] Camera frustum playback (animated)
- [ ] Aria MPS data converter
- [ ] TUM RGB-D data converter
- [ ] EuRoC data converter
- [ ] Playback controls (play/pause/step)
- [ ] Point cloud confidence-based filtering
- [ ] GitHub Pages deployment
- [ ] WebXR VR mode (explore in VR headset)
- [ ] Multiple session overlay (compare trajectories)

## Technical Notes

**Why browser-based?** Existing SLAM visualization tools (rviz, Pangolin, Rerun) require local installation. A browser-based viewer lets anyone — including recruiters — see the result instantly.

**Performance:** Three.js `BufferGeometry` with `Points` material handles 100K+ points at 60fps. For larger point clouds, octree-based LOD is planned.

**Data privacy:** All processing happens client-side. No data is uploaded to any server.

## References

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) — Visual-inertial SLAM
- [Project Aria Tools](https://github.com/facebookresearch/projectaria_tools) — Aria data utilities
- [Three.js](https://threejs.org/) — WebGL 3D library

## License

MIT
