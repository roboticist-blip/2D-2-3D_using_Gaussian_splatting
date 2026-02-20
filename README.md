# 🎯 Video → 3D Gaussian Splatting Pipeline

> Convert any 2D video into a photorealistic 3D scene and extract individual objects as `.ply` files — all inside Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/video_to_3d_gaussian_splatting_v2.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [What Is This?](#-what-is-this)
- [How It Works](#-how-it-works)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Bounding Box Guide](#-bounding-box-guide-extracting-the-right-region)
- [Output Files](#-output-files)
- [Tips for Best Results](#-tips-for-best-results)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Tech Stack](#-tech-stack)

---

## 🔍 What Is This?

This notebook implements a full **video-to-3D reconstruction pipeline** using **3D Gaussian Splatting (3DGS)** — a state-of-the-art technique that represents a 3D scene as millions of tiny, coloured, semi-transparent ellipsoids called *Gaussians*.

Unlike traditional mesh-based 3D reconstruction, Gaussian Splatting:

- Renders in **real-time** at high quality
- Preserves fine details like fur, hair, and translucent surfaces
- Produces compact `.ply` files compatible with standard 3D viewers
- Enables **object-level extraction** via spatial bounding box filtering

---

## ⚙️ How It Works

```
Input Video
    │
    ▼
┌─────────────────────┐
│  Frame Extraction   │  ffmpeg — pull N frames per second as JPEGs
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  COLMAP (SfM)       │  Estimate camera position for every frame
└─────────────────────┘
    │  (sparse 3D point cloud + camera poses)
    ▼
┌─────────────────────┐
│  Gaussian Splatting │  Train millions of 3D Gaussians to match all views
│  Training           │  (~7k iterations for preview, 30k for high quality)
└─────────────────────┘
    │  (point_cloud.ply — full scene)
    ▼
┌─────────────────────┐
│  Bounding Box Crop  │  Filter Gaussians by XYZ position to isolate object
└─────────────────────┘
    │
    ▼
cropped_object.ply  ✅
```

### What is a Gaussian?

Each Gaussian is a single data point that stores:

| Property | Description |
|---|---|
| **Position** (x, y, z) | 3D location in the scene |
| **Colour / SH coefficients** | View-dependent colour (looks different from each angle) |
| **Opacity (α)** | How transparent or solid it is |
| **Scale** (sx, sy, sz) | Size along each axis |
| **Rotation** (quaternion) | Orientation in 3D space |

Together, millions of these Gaussians form a photorealistic 3D representation that can be rendered from any angle.

---

## 📦 Requirements

### Google Colab (Recommended)
No local setup needed. The notebook handles everything automatically.

- ✅ Google account
- ✅ Colab Pro or Free tier (**T4 GPU** recommended)
- ✅ Your input video file

### Local Setup (Advanced)
```bash
# System dependencies
apt-get install colmap ffmpeg

# Python dependencies
pip install plyfile==0.8.1 tqdm torch torchvision

# Clone repo
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

> ⚠️ Local setup requires a CUDA-capable NVIDIA GPU with at least **6 GB VRAM**.

---

## 🚀 Quick Start

1. Open the notebook in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells from top to bottom
4. Upload your video when prompted in Step 2
5. After training completes, set your bounding box in Step 7 and download your `.ply` in Step 9

Total time: **~40–60 minutes** for a typical video (including COLMAP + 7k training iterations)

---

## 🗺️ Pipeline Walkthrough

### Step 1 — Environment Setup
Installs COLMAP, ffmpeg, and all Python dependencies. Clones the official [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) repository.

### Step 2 — Upload Video
Upload a video directly from your computer, or provide a download URL. Supported formats: `.mp4`, `.mov`, `.avi`.

### Step 3 — Frame Extraction
Uses `ffmpeg` to pull frames from the video at a configurable frame rate. The frames are saved as high-quality JPEGs while preserving the original aspect ratio.

```
fps = 2–5    →  good for long videos (1+ min), faster COLMAP
fps = 10     →  good for short, detailed videos (<30 sec)
```

**Aim for 100–400 frames total.** Too few means COLMAP has less overlap to work with; too many means slower training.

### Step 4 — COLMAP (Structure-from-Motion)
COLMAP analyses all frames and figures out:
- The **3D position** of each camera at each frame
- A **sparse point cloud** of the scene (key feature points)
- Lens **distortion parameters** for each image

This step runs on CPU (to avoid OpenGL issues in Colab headless mode) and can take 5–20 minutes depending on frame count and scene complexity.

> 💡 If COLMAP fails, the most common cause is insufficient texture in the video (e.g., plain walls, uniform backgrounds). Add a more detailed object or re-shoot with more varied texture.

### Step 5 — Gaussian Splatting Training
This is the core ML training step. The model starts from the COLMAP sparse point cloud and iteratively optimises millions of Gaussians to reproduce every input image as accurately as possible.

| Setting | Iterations | Time | Use Case |
|---|---|---|---|
| Preview | 7,000 | ~10 min | Quick check, debugging |
| Standard | 15,000 | ~20 min | Good balance |
| High Quality | 30,000 | ~30 min | Final output |

### Step 6 — Scene Exploration
Generates statistics and visualisations to help you locate your target object in 3D space:

- **Percentile table** — min, 5th–95th pct, max per axis
- **Axis histograms** — see where point density clusters
- **4-angle 3D scatter plots** — bird's eye, front, side, perspective views

### Step 7 — Bounding Box Crop & Export
Set `bbox_min` and `bbox_max` as 3D coordinates, and the notebook filters all Gaussians spatially to produce a cropped `.ply` containing only your target region.

### Step 8 — Visualise Cropped Result
Renders the cropped model from 4 angles so you can verify the crop looks correct before downloading.

### Step 9 — Download
Downloads the full scene `.ply`, cropped object `.ply`, and all visualisation images.

---

## 📦 Bounding Box Guide: Extracting the Right Region

This is the most nuanced step. Here's a systematic approach:

### Understanding the Coordinate System

```
        +Y (up)
         │
         │
         └────── +X (right)
        /
       /
      +Z (towards viewer / depth)
```

> ⚠️ The Y axis is sometimes **inverted** in COLMAP outputs. If your crop looks upside-down, negate your Y values.

### Method 1 — Use the Percentile Table (Easiest)

Look at the table printed in Step 6. Choose a row that corresponds to the density of your target:

| Scenario | Recommended range |
|---|---|
| Object fills most of the video | 10th – 90th percentile |
| Object is surrounded by background | 25th – 75th percentile |
| Scene has noisy outliers at edges | Ignore min/max, use percentiles |
| Object is off-centre | Use mean ± N×std to find it |

### Method 2 — Use the Axis Histograms (Accurate)

Each histogram shows how Gaussians are distributed along one axis. Your object appears as a **dense cluster**; background and noise appear as sparse tails.

- Find the left edge of the cluster → that becomes your `bbox_min` for that axis
- Find the right edge → that becomes your `bbox_max`
- Repeat for all three axes independently

### Method 3 — Use the 3D Views (Visual)

| View | What to look at |
|---|---|
| **XY (top)** | Set X and Y bounds |
| **XZ (front)** | Set X and Z bounds |
| **YZ (side)** | Set Y and Z bounds |
| **Perspective** | Confirm overall shape |

### Method 4 — Iterative Refinement (Most Precise)

Use the `quick_crop_stats()` cell to test multiple bounding boxes instantly without writing any file:

```python
quick_crop_stats(raw_ply, np.array([-1.0, -1.0, -1.0]),
                           np.array([ 1.0,  1.0,  1.0]), label="attempt 1")
```

**A good crop typically retains 20–60% of total Gaussians.**

- If you get > 80% → box is too large, tighten it
- If you get < 5% → box is too tight or wrongly placed, widen it
- If you get 0% → your values are outside the scene range entirely

### Common Bbox Problems & Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| 0 Gaussians found | Values outside scene range | Check Step 6 stats and use suggested ranges |
| Background included in crop | Bbox too wide | Tighten min/max toward the mean |
| Object is partially cut off | Bbox too narrow on one axis | Expand that axis's min or max |
| Object appears upside down | Y-axis inverted | Negate Y values in bbox |
| Floater artefacts in crop | Noise outside object | Add opacity threshold filter (see FAQ) |

---

## 📁 Output Files

| File | Description |
|---|---|
| `point_cloud.ply` | Full reconstructed scene (all Gaussians) |
| `cropped_object.ply` | Your extracted object/region |
| `axis_distribution.png` | Histogram plots per axis |
| `point_cloud_3d_preview.png` | 4-angle 3D scatter of full scene |
| `cropped_preview.png` | 4-angle 3D scatter of cropped region |

### Viewing `.ply` Files

Open your `.ply` files in any of these free viewers:

| Viewer | Platform | Link |
|---|---|---|
| **SuperSplat** | Browser | [playcanvas.com/supersplat](https://playcanvas.com/supersplat/editor) |
| **Luma AI** | Browser | [lumalabs.ai](https://lumalabs.ai) |
| **antimatter15/splat** | Browser / local | [GitHub](https://github.com/antimatter15/splat) |
| **GaussianSplatting-Unity** | Unity Editor | [GitHub](https://github.com/aras-p/UnityGaussianSplatting) |

---

## 💡 Tips for Best Results

### Video Recording
- **Move slowly** — fast movement causes motion blur that breaks COLMAP feature matching
- **Circle the object** — aim for 360° coverage so every surface is seen from multiple angles
- **Consistent lighting** — avoid harsh shadows or changing light sources
- **Keep object in frame** — COLMAP needs overlapping views; don't pan away and back
- **Textured subjects work best** — plain white walls, glass, and mirrors are very difficult for COLMAP
- **Short videos work fine** — a 30-second slow 360° orbit is better than a 5-minute shaky walkthrough

### Frame Extraction
- For a **30-second video**: use `fps=10` (300 frames)
- For a **2-minute video**: use `fps=3` (360 frames)
- For a **10-minute video**: use `fps=1` (600 frames) or clip the relevant portion first

### Training
- Start with `iterations=7000` to verify the pipeline works, then re-run at `30000` for the final output
- Higher resolution frames = better quality but more VRAM needed
- If you run out of GPU memory, reduce frame resolution in Step 3

---

## 🛠 Troubleshooting

### COLMAP Fails / No Sparse Model
```
sparse/0 directory not found
```
**Causes & fixes:**
- Not enough overlapping frames → increase `fps` in Step 3
- Scene lacks texture → video needs more surface detail
- Too much motion blur → re-record more slowly
- Too many frames with identical viewpoints → trim repetitive sections

### Training Crashes (CUDA Out of Memory)
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce image resolution in Step 3:
```python
max_dimension = 1280   # instead of 1920
```

### Training Produces Blurry/Noisy Results
- Increase `iterations` from 7000 to 30000
- Ensure COLMAP produced a good sparse model (check Step 4 output)
- Verify the input frames are sharp, not motion-blurred

### Cropped `.ply` Has Floaters / Scattered Noise
The bounding box includes some stray Gaussians. Options:
1. Tighten the bounding box further
2. Filter by opacity (see FAQ below)
3. Open in SuperSplat viewer and use its built-in crop tool

### Colab Session Times Out During Training
- Enable **Colab Pro** for longer runtimes
- Reduce `iterations` to 15000 as a compromise
- Mount Google Drive and save checkpoints periodically

---

## ❓ FAQ

**Q: Can I use this for indoor scenes?**
Yes. Indoor scenes work well as long as there is enough texture and you capture sufficient coverage of the space. Empty corridors or plain-painted rooms are more difficult.

**Q: How do I filter by opacity to remove noise?**
Add this filter after loading the PLY in Step 7:
```python
opacity = np.array(vertex['opacity'])
opacity_mask = opacity > -2.0   # adjust threshold (sigmoid scale)
mask = spatial_mask & opacity_mask
```

**Q: Can I extract multiple objects?**
Yes — run the crop cell multiple times with different bounding boxes and different output filenames.

**Q: Can I use this without Colab (locally)?**
Yes, but you need an NVIDIA GPU with CUDA support. Install the dependencies listed in the Requirements section and run the cells manually as Python scripts.

**Q: Why is the Y axis sometimes inverted?**
COLMAP and Gaussian Splatting use a right-handed coordinate system where Y points **down** (image convention). Some viewers and tools flip this. If your scene appears upside down, negate all Y values in your bounding box.

**Q: How large will the output `.ply` file be?**
The full scene `.ply` is typically **100–500 MB** depending on scene complexity and training iterations. Cropped objects are proportionally smaller based on the fraction of Gaussians retained.

**Q: Can I re-render the cropped object from new viewpoints?**
Yes. Load `cropped_object.ply` into the Gaussian Splatting viewer script and render from any camera angle.

---

## 🔬 Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Frame extraction | `ffmpeg` | Decode video into image frames |
| Camera pose estimation | `COLMAP` | Structure-from-Motion (SfM) |
| 3D reconstruction | `gaussian-splatting` (graphdeco-inria) | Train 3D Gaussian scene |
| CUDA rasterizer | `diff-gaussian-rasterization` | Differentiable rendering |
| KNN acceleration | `simple-knn` | Nearest-neighbour queries during training |
| PLY I/O | `plyfile` | Read / write `.ply` point cloud files |
| Numerics | `numpy`, `torch` | Array operations and GPU tensors |
| Visualisation | `matplotlib` | Histograms and 3D scatter plots |
| Runtime | Google Colab (T4 GPU) | Managed cloud GPU environment |

---

## 📚 References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [COLMAP: Structure-from-Motion Revisited](https://colmap.github.io/) — Schönberger & Frahm, CVPR 2016
- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Official implementation


---

<div align="center">
  Made with ❤️ using 3D Gaussian Splatting
</div>