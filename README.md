# Hawk-Eye Lite 🏏

A modular Computer Vision pipeline in Python for detecting, tracking, smoothing, and visualizing a cricket ball from broadcast match footage.

## 🚀 Features & Architecture

- **Ball Detection (detector.py)**: YOLOv8 wrapper with adaptive dual-confidence thresholding for fast bowling motion blur streaks (>140 km/h) and false-positive heuristic filtering (size, aspect ratio, kinematic speed gating).
- **Tracking & GMC (	racker.py)**: BoT-SORT integration featuring Global Motion Compensation (GMC) via Lucas-Kanade optical flow affine estimation to handle broadcast camera panning and zooming. Includes automatic scene-cut detection to reset track IDs across camera switches.
- **Trajectory & Physics (	rajectory.py)**:
  - 6-state Kalman Filter ($[x, y, v_x, v_y, a_x, a_y]^T$) for trajectory smoothing and forward-backward 5+ frame occlusion interpolation.
  - Sub-frame Bounce Point Detector analyzing vertical velocity ($) and deceleration inflections (scipy.signal.find_peaks / topological prominence) within $\le 1$ frame margin of error.
- **Visualization (isualizer.py)**:
  - Glowing fading trajectory ribbon HUD on video frames.
  - 2D Top-Down Cricket Pitch Map with pitch creases, length zones (Short, Good Length, Full, Yorker), and bounce impact points.
- **Benchmarking (enchmark.py)**: Evaluates Detection AP@0.5, Tracking MOTA, MOTP, ID Switches, FP/FN counts, and module latency breakdown.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV**
- **Ultralytics (YOLOv8)**
- **NumPy & SciPy**
- **Matplotlib**
- **PyTest**

---

## 📦 Installation

`ash
git clone https://github.com/<your-username>/hawk-eye-lite.git
cd hawk-eye-lite
pip install -r requirements.txt
`

---

## 🧪 Running Tests

The test suite runs with synthetic NumPy arrays (zero external video dependencies required):

`ash
pytest tests/ -v
`

---

## 🎬 Running Pipeline Simulation

`ash
python pipeline.py
`
Generated artifacts will be saved in output/:
- output/pitch_map.png - 2D Cricket Pitch Map
- output/benchmark_metrics.png - Performance breakdown charts
- output/synthetic_hawkeye_demo.mp4 - Video with trajectory tail HUD
