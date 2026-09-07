# Hawk-Eye Lite 3D: Cricket Ball Tracking & DRS Decision Engine 🏏

An advanced, modular Computer Vision and Kinematics pipeline in Python for detecting, tracking, metric 3D calibration, trajectory smoothing, and official ICC Decision Review System (DRS / LBW) adjudication from broadcast cricket match footage.

---

## 🚀 Key Features

### 1. Metric Calibration & 3D Reconstruction (core/calibration.py)
- **Metric Coordinate System:** Origin at bowling crease stumps 0 0 0$, $+Y$ towards batsman stumps ( \to 20.12\text{ m}$), $+X$ across pitch width ($-1.524\text{ m} \to +1.524\text{ m}$), $+Z$ height ( \to \text{stump height } 0.7112\text{ m}$).
- **Planar Homography & PnP:** Maps monocular 2D broadcast pixels $[u, v]$ to ground plane coordinates $[X, Y, Z=0]$.
- **Gravity-Constrained 3D Synthesis:** Fuses monocular ground homography with vertical projectile kinematics ($\ddot{Z} = -g = -9.81\text{ m/s}^2$) with (t_{bounce}) = 0$ to resolve the unconstrained depth ambiguity.

### 2. 6-State 3D Kalman Filter & Occlusion Interpolator (core/trajectory.py)
- State space: $\mathbf{x} = [X, Y, Z, \dot{X}, \dot{Y}, \dot{Z}]^T$.
- Interpolates complete 7+ frame occlusions with RMSE $< 0.03\text{ m}$ ($< 3\text{ cm}$).
- **Segmented Trajectory Modeling:**
  1. *Pre-bounce:* Parabolic flight.
  2. *Bounce Inflection:* Coefficient of restitution ( \in [0.55, 0.75]$) with spin angle deflection.
  3. *Post-Impact Projection to Stumps (=20.12\text{ m}$):* Numerical forward simulation considering aerodynamic drag ($) to predict 3D impact at the wicket line.

### 3. Official ICC DRS / LBW Decision Support Engine (core/drs_engine.py)
- Implements official ICC LBW playing conditions:
  - **Pitching:** IN-LINE, OUTSIDE LEG, OUTSIDE OFF (supporting RHB & LHB).
  - **Impact (at pad):** IN-LINE, UMPIRE\'S CALL ($>50\%$ ball volume intersecting outer stump edge), OUTSIDE OFF.
  - **Wickets (at =20.12\text{ m}$):** HITTING, UMPIRE\'S CALL (clipping outer edges/bails), MISSING.
- **Broadcast Banner Renderer:** Generates the official 3-light overlay banner:
  [ PITCHING: IN-LINE ] -> [ IMPACT: IN-LINE ] -> [ WICKETS: HITTING ] -> [ DECISION: OUT / NOT OUT ].

### 4. Interactive Streamlit App with Plotly 3D (ui/app.py)
- Full interactive 3D pitch rendering with metric pitch surface, stumps, bails, and segmented ball paths.
- Preset 3D camera angles: *Side-on*, *Bird\'s Eye*, *Bowler\'s POV*, *Batsman\'s POV*.
- Real-time parameter tuning: Handedness, on-field call, bowling speed, and pitch restitution.

### 5. Multi-Threaded CLI (main.py)
- Producer-consumer frame reader queue maximizing decoding and inference throughput.

### 6. Synthetic Training Pipeline (	raining/)
- 	raining/dataset.py: Synthetic broadcast generator using Albumentations for angled motion-blurred streaks (>140 km/h).
- 	raining/train.py: Ultralytics YOLOv8 fine-tuning script with small-object focus (imgsz=1280, focal loss).

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Ultralytics (YOLOv8)** & **OpenCV**
- **NumPy** & **SciPy**
- **Plotly** & **Matplotlib**
- **Streamlit**
- **Albumentations**
- **PyTest**

---

## 🧪 Running the Test Suite

`ash
pytest tests/ -v
`
All 25 deterministic unit tests pass with zero external video dependencies.

---

## 💻 Running the Streamlit Web Application

`ash
streamlit run ui/app.py
`

---

## ⚡ Running the Multi-Threaded CLI

`ash
# Run synthetic simulation and DRS adjudication
python main.py --simulate

# Run on a broadcast video feed
python main.py --input match.mp4 --output output/ --drs --plot-3d --threaded
`
