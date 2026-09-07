# Hawk-Eye 3D Enterprise: Cricket Ball Tracking & DRS Decision Engine 🏏

[![CI/CD Pipeline](https://github.com/sathviknalla/cricket-ball-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/sathviknalla/cricket-ball-tracking/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced, modular Computer Vision, Aerodynamics, and Kinematics platform in Python for detecting, tracking, metric 3D calibration, trajectory smoothing, UltraEdge acoustic snicko analysis, Front-Foot No-Ball detection, Expected Dismissal (xD) metrics, and official ICC Decision Review System (DRS / LBW) adjudication from broadcast cricket match footage.

---

## 🚀 Key Features & Capabilities

### 1. 📐 Metric Calibration & 3D Reconstruction (`core/calibration.py`)
- **Metric Coordinate System:** Origin at bowling crease stumps $[0, 0, 0]$, $+Y$ towards batsman stumps ($0 \to 20.12\text{ m}$), $+X$ across pitch width ($-1.524\text{ m} \to +1.524\text{ m}$), $+Z$ height ($0 \to \text{stump height } 0.7112\text{ m}$).
- **Planar Homography & PnP:** Maps monocular 2D broadcast pixels $[u, v]$ to ground plane coordinates $[X, Y, Z=0]$.
- **Gravity-Constrained 3D Synthesis:** Fuses monocular ground homography with vertical projectile kinematics ($\ddot{Z} = -g = -9.81\text{ m/s}^2$) with $Z(t_{\text{bounce}}) = 0$ to resolve depth ambiguity.

### 2. 🌪️ 6-State 3D Kalman Filter with Aerodynamics (`core/trajectory.py`)
- **State space:** $\mathbf{x} = [X, Y, Z, \dot{X}, \dot{Y}, \dot{Z}]^T$.
- **Magnus Effect Modeling:** Simulates dynamic aerodynamic lift and lateral spin drift forces:
  $$\mathbf{F}_{\text{magnus}} = \frac{1}{2} C_L \rho A r (\boldsymbol{\omega} \times \mathbf{v})$$
- **Uncertainty Quantification:** Computes expanding 3D covariance error cones ($\pm \sigma_X, \pm \sigma_Z$) from pad contact to the stumps.
- **Continuous Velocity Telemetry:** Tracks ball deceleration curves under drag coefficient $\gamma = 0.0070$.

### 3. 🎯 Official ICC DRS / LBW Decision Support Engine (`core/drs_engine.py`)
- Implements official ICC LBW playing conditions & Law 36:
  - **Pitching:** IN-LINE, OUTSIDE LEG, OUTSIDE OFF (supporting RHB & LHB).
  - **Impact (at pad):** IN-LINE, UMPIRE'S CALL ($>50\%$ ball volume intersecting outer stump edge), OUTSIDE OFF (incorporating Law 36.1(e) Shot-Offered clause).
  - **Wickets (at $Y=20.12\text{ m}$):** HITTING, UMPIRE'S CALL (clipping outer edges/bails), MISSING.
  - **ICC 3-Meter Law & Close Proximity Pitch Rule:** Protects batsman when impact occurs $\ge 3.0\text{ m}$ from stumps or within $40\text{ cm}$ of pitch point.
- **Official Broadcast Banner Renderer:** Generates 3-light overlay banner:
  `[ PITCHING: IN-LINE ] -> [ IMPACT: IN-LINE ] -> [ WICKETS: HITTING ] -> [ DECISION: OUT / NOT OUT ]`.

### 4. 📊 Expected Dismissal (xD) & Match Analytics (`core/analytics.py`)
- **Expected Dismissal (xD) Score:** Evaluates statistical wicket threat ($0 \to 100\%$) based on pitch length zone (Yorker, Full, Good Length, Short, Bouncer), line alignment, and seam deviation.
- **2D Pitch Heatmap Aggregator:** Calculates spatial Gaussian density maps of pitch landing spots.
- **Session Spell Tracker:** Aggregates spell metrics (Good Length %, average speed, xD threat).

### 5. ⚖️ Front-Foot No-Ball & UltraEdge Acoustic Analysis (`core/umpiring.py`)
- **No-Ball Crease Inspector:** Evaluates bowler heel grounding relative to the $1.22\text{ m}$ popping crease with margin measurements in cm.
- **UltraEdge Acoustic Waveform Simulator:** Synthesizes $320\text{ Hz}$ resonant acoustic spikes for bat-edge deflections vs low-frequency thuds for pad contact.

### 6. 💻 5-Tab Interactive Streamlit Dashboard (`ui/app.py`)
- Tab 1: 📊 **3D Trajectory & Confidence Cones** (with 4 camera presets: *Side-on*, *Bird's Eye*, *Bowler POV*, *Batsman POV*).
- Tab 2: 🎯 **DRS Decision Banner + Live UltraEdge Audio Waveform**.
- Tab 3: 📈 **Velocity Telemetry & Drag Decay Curve**.
- Tab 4: 🗺️ **2D Pitch Heatmap & Expected Dismissal (xD) Gauge**.
- Tab 5: ⚖️ **Front-Foot No-Ball Crease Inspector**.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Ultralytics (YOLOv8)** & **OpenCV**
- **NumPy** & **SciPy**
- **Plotly** & **Matplotlib**
- **Streamlit**
- **Albumentations**
- **PyTest** & **GitHub Actions**

---

## 🧪 Running the Test Suite

```bash
pytest tests/ -v
```
All **39 deterministic unit tests** pass with 100% test coverage and zero external dependencies.

---

## 💻 Running the Streamlit Web Application

```bash
streamlit run ui/app.py
```
Access the dashboard live at **`http://localhost:8501`**.

---

## ⚡ Running the Multi-Threaded CLI

```bash
# Run synthetic simulation and DRS adjudication
python main.py --simulate

# Run on a broadcast video feed with multithreading
python main.py --input match.mp4 --output output/ --threaded
```
