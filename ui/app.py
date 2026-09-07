import os
import cv2
import random
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from core.calibration import MetricPitchCalibration, PitchDimensions
from core.trajectory import Kalman3DSmoother
from core.drs_engine import DRSEngine, BatsmanHand, OnFieldCall, PitchingZone, ImpactZone, WicketsResult
from core.visualizer import Plotly3DVisualizer

st.set_page_config(page_title="Hawk-Eye 3D & DRS Decision Engine", page_icon="🏏", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #0f172a; color: #f8fafc; }
    .metric-card { background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

st.title("🏏 Hawk-Eye Lite: 3D Trajectory & DRS Decision Engine")
st.caption("Broadcast Cricket Ball Tracking, Metric 3D Reconstruction, and Official ICC LBW Adjudication")

# Sidebar Configuration
st.sidebar.header("⚙️ Simulation & Match Controls")
source_option = st.sidebar.selectbox("Trajectory Source", ["Synthetic Broadcast Delivery", "Upload Video Feed"])

col_side1, col_side2 = st.sidebar.columns(2)
with col_side1:
    handedness_str = st.selectbox("Batsman Stance", ["Right-Handed (RHB)", "Left-Handed (LHB)"])
    batsman_hand = BatsmanHand.RIGHT_HAND if "Right" in handedness_str else BatsmanHand.LEFT_HAND
with col_side2:
    umpire_call_str = st.selectbox("On-Field Call", ["NOT OUT", "OUT"])
    on_field_call = OnFieldCall.NOT_OUT if umpire_call_str == "NOT OUT" else OnFieldCall.OUT

delivery_type = st.sidebar.selectbox(
    "Delivery Preset",
    ["Classic Good Length (LBW Three Reds)", "Outside Leg Stump (Pitching Outside Leg)", "Edge Clipping (Umpire's Call on Wickets)", "Sliding Down Leg (Missing Stumps)"]
)

st.sidebar.subheader("Physics & Tracking Parameters")
speed_kmh = st.sidebar.slider("Bowling Speed (km/h)", 110.0, 160.0, 142.5, step=0.5)
conf_thresh = st.sidebar.slider("Detector Confidence", 0.10, 0.90, 0.25, step=0.05)
restitution = st.sidebar.slider("Pitch Restitution (e)", 0.50, 0.80, 0.65, step=0.01)

# Generate Trajectory Based on Presets
n_frames = 60
fps = 30.0
dt = 1.0 / fps
speed_ms = speed_kmh / 3.6

# Setup 3D delivery coordinates
if "Good Length" in delivery_type:
    bounce_y = 11.5
    bounce_x = -0.04
    pad_y = 18.8
    pad_x = -0.02
    pad_z = 0.45
    proj_vx = 0.03
    proj_vz = -0.3
elif "Outside Leg" in delivery_type:
    bounce_y = 10.5
    bounce_x = 0.18 if batsman_hand == BatsmanHand.RIGHT_HAND else -0.18
    pad_y = 18.8
    pad_x = 0.12 if batsman_hand == BatsmanHand.RIGHT_HAND else -0.12
    pad_z = 0.48
    proj_vx = -0.04 if batsman_hand == BatsmanHand.RIGHT_HAND else 0.04
    proj_vz = -0.2
elif "Edge Clipping" in delivery_type:
    bounce_y = 12.0
    bounce_x = -0.08
    pad_y = 18.8
    pad_x = -0.11
    pad_z = 0.42
    proj_vx = -0.03
    proj_vz = -0.1
else: # Missing
    bounce_y = 11.0
    bounce_x = 0.05
    pad_y = 18.8
    pad_x = 0.15
    pad_z = 0.52
    proj_vx = 0.15
    proj_vz = 0.2

# 1. Pre-bounce points
pre_pts = []
for y in np.linspace(1.5, bounce_y, 25):
    prog = (y - 1.5) / (bounce_y - 1.5)
    x = 0.1 + prog * (bounce_x - 0.1)
    z = 2.15 - (prog ** 1.6) * 2.15
    pre_pts.append((x, y, max(0.0, z)))

pre_arr = np.array(pre_pts, dtype=np.float64)
bounce_point = (bounce_x, bounce_y, 0.0)

# 2. Post-bounce to pad impact
post_pts = []
for y in np.linspace(bounce_y, pad_y, 18):
    prog = (y - bounce_y) / (pad_y - bounce_y)
    x = bounce_x + prog * (pad_x - bounce_x)
    z = 0.0 + (prog * 0.7 - 0.25 * (prog ** 2))
    post_pts.append((x, y, max(0.0, z)))

post_arr = np.array(post_pts, dtype=np.float64)
pad_impact_point = (pad_x, pad_y, pad_z)

# 3. Post-impact DRS projection to stumps (Y=20.12m)
proj_arr, stump_point = Kalman3DSmoother.project_to_stumps(
    impact_point=pad_impact_point,
    velocity=(proj_vx * speed_ms, speed_ms * 0.85, proj_vz * speed_ms),
    target_y=20.12,
    fps=fps
)

# 4. Evaluate Official DRS LBW Verdict
drs_engine = DRSEngine()
verdict = drs_engine.evaluate_lbw(
    bounce_point=bounce_point,
    pad_impact_point=pad_impact_point,
    predicted_stump_point=stump_point,
    batsman_hand=batsman_hand,
    on_field_call=on_field_call
)

# Layout Presentation
tab1, tab2, tab3 = st.tabs(["📊 Interactive 3D Ball Tracking", "🎯 Official DRS Decision", "📈 Trajectory Telemetry"])

with tab1:
    st.subheader("Interactive 3D Pitch Reconstruction")
    visualizer_3d = Plotly3DVisualizer()
    fig_3d = visualizer_3d.build_3d_pitch_figure(
        pre_bounce_traj=pre_arr,
        post_bounce_traj=post_arr,
        projected_traj=proj_arr,
        bounce_point=bounce_point,
        impact_point=pad_impact_point,
        stump_impact=stump_point,
        verdict=verdict
    )
    
    # Camera Presets
    cam_col1, cam_col2, cam_col3, cam_col4 = st.columns(4)
    with cam_col1:
        if st.button("👁️ Side-on View"):
            fig_3d.update_layout(scene_camera=dict(eye=dict(x=-3.2, y=0.0, z=0.5), center=dict(x=0, y=0.5, z=0)))
    with cam_col2:
        if st.button("🦅 Bird's Eye View"):
            fig_3d.update_layout(scene_camera=dict(eye=dict(x=0.0, y=0.0, z=4.0), center=dict(x=0, y=0.5, z=0)))
    with cam_col3:
        if st.button("🎯 Bowler's POV"):
            fig_3d.update_layout(scene_camera=dict(eye=dict(x=0.0, y=-2.5, z=1.8), center=dict(x=0, y=0.6, z=0)))
    with cam_col4:
        if st.button("🏏 Batsman's POV"):
            fig_3d.update_layout(scene_camera=dict(eye=dict(x=0.0, y=23.0, z=1.2), center=dict(x=0, y=0.4, z=0)))

    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Official Decision Review System (DRS)")
    banner_img = drs_engine.render_drs_banner(verdict, save_path="output/drs_banner.png")
    st.image(banner_img, channels="BGR", use_column_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("1. Pitching", verdict.pitching.value, f"X = {verdict.pitching_coord[0]:.2f}m")
    c2.metric("2. Impact", verdict.impact.value, f"X = {verdict.impact_coord[0]:.2f}m")
    c3.metric("3. Wickets", verdict.wickets.value, f"Z = {verdict.stump_coord[2]:.2f}m")

    st.info(f"**DRS Outcome:** {verdict.final_verdict} — {verdict.reasons}")

with tab3:
    st.subheader("Trajectory Telemetry & Kinematics")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Release Speed", f"{speed_kmh:.1f} km/h")
    t2.metric("Pitch Length Zone", f"{bounce_y:.2f} m", "Good Length" if 10 <= bounce_y <= 14 else "Full/Short")
    t3.metric("Pad Impact Distance", f"{20.12 - pad_y:.2f} m from stumps")
    t4.metric("Predicted Stump Height", f"{stump_point[2]*100:.1f} cm", "Stumps (71.1cm)")

st.sidebar.markdown("---")
st.sidebar.caption("Hawk-Eye Lite v2.0 • Advanced Agentic CV")
