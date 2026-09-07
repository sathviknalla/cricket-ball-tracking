import os
import cv2
import random
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from core.calibration import MetricPitchCalibration, PitchDimensions
from core.trajectory import Kalman3DSmoother
from core.drs_engine import DRSEngine, BatsmanHand, OnFieldCall
from core.visualizer import Plotly3DVisualizer
from core.analytics import ExpectedDismissalEngine, DeliverySessionTracker
from core.umpiring import NoBallDetector, UltraEdgeWaveformSimulator, UltraEdgeType

st.set_page_config(page_title="Hawk-Eye 3D Enterprise & DRS Engine", page_icon="🏏", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #0f172a; color: #f8fafc; }
    .metric-card { background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
    .badge-out { color: #f43f5e; font-weight: bold; }
    .badge-notout { color: #10b981; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏏 Hawk-Eye 3D Enterprise: Ball Tracking, DRS & Match Analytics")
st.caption("Broadcast Ball Tracking, Metric 3D Kinematics, Aerodynamics, UltraEdge, No-Ball Crease Adjudication, and Expected Dismissal (xD)")

# Session state initialization
if 'session_tracker' not in st.session_state:
    st.session_state.session_tracker = DeliverySessionTracker()
    # Pre-seed 5 deliveries for heatmap richness
    st.session_state.session_tracker.record_delivery(142.5, (-0.04, 11.5, 0.0), (-0.02, 18.8, 0.45), (-0.04, 20.12, 0.38), "OUT")
    st.session_state.session_tracker.record_delivery(138.0, (0.12, 10.2, 0.0), (0.08, 18.8, 0.50), (0.02, 20.12, 0.42), "NOT OUT")
    st.session_state.session_tracker.record_delivery(145.2, (-0.15, 12.8, 0.0), (-0.12, 18.8, 0.41), (-0.10, 20.12, 0.35), "OUT")
    st.session_state.session_tracker.record_delivery(140.0, (0.05, 9.1, 0.0), (0.10, 18.8, 0.65), (0.15, 20.12, 0.78), "NOT OUT")

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
    [
        "Classic Good Length (LBW Three Reds)",
        "Outside Leg Stump (Pitching Outside Leg)",
        "Edge Clipping (Umpire's Call on Wickets)",
        "Sliding Down Leg (Missing Stumps)",
        "UltraEdge Bat Snick Deflection",
        "Close Proximity Pitch (<40cm to Pad)"
    ]
)

st.sidebar.subheader("Physics & Aerodynamics")
speed_kmh = st.sidebar.slider("Bowling Speed (km/h)", 110.0, 160.0, 142.5, step=0.5)
spin_rpm = st.sidebar.slider("Ball Spin (RPM / Magnus Effect)", -2500, 2500, 450, step=50, help="Positive = off-spin drift, Negative = leg-spin drift")
foot_landing_y = st.sidebar.slider("Bowler Landing Crease Y (m)", 0.80, 1.45, 1.15, step=0.01, help="Popping crease at 1.22m")

# Setup 3D delivery coordinates
fps = 30.0
speed_ms = speed_kmh / 3.6
bat_edge_detected = False

if "Good Length" in delivery_type:
    bounce_y = 11.5
    bounce_x = -0.04
    pad_y = 18.8
    pad_x = -0.02
    pad_z = 0.45
    proj_vx = 0.03
    proj_vz = -0.30
elif "Outside Leg" in delivery_type:
    bounce_y = 10.5
    bounce_x = 0.18 if batsman_hand == BatsmanHand.RIGHT_HAND else -0.18
    pad_y = 18.8
    pad_x = 0.12 if batsman_hand == BatsmanHand.RIGHT_HAND else -0.12
    pad_z = 0.48
    proj_vx = -0.04 if batsman_hand == BatsmanHand.RIGHT_HAND else 0.04
    proj_vz = -0.20
elif "Edge Clipping" in delivery_type:
    bounce_y = 12.0
    bounce_x = -0.08
    pad_y = 18.8
    pad_x = -0.11
    pad_z = 0.42
    proj_vx = -0.03
    proj_vz = -0.10
elif "UltraEdge" in delivery_type:
    bounce_y = 11.8
    bounce_x = -0.06
    pad_y = 18.8
    pad_x = -0.04
    pad_z = 0.44
    proj_vx = 0.08
    proj_vz = -0.15
    bat_edge_detected = True
elif "Close Proximity" in delivery_type:
    bounce_y = 18.50  # 30cm from pad at 18.80
    bounce_x = -0.02
    pad_y = 18.80
    pad_x = -0.01
    pad_z = 0.22
    proj_vx = 0.02
    proj_vz = -0.05
else: # Missing
    bounce_y = 11.0
    bounce_x = 0.05
    pad_y = 18.8
    pad_x = 0.15
    pad_z = 0.52
    proj_vx = 0.15
    proj_vz = 0.20

# 1. Pre-bounce points
pre_pts = []
for y in np.linspace(1.5, bounce_y, 25):
    prog = (y - 1.5) / max(0.1, (bounce_y - 1.5))
    x = 0.1 + prog * (bounce_x - 0.1)
    z = 2.15 - (prog ** 1.6) * 2.15
    pre_pts.append((x, y, max(0.0, z)))

pre_arr = np.array(pre_pts, dtype=np.float64)
bounce_point = (bounce_x, bounce_y, 0.0)

# 2. Post-bounce to pad impact
post_pts = []
for y in np.linspace(bounce_y, pad_y, 18):
    prog = (y - bounce_y) / max(0.1, (pad_y - bounce_y))
    x = bounce_x + prog * (pad_x - bounce_x)
    z = 0.0 + (prog * 0.7 - 0.25 * (prog ** 2))
    post_pts.append((x, y, max(0.0, z)))

post_arr = np.array(post_pts, dtype=np.float64)
pad_impact_point = (pad_x, pad_y, pad_z)

# 3. Advanced DRS projection to stumps (Y=20.12m) with Magnus effect & Confidence Cones
proj_res = Kalman3DSmoother.project_to_stumps_advanced(
    impact_point=pad_impact_point,
    velocity=(proj_vx * speed_ms, speed_ms * 0.85, proj_vz * speed_ms),
    target_y=20.12,
    fps=fps,
    spin_rpm=float(spin_rpm),
    spin_axis=(0.0, 0.0, 1.0)
)
proj_arr = proj_res["trajectory"]
stump_point = proj_res["stump_impact"]
cone_radii = proj_res["cone_radii"]

# 4. Evaluate Official DRS LBW Verdict
drs_engine = DRSEngine()
verdict = drs_engine.evaluate_lbw(
    bounce_point=bounce_point,
    pad_impact_point=pad_impact_point,
    predicted_stump_point=stump_point,
    batsman_hand=batsman_hand,
    on_field_call=on_field_call,
    bat_edge_detected=bat_edge_detected
)

# 5. Expected Dismissal (xD) Analytics
xd_engine = ExpectedDismissalEngine()
xd_result = xd_engine.calculate_xd(
    bounce_point=bounce_point,
    impact_point=pad_impact_point,
    stump_impact=stump_point,
    speed_kmh=speed_kmh,
    is_rhb=(batsman_hand == BatsmanHand.RIGHT_HAND)
)

# 6. Front-Foot No-Ball Check
no_ball_detector = NoBallDetector()
no_ball_verdict = no_ball_detector.evaluate_front_foot(foot_landing_y)

# Layout Presentation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 3D Trajectory & Confidence Cones",
    "🎯 DRS Decision & UltraEdge",
    "📈 Velocity Telemetry & Drag Decay",
    "🗺️ Pitch Heatmap & Expected Dismissal (xD)",
    "⚖️ Front-Foot No-Ball Inspector"
])

visualizer_3d = Plotly3DVisualizer()

with tab1:
    st.subheader("Interactive 3D Pitch Reconstruction with 95% Confidence Cones")
    fig_3d = visualizer_3d.build_3d_pitch_figure(
        pre_bounce_traj=pre_arr,
        post_bounce_traj=post_arr,
        projected_traj=proj_arr,
        bounce_point=bounce_point,
        impact_point=pad_impact_point,
        stump_impact=stump_point,
        verdict=verdict,
        confidence_cone_radii=cone_radii
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
    st.subheader("Official Decision Review System (DRS) & UltraEdge")
    banner_img = drs_engine.render_drs_banner(verdict, save_path="output/drs_banner.png")
    st.image(banner_img, channels="BGR", use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1. Pitching", verdict.pitching.value, f"X = {verdict.pitching_coord[0]:.2f}m")
    c2.metric("2. Impact", verdict.impact.value, f"X = {verdict.impact_coord[0]:.2f}m")
    c3.metric("3. Wickets", verdict.wickets.value, f"Z = {verdict.stump_coord[2]:.2f}m")
    c4.metric("Uncertainty Margin (σ)", f"±{proj_res['confidence_cone'][0]*100:.1f} cm", "95% Confidence")

    st.info(f"**DRS Outcome:** {verdict.final_verdict} — {verdict.reasons}")

    # UltraEdge Waveform
    st.markdown("#### 🔊 UltraEdge / Snickometer Acoustic Waveform")
    ue_sim = UltraEdgeWaveformSimulator()
    ue_type = UltraEdgeType.BAT_EDGE if bat_edge_detected else UltraEdgeType.PAD_CONTACT
    wf_data = ue_sim.generate_waveform(contact_type=ue_type, contact_frame=35)

    wf_fig = go.Figure()
    wf_fig.add_trace(go.Scatter(
        x=wf_data["time_axis"], y=wf_data["waveform"],
        mode='lines', line=dict(color='#38bdf8' if not bat_edge_detected else '#f43f5e', width=1.5),
        name='Acoustic Signal'
    ))
    wf_fig.add_vline(x=wf_data["contact_time_sec"], line_width=2, line_dash="dash", line_color="#eab308",
                     annotation_text="Frame Contact", annotation_position="top right")
    wf_fig.update_layout(
        template='plotly_dark',
        title=f"UltraEdge Audio Trace: {wf_data['contact_type']}",
        xaxis_title="Time (seconds)", yaxis_title="Amplitude",
        height=220, margin=dict(l=40, r=20, t=35, b=25),
        paper_bgcolor='#0f172a', plot_bgcolor='#1e293b'
    )
    st.plotly_chart(wf_fig, use_container_width=True)

with tab3:
    st.subheader("Continuous Velocity Telemetry & Drag Profile")
    
    # Stitch continuous Y coordinates
    all_y = np.concatenate([pre_arr[:, 1], post_arr[:, 1], proj_arr[:, 1]])
    all_speeds = []
    for y_val in all_y:
        # Physical deceleration model from aerodynamic drag
        drag_loss = (y_val / 20.12) * (speed_kmh * 0.16)
        all_speeds.append(speed_kmh - drag_loss)

    vel_fig = visualizer_3d.build_velocity_timeline_figure(
        y_coords=all_y,
        speeds_kmh=all_speeds,
        bounce_y=bounce_y,
        impact_y=pad_y
    )
    st.plotly_chart(vel_fig, use_container_width=True)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Release Speed", f"{speed_kmh:.1f} km/h")
    t2.metric("Speed at Bounce", f"{speed_kmh * 0.91:.1f} km/h", f"-{speed_kmh*0.09:.1f} km/h")
    t3.metric("Speed at Stumps", f"{speed_kmh * 0.84:.1f} km/h", f"-{speed_kmh*0.16:.1f} km/h")
    t4.metric("Magnus Lateral Drift", f"{abs(spin_rpm)/2500.0 * 4.2:.1f} cm", f"{spin_rpm} RPM")

with tab4:
    st.subheader("2D Pitch Heatmap & Expected Dismissal (xD)")
    
    col_hm1, col_hm2 = st.columns([1.5, 1.0])
    with col_hm1:
        hm_fig = visualizer_3d.build_pitch_heatmap_figure(
            density_matrix=st.session_state.session_tracker.heatmap.get_normalized_heatmap(),
            deliveries=st.session_state.session_tracker.heatmap.deliveries
        )
        st.plotly_chart(hm_fig, use_container_width=True)
    
    with col_hm2:
        st.markdown("#### 🎯 Expected Dismissal (xD)")
        st.metric("xD Threat Score", f"{xd_result['xd_percentage']}%", xd_result['length_zone'])
        st.progress(xd_result['xd_score'])
        
        st.markdown(f"**Line Zone:** `{xd_result['line_zone']}`")
        st.markdown(f"**Length Zone:** `{xd_result['length_zone']}`")
        
        st.markdown("##### Spell Aggregates:")
        stats = st.session_state.session_tracker.get_summary_stats()
        st.write(f"- **Deliveries in Spell:** {stats['total_deliveries']}")
        st.write(f"- **Good Length %:** {stats['good_length_percentage']:.1f}%")
        st.write(f"- **Average Spell Speed:** {stats['avg_speed_kmh']:.1f} km/h")
        st.write(f"- **Average xD Threat:** {stats['avg_xd_score']*100:.1f}%")

with tab5:
    st.subheader("⚖️ Front-Foot No-Ball Crease Adjudication")
    st.markdown("Automated inspection of bowler's front-foot grounding relative to Law 21.5.")
    
    nb_c1, nb_c2 = st.columns([1.2, 1.0])
    with nb_c1:
        # Draw Crease Inspection diagram
        crease_fig = go.Figure()
        # Popping Crease line
        crease_fig.add_vline(x=1.22, line_width=3, line_color="white", annotation_text="Popping Crease (1.22m)", annotation_position="top right")
        # Bowling Crease line
        crease_fig.add_vline(x=0.0, line_width=3, line_color="#94a3b8", annotation_text="Bowling Crease (0.0m)", annotation_position="top left")
        
        # Shoe polygon
        shoe_color = "#f43f5e" if no_ball_verdict.is_no_ball else "#10b981"
        crease_fig.add_trace(go.Scatter(
            x=[foot_landing_y, foot_landing_y + 0.28, foot_landing_y + 0.28, foot_landing_y, foot_landing_y],
            y=[-0.05, -0.05, 0.05, 0.05, -0.05],
            fill='toself', fillcolor=shoe_color,
            line=dict(color='white', width=1.5),
            name="Bowler Foot"
        ))
        crease_fig.update_layout(
            template='plotly_dark',
            title="Front Foot Crease Grounding View",
            xaxis_title="Pitch Length Y (meters)",
            xaxis=dict(range=[-0.2, 1.8]), yaxis=dict(range=[-0.2, 0.2], showticklabels=False),
            height=260, margin=dict(l=30, r=30, t=35, b=25),
            paper_bgcolor='#0f172a', plot_bgcolor='#1e293b'
        )
        st.plotly_chart(crease_fig, use_container_width=True)

    with nb_c2:
        st.markdown("#### Adjudication Verdict")
        if no_ball_verdict.is_no_ball:
            st.error(f"🚨 **{no_ball_verdict.verdict_text}**")
        else:
            st.success(f"✅ **{no_ball_verdict.verdict_text}**")
        
        st.metric("Landing Position", f"{foot_landing_y:.2f} m", f"Crease: 1.22m")
        st.metric("Overstep Margin", f"{no_ball_verdict.margin_cm:+.1f} cm", "Law 21.5")

st.sidebar.markdown("---")
st.sidebar.caption("Hawk-Eye Enterprise v3.0 • Advanced Ball Tracking & DRS")
