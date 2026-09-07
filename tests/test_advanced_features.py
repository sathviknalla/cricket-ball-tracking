import pytest
import numpy as np
from core.trajectory import Kalman3DSmoother
from core.analytics import ExpectedDismissalEngine, PitchHeatmapAggregator, DeliverySessionTracker
from core.umpiring import NoBallDetector, UltraEdgeWaveformSimulator, UltraEdgeType

def test_magnus_force_computation():
    # Test Magnus lateral drift for spinning ball
    vel = (0.0, 38.0, -2.0)
    # 2000 RPM top-spin vs back-spin
    force_top = Kalman3DSmoother.compute_magnus_force(vel, spin_rpm=2000.0, spin_axis=(1.0, 0.0, 0.0))
    force_back = Kalman3DSmoother.compute_magnus_force(vel, spin_rpm=-2000.0, spin_axis=(1.0, 0.0, 0.0))
    
    assert isinstance(force_top, tuple)
    assert len(force_top) == 3
    # Opposing spin directions should produce opposing Z accelerations
    assert force_top[2] * force_back[2] <= 0.0

def test_confidence_cone_growth():
    sig_x_short, sig_z_short = Kalman3DSmoother.compute_confidence_ellipsoid(distance_projected=0.5)
    sig_x_long, sig_z_long = Kalman3DSmoother.compute_confidence_ellipsoid(distance_projected=3.5)
    
    assert sig_x_long > sig_x_short
    assert sig_z_long > sig_z_short
    assert sig_x_short > 0.0
    assert sig_z_short > 0.0

def test_advanced_projection_result():
    res = Kalman3DSmoother.project_to_stumps_advanced(
        impact_point=(0.0, 18.8, 0.45),
        velocity=(0.0, 35.0, -2.0),
        target_y=20.12,
        spin_rpm=1500.0
    )
    assert "trajectory" in res
    assert "stump_impact" in res
    assert "confidence_cone" in res
    assert "speed_timeline_kmh" in res
    assert len(res["trajectory"]) > 0
    assert abs(res["stump_impact"][1] - 20.12) < 1e-3
    assert res["confidence_cone"][0] > 0.0

def test_expected_dismissal_scoring():
    xd_engine = ExpectedDismissalEngine()
    
    # Delivery right on good length hitting middle stump at 145 km/h
    res_out = xd_engine.calculate_xd(
        bounce_point=(0.0, 12.0, 0.0),
        impact_point=(0.0, 18.8, 0.45),
        stump_impact=(0.0, 20.12, 0.35),
        speed_kmh=145.0,
        is_rhb=True
    )
    assert 0.0 <= res_out["xd_score"] <= 1.0
    assert res_out["xd_percentage"] > 50.0
    assert res_out["length_zone"] == "Good Length"
    assert res_out["line_zone"] == "On The Stumps"

    # Wild bouncer down leg side
    res_bouncer = xd_engine.calculate_xd(
        bounce_point=(0.60, 6.5, 0.0),
        impact_point=(0.80, 18.8, 0.95),
        stump_impact=(1.10, 20.12, 1.40),
        speed_kmh=130.0,
        is_rhb=True
    )
    assert res_bouncer["xd_score"] < res_out["xd_score"]
    assert res_bouncer["length_zone"] == "Bouncer"

def test_pitch_heatmap_aggregation():
    heatmap = PitchHeatmapAggregator(grid_rows=10, grid_cols=10)
    heatmap.add_bounce(0.0, 11.5)
    heatmap.add_bounce(-0.1, 12.0)
    
    norm_hm = heatmap.get_normalized_heatmap()
    assert norm_hm.shape == (10, 10)
    assert np.max(norm_hm) == pytest.approx(1.0, abs=1e-3)
    assert len(heatmap.deliveries) == 2

def test_delivery_session_tracker():
    tracker = DeliverySessionTracker()
    tracker.record_delivery(
        speed_kmh=142.0,
        bounce_point=(-0.04, 11.5, 0.0),
        pad_point=(-0.02, 18.8, 0.45),
        stump_point=(-0.04, 20.12, 0.38),
        verdict="OUT"
    )
    stats = tracker.get_summary_stats()
    assert stats["total_deliveries"] == 1
    assert stats["avg_speed_kmh"] == 142.0
    assert stats["good_length_percentage"] == 100.0

def test_no_ball_detector():
    detector = NoBallDetector(popping_crease_y=1.22)
    
    # Legal delivery: heel lands at 1.15m (behind 1.22m)
    legal = detector.evaluate_front_foot(foot_landing_y=1.15)
    assert legal.is_no_ball is False
    assert legal.margin_cm < 0.0

    # Overstep No-Ball: heel lands at 1.28m (ahead of 1.22m)
    no_ball = detector.evaluate_front_foot(foot_landing_y=1.28)
    assert no_ball.is_no_ball is True
    assert no_ball.margin_cm > 0.0

def test_ultraedge_waveform():
    sim = UltraEdgeWaveformSimulator(sample_rate_hz=500)
    
    # Bat edge should generate prominent spike
    edge_res = sim.generate_waveform(n_frames=30, contact_type=UltraEdgeType.BAT_EDGE, contact_frame=15)
    assert edge_res["is_spike_detected"] is True
    assert edge_res["peak_amplitude"] > 0.60
    assert len(edge_res["waveform"]) == len(edge_res["time_axis"])

    # Clean pass should have low amplitude ambient signal
    clean_res = sim.generate_waveform(n_frames=30, contact_type=UltraEdgeType.CLEAN_PASS, contact_frame=15)
    assert clean_res["peak_amplitude"] < 0.40
