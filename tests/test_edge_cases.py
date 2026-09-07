import numpy as np
import pytest
from core.trajectory import Kalman3DSmoother
from core.drs_engine import DRSEngine, BatsmanHand, OnFieldCall, PitchingZone, ImpactZone, WicketsResult
from core.calibration import MetricPitchCalibration

def test_full_toss_delivery():
    """
    Edge Case: Delivery does not bounce on the pitch (full toss / yorker on the full).
    Assert pipeline adjudicates without pitch bounce error and evaluates LBW correctly.
    """
    engine = DRSEngine()

    pad_impact = (-0.02, 18.80, 0.42)
    
    proj_traj, stump_impact = Kalman3DSmoother.project_to_stumps(
        impact_point=pad_impact,
        velocity=(0.01 * 35.0, 35.0, -0.2 * 35.0),
        target_y=20.12
    )

    verdict = engine.evaluate_lbw(
        bounce_point=None,  # No pitch bounce
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.OUT
    )

    assert verdict.pitching == PitchingZone.FULL_TOSS
    assert verdict.impact == ImpactZone.IN_LINE
    assert verdict.wickets == WicketsResult.HITTING
    assert verdict.final_verdict == "OUT"

def test_bat_edge_invalidates_lbw():
    """
    Edge Case: Trajectory exhibits a sharp angular deflection right before pad contact (bat edge).
    Assert bat deflection detector flags it and DRS adjudicates NOT OUT (Bat Involved).
    """
    # Generate trajectory with a distinct deflection spike at frame 22 (Y=17.5m, Z=0.45m)
    traj_3d = []
    for i in range(25):
        y = 10.0 + i * 0.35
        if i < 22:
            x = -0.05 + i * 0.002
            z = 0.20 + i * 0.015
        else: # Sharp deflection after glancing inside edge (dx=-0.08, dz=+0.05)
            x = -0.05 + 21 * 0.002 - (i - 21) * 0.08
            z = 0.20 + 21 * 0.015 + (i - 21) * 0.05
        traj_3d.append((x, y, z))

    traj_arr = np.array(traj_3d, dtype=np.float64)
    has_edge, frame_idx = Kalman3DSmoother.detect_bat_deflection(traj_arr, threshold_angle_deg=5.0)

    assert has_edge is True
    assert frame_idx is not None and frame_idx >= 21

    # Pass to DRS Engine with bat edge flag
    engine = DRSEngine()
    verdict = engine.evaluate_lbw(
        bounce_point=(-0.04, 11.5, 0.0),
        pad_impact_point=(traj_arr[-1, 0], traj_arr[-1, 1], traj_arr[-1, 2]),
        predicted_stump_point=(-0.01, 20.12, 0.40),
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.OUT,
        bat_edge_detected=has_edge
    )

    assert verdict.bat_edge_detected is True
    assert verdict.final_verdict == "NOT OUT"
    assert "Bat edge detected" in verdict.reasons

def test_icc_3_meter_distance_law():
    """
    Edge Case: Batsman struck >= 3.0m down the pitch from the stumps (Y_pad <= 17.12m).
    Assert 3-meter rule triggers and upholds on-field NOT OUT call for borderline wickets.
    """
    engine = DRSEngine()
    pad_impact = (0.05, 16.80, 0.45)
    bounce_pt = (0.02, 11.0, 0.0)
    stump_impact = (-0.12, 20.12, 0.50)

    verdict = engine.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.NOT_OUT
    )

    assert verdict.is_3_meter_rule_triggered is True
    assert verdict.final_verdict == "NOT OUT"
    assert "3-Meter Law" in verdict.reasons

def test_sharp_seam_spin_deviation():
    """
    Edge Case: Ball pitches and cuts sharply off the seam.
    Assert seam deviation calculator accurately measures angular shift in degrees.
    """
    v_pre = (-0.02, 35.0, -1.8)
    v_post = (2.2, 32.0, 1.2)

    angle_diff = Kalman3DSmoother.compute_seam_spin_deviation(v_pre, v_post)
    assert angle_diff > 3.5
    print(f"Seam deviation: {angle_diff:.2f} degrees")

def test_close_proximity_bounce():
    """
    Edge Case: Ball pitches within 0.25m (< 0.40m) of batsman's pad.
    Assert close proximity flag is marked in verdict.
    """
    engine = DRSEngine()
    bounce_pt = (0.0, 18.55, 0.0)
    pad_impact = (0.0, 18.80, 0.20)
    stump_impact = (0.0, 20.12, 0.35)

    verdict = engine.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.NOT_OUT
    )

    assert verdict.is_close_proximity_bounce is True
