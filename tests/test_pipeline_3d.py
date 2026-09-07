import numpy as np
import pytest
from core.calibration import MetricPitchCalibration
from core.trajectory import Kalman3DSmoother
from core.drs_engine import DRSEngine, BatsmanHand, OnFieldCall
from core.visualizer import Plotly3DVisualizer
from training.dataset import SyntheticCricketBallDataset

def test_full_3d_pipeline_synthetic_delivery():
    """
    End-to-end integration test of 3D trajectory reconstruction,
    stump projection, DRS verdict evaluation, and 3D Plotly rendering.
    """
    # 1. 3D Trajectory setup
    calib = MetricPitchCalibration()
    drs = DRSEngine()
    vis = Plotly3DVisualizer()

    # Pre-bounce
    pre_traj = np.array([
        [0.0, 1.5, 2.15],
        [-0.02, 6.0, 1.2],
        [-0.04, 11.5, 0.0]
    ])
    bounce_pt = (-0.04, 11.5, 0.0)

    # Post-bounce to pad
    post_traj = np.array([
        [-0.04, 11.5, 0.0],
        [-0.03, 15.0, 0.35],
        [-0.02, 18.8, 0.45]
    ])
    pad_impact = (-0.02, 18.8, 0.45)

    # Post-impact projection
    proj_traj, stump_pt = Kalman3DSmoother.project_to_stumps(
        impact_point=pad_impact,
        velocity=(0.02 * 35.0, 35.0, -0.2 * 35.0),
        target_y=20.12
    )

    # Adjudicate
    verdict = drs.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_pt,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.NOT_OUT
    )

    assert verdict.final_verdict in ("OUT", "NOT OUT")
    assert stump_pt[1] == pytest.approx(20.12, abs=0.01)

    # Plotly 3D Figure generation
    fig = vis.build_3d_pitch_figure(
        pre_bounce_traj=pre_traj,
        post_bounce_traj=post_traj,
        projected_traj=proj_traj,
        bounce_point=bounce_pt,
        impact_point=pad_impact,
        stump_impact=stump_pt,
        verdict=verdict
    )
    assert fig is not None
    assert len(fig.data) > 4

def test_synthetic_dataset_generator():
    ds = SyntheticCricketBallDataset(output_dir="dataset_test")
    frame, bbox = ds.generate_synthetic_frame(width=640, height=480)
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert len(bbox) == 4
    # All YOLO coords between 0 and 1
    assert all(0.0 <= c <= 1.0 for c in bbox)
