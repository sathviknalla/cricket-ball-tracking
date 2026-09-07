import numpy as np
import pytest
from core.calibration import MetricPitchCalibration, PitchDimensions

def test_calibration_4point_metric_projection():
    """
    Verify known 4-point pitch coordinates project to accurate metric dimensions
    within 2% numerical tolerance.
    """
    calib = MetricPitchCalibration()
    dims = PitchDimensions()

    # Ground truth reference points on pitch plane (Z=0) in meters
    gt_pts = [
        (-dims.return_crease_half_width, dims.bowling_popping_crease),  # (-1.32, 1.22)
        (dims.return_crease_half_width, dims.bowling_popping_crease),   # (1.32, 1.22)
        (dims.return_crease_half_width, dims.batting_popping_crease),   # (1.32, 18.90)
        (-dims.return_crease_half_width, dims.batting_popping_crease),  # (-1.32, 18.90)
        (0.0, 0.0),                                                     # Bowler Stumps
        (0.0, dims.length)                                              # Batsman Stumps (20.12m)
    ]

    for gt_x, gt_y in gt_pts:
        # 1. Project ground (X, Y) to pixel (u, v)
        u, v = calib.ground_to_pixel(gt_x, gt_y)
        assert u > 0 and v > 0, f"Projected pixel out of bounds: ({u}, {v})"

        # 2. Back-project pixel (u, v) to ground (X, Y)
        reproj_x, reproj_y = calib.pixel_to_ground(u, v)

        # 3. Assert error is well within 2% tolerance of pitch length
        err_x = abs(reproj_x - gt_x)
        err_y = abs(reproj_y - gt_y)

        assert err_x < 0.05, f"X projection error {err_x:.4f}m exceeds tolerance for point ({gt_x}, {gt_y})"
        assert err_y < 0.10, f"Y projection error {err_y:.4f}m exceeds tolerance for point ({gt_x}, {gt_y})"

        # Relative percentage error
        if abs(gt_y) > 1.0:
            rel_err_y = (err_y / gt_y) * 100.0
            assert rel_err_y < 2.0, f"Relative Y error {rel_err_y:.2f}% exceeds 2% tolerance"
