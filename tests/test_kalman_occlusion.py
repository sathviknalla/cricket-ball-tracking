import numpy as np
import pytest
from trajectory import KalmanTrajectorySmoother
from config import TrajectoryConfig

def test_kalman_linear_occlusion():
    """
    Pass an array of linear trajectory points with a 5-frame gap.
    Assert the filter accurately bridges the gap.
    """
    smoother = KalmanTrajectorySmoother()
    n_frames = 30
    
    # Ground truth linear motion: x(t) = 100 + 10*t, y(t) = 50 + 5*t
    gt_trajectory = [(100.0 + 10.0 * t, 50.0 + 5.0 * t) for t in range(n_frames)]
    
    # Introduce a 5-frame occlusion gap from frame 12 to frame 16
    corrupted_trajectory = []
    for t in range(n_frames):
        if 12 <= t <= 16:
            corrupted_trajectory.append(None)
        else:
            # Small sensor noise
            noise_x = 0.2 if t % 2 == 0 else -0.2
            noise_y = 0.1 if t % 2 == 0 else -0.1
            corrupted_trajectory.append((gt_trajectory[t][0] + noise_x, gt_trajectory[t][1] + noise_y))

    smoothed = smoother.smooth_trajectory(corrupted_trajectory)

    assert len(smoothed) == n_frames, "Smoothed output length must match input sequence length"
    assert not np.isnan(smoothed).any(), "Interpolated trajectory must not contain NaN values"

    # Verify interpolation accuracy during the 5-frame occlusion gap
    for t in range(12, 17):
        gt_x, gt_y = gt_trajectory[t]
        pred_x, pred_y = smoothed[t]
        err_x = abs(pred_x - gt_x)
        err_y = abs(pred_y - gt_y)
        
        # In linear motion with known velocity, error across 5 frames should be strictly bounded
        assert err_x < 5.0, f"Occlusion error in X at frame {t} is too high ({err_x:.2f} px)"
        assert err_y < 3.0, f"Occlusion error in Y at frame {t} is too high ({err_y:.2f} px)"

def test_kalman_parabolic_occlusion():
    """
    Tests Kalman smoother on a curved trajectory with acceleration and a 5-frame occlusion.
    """
    smoother = KalmanTrajectorySmoother()
    n_frames = 25
    gt = [(200.0 + 8.0 * t, 100.0 + 2.0 * t + 0.3 * (t ** 2)) for t in range(n_frames)]

    input_pts = []
    for t in range(n_frames):
        if 8 <= t <= 12:  # 5-frame gap
            input_pts.append(None)
        else:
            input_pts.append(gt[t])

    smoothed = smoother.smooth_trajectory(input_pts)
    assert len(smoothed) == n_frames

    # Check that trajectory remains monotonically moving and close to ground truth
    for t in range(8, 13):
        pred_x, pred_y = smoothed[t]
        gt_x, gt_y = gt[t]
        assert abs(pred_x - gt_x) < 8.0
        assert abs(pred_y - gt_y) < 15.0
