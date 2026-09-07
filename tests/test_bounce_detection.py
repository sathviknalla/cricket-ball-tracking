import numpy as np
import pytest
from core.trajectory import BounceDetector
from config import TrajectoryConfig

def test_bounce_detection_v_shape():
    """
    Pass an array representing a V-shaped trajectory.
    Assert the bounce frame is detected within a 1-frame margin of error.
    """
    detector = BounceDetector()
    
    # Simulate a V-shaped trajectory bouncing at frame 20
    # Before frame 20: Ball moves downward in image (increasing Y)
    # After frame 20: Ball bounces upward (decreasing Y)
    bounce_gt_frame = 20
    n_frames = 40
    
    trajectory = np.zeros((n_frames, 2), dtype=np.float64)
    for t in range(n_frames):
        trajectory[t, 0] = 100.0 + 10.0 * t  # Constant forward motion along X
        if t <= bounce_gt_frame:
            # Falling towards ground
            trajectory[t, 1] = 200.0 + (t / float(bounce_gt_frame)) * 300.0  # From 200 to 500
        else:
            # Rebounding upwards
            prog = (t - bounce_gt_frame) / float(n_frames - bounce_gt_frame)
            trajectory[t, 1] = 500.0 - prog * 200.0  # From 500 to 300

    result = detector.detect_bounce(trajectory)

    assert result is not None, "Bounce detector failed to detect the bounce point"
    detected_frame = result["frame_idx"]
    
    # Assert within 1 frame margin of error
    frame_error = abs(detected_frame - bounce_gt_frame)
    assert frame_error <= 1, f"Detected bounce frame {detected_frame} differs from ground truth {bounce_gt_frame} by {frame_error} frames (>1 frame margin)"
    
    # Check coordinates
    assert abs(result["x"] - trajectory[bounce_gt_frame, 0]) < 15.0
    assert abs(result["y"] - 500.0) < 5.0

def test_bounce_detection_parabolic_delivery():
    """
    Tests bounce detection on realistic curved flight before and after pitch impact.
    """
    detector = BounceDetector()
    bounce_gt = 25
    n_frames = 50
    trajectory = np.zeros((n_frames, 2), dtype=np.float64)

    for t in range(n_frames):
        trajectory[t, 0] = 50.0 + 12.0 * t
        if t <= bounce_gt:
            # Parabolic descent with gravity acceleration
            u = t / float(bounce_gt)
            trajectory[t, 1] = 150.0 + 350.0 * (u ** 1.5)
        else:
            # Parabolic rise
            u = (t - bounce_gt) / float(n_frames - bounce_gt)
            trajectory[t, 1] = 500.0 - (200.0 * u - 60.0 * (u ** 2))

    result = detector.detect_bounce(trajectory)
    assert result is not None
    detected_frame = result["frame_idx"]
    assert abs(detected_frame - bounce_gt) <= 1
