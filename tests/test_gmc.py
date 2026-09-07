import numpy as np
import cv2
import pytest
from core.tracker import GMC

def test_gmc_identity_on_static_frames():
    gmc = GMC()
    
    # Create deterministic textured frame without np.random
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(10, 470, 30):
        for j in range(10, 630, 30):
            cv2.rectangle(frame1, (j, i), (j + 15, i + 15), (200, 200, 200), -1)

    t1 = gmc.apply(frame1)
    
    # First frame returns identity
    assert np.allclose(t1, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

def test_gmc_translation_compensation():
    gmc = GMC()
    
    # Create textured synthetic background
    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(20, 460, 40):
        for j in range(20, 620, 40):
            cv2.circle(base_frame, (j, i), 6, (255, 255, 255), -1)

    # Frame 1
    gmc.apply(base_frame)

    # Frame 2: Shifted by dx=15, dy=10 (camera pan)
    dx, dy = 15.0, 10.0
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_frame = cv2.warpAffine(base_frame, M, (640, 480))

    t2 = gmc.apply(shifted_frame)

    # Warping a point with estimated transformation
    orig_pt = (200.0, 200.0)
    warped_pt = GMC.warp_point(orig_pt, t2)

    # Should approximate the shift
    assert abs((warped_pt[0] - orig_pt[0]) - dx) < 2.0
    assert abs((warped_pt[1] - orig_pt[1]) - dy) < 2.0
