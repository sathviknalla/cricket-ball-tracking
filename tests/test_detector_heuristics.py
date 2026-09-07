import numpy as np
import pytest
from detector import CricketBallDetector, Detection
from config import DetectorConfig

def test_detector_valid_ball():
    detector = CricketBallDetector()
    frame_shape = (720, 1280)
    
    # Valid ball detection: radius ~ 8px, aspect ratio ~ 1.0, confidence 0.85
    valid_det = Detection(bbox=(500, 300, 516, 316), confidence=0.85, class_id=32)
    assert detector.is_valid_candidate(valid_det, frame_shape) is True

def test_detector_rejects_oversized_shoe_or_helmet():
    detector = CricketBallDetector()
    frame_shape = (720, 1280)
    
    # Huge white object (e.g. batsman white shoe or helmet, area = 120x90 = 10800 > max_area 6500)
    oversized_det = Detection(bbox=(400, 300, 520, 390), confidence=0.90, class_id=32)
    assert detector.is_valid_candidate(oversized_det, frame_shape) is False

def test_detector_motion_blur_streak_acceptance():
    detector = CricketBallDetector()
    frame_shape = (720, 1280)
    
    # Motion streak: elongated box (w=32, h=10 -> aspect ratio 3.2), low confidence 0.22 (below base 0.30, but above blur 0.15)
    motion_streak_det = Detection(bbox=(600, 200, 632, 210), confidence=0.22, class_id=32)
    assert detector.is_valid_candidate(motion_streak_det, frame_shape) is True
    
    # Low confidence round object with no prior position (should be rejected as weak noise)
    weak_noise_det = Detection(bbox=(600, 200, 610, 210), confidence=0.22, class_id=32)
    assert detector.is_valid_candidate(weak_noise_det, frame_shape) is False

def test_detector_kinematic_distance_gating():
    detector = CricketBallDetector()
    frame_shape = (720, 1280)
    prev_pos = (500.0, 300.0)
    
    # Detection reasonably close (moving at ~40px per frame)
    nearby_det = Detection(bbox=(530, 325, 546, 341), confidence=0.75, class_id=32)
    assert detector.is_valid_candidate(nearby_det, frame_shape, prev_ball_pos=prev_pos) is True

    # Teleporting false positive (e.g. white bird or crowd artifact 400px away)
    teleport_det = Detection(bbox=(900, 600, 916, 616), confidence=0.75, class_id=32)
    assert detector.is_valid_candidate(teleport_det, frame_shape, prev_ball_pos=prev_pos) is False
