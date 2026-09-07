import numpy as np
import cv2
import pytest
from core.tracker import SceneChangeDetector, BoTSORTTracker, Track, TrackState
from core.detector import Detection

def test_scene_change_detector_direct():
    """
    Pass two wildly different frame representations.
    Assert the scene-cut logic triggers.
    """
    detector = SceneChangeDetector(threshold=0.60)

    # Frame 1: Bright Green Grass (Pitch broadcast view)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame1[:, :, 1] = 220  # Strong Green channel
    frame1[:, :, 0] = 30
    frame1[:, :, 2] = 30

    # Frame 2: Continuous next frame with slight camera movement (no cut)
    frame2 = frame1.copy()
    frame2[100:150, 100:150] = 255  # Small local change

    # Frame 3: Drastic scene change (Dark Blue / Red crowd or studio replay graphic)
    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame3[:, :, 0] = 230  # Blue
    frame3[:, :, 2] = 200  # Red

    # First frame initializes reference
    cut1 = detector.detect_cut(frame1)
    assert not cut1, "First frame should not trigger scene cut"

    # Second continuous frame should NOT trigger scene cut
    cut2 = detector.detect_cut(frame2)
    assert not cut2, "Continuous frame should not trigger scene cut"

    # Third radically different frame MUST trigger scene cut
    cut3 = detector.detect_cut(frame3)
    assert cut3, "Drastic broadcast cut must trigger scene cut detection"

def test_scene_change_resets_tracker():
    """
    Verify that when a scene cut occurs during tracking,
    the tracker resets its active tracks to prevent drawing lines across cuts.
    """
    tracker = BoTSORTTracker()
    
    # Frame 1: Grass frame with a ball detection
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame1[:, :] = (30, 200, 30)
    det1 = Detection(bbox=(100, 100, 115, 115), confidence=0.90, class_id=32)
    tracker.update(frame1, [det1])

    # Frame 2: Grass frame moving ball (confirm track)
    frame2 = frame1.copy()
    det2 = Detection(bbox=(110, 110, 125, 125), confidence=0.92, class_id=32)
    tracks_f2 = tracker.update(frame2, [det2])
    assert len(tracks_f2) > 0
    original_id = tracks_f2[0].track_id

    # Frame 3: Broadcast instant camera cut to crowd/replay
    frame_cut = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_cut[:, :] = (210, 20, 200)  # Magenta graphic
    det3 = Detection(bbox=(300, 300, 315, 315), confidence=0.88, class_id=32)

    tracks_f3 = tracker.update(frame_cut, [det3])
    
    # Assert tracker was reset on scene cut: previous track should be gone, new track created
    if len(tracks_f3) > 0:
        new_id = tracks_f3[0].track_id
        assert new_id != original_id, "Track ID must reset across camera scene cut"
