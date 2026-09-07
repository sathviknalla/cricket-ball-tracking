import numpy as np
import os
import pytest
from visualizer import HawkEyeVisualizer
from config import VisualizerConfig

def test_visualizer_draw_trajectory():
    vis = HawkEyeVisualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    history = [(100.0 + 5.0 * i, 150.0 + 3.0 * i) for i in range(15)]
    bounce_pt = (150.0, 180.0)

    out_frame = vis.draw_trajectory(
        frame=frame,
        history=history,
        bounce_point=bounce_pt,
        track_id=1,
        current_speed_kmh=145.0
    )

    assert out_frame is not None
    assert out_frame.shape == frame.shape
    # Check that pixels were drawn
    assert not np.all(out_frame == 0)

def test_visualizer_generate_pitch_map(tmp_path):
    vis = HawkEyeVisualizer()
    traj = np.array([(100.0 + i * 5, 200.0 + i * 10) for i in range(20)])
    bounce_info = {'frame_idx': 10, 'x': 150.0, 'y': 300.0}

    map_path = str(tmp_path / "test_pitch_map.png")
    fig = vis.generate_pitch_map(trajectory=traj, bounce_point=bounce_info, save_path=map_path)

    assert os.path.exists(map_path)
    assert os.path.getsize(map_path) > 1000
