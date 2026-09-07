import pytest
import numpy as np
from config import DetectorConfig, TrackerConfig, TrajectoryConfig, VisualizerConfig, HawkEyeConfig
from core.detector import CricketBallDetector, Detection
from core.tracker import BoTSORTTracker, Track
from core.trajectory import KalmanTrajectorySmoother, Kalman3DSmoother, BounceDetector
from core.calibration import MetricPitchCalibration, PitchDimensions
from core.drs_engine import DRSEngine, BatsmanHand, OnFieldCall
from core.visualizer import HawkEyeVisualizer, Plotly3DVisualizer

@pytest.fixture
def detector():
    return CricketBallDetector(DetectorConfig())

@pytest.fixture
def tracker():
    return BoTSORTTracker(TrackerConfig())

@pytest.fixture
def smoother_2d():
    return KalmanTrajectorySmoother()

@pytest.fixture
def smoother_3d():
    return Kalman3DSmoother()

@pytest.fixture
def bounce_detector():
    return BounceDetector()

@pytest.fixture
def calibration():
    return MetricPitchCalibration()

@pytest.fixture
def drs_engine():
    return DRSEngine()

@pytest.fixture
def visualizer_2d():
    return HawkEyeVisualizer()

@pytest.fixture
def visualizer_3d():
    return Plotly3DVisualizer()

@pytest.fixture
def sample_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = (34, 139, 34)
    return frame
