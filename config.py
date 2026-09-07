from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class DetectorConfig:
    weights_path: str = "yolov8n.pt"
    base_conf_threshold: float = 0.30
    motion_blur_conf_threshold: float = 0.15
    sports_ball_class_id: int = 32  # COCO class for sports ball (or 0 for custom trained ball detector)
    target_class_ids: List[int] = field(default_factory=lambda: [32, 0])
    
    # Heuristic filtering
    min_radius: float = 2.0
    max_radius: float = 45.0
    max_aspect_ratio: float = 4.5  # Accommodates motion streak elongation
    min_aspect_ratio: float = 0.22
    max_speed_pixels_per_frame: float = 160.0
    max_area: float = 6500.0
    min_area: float = 12.0

@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.15
    new_track_thresh: float = 0.40
    track_buffer: int = 30
    match_thresh: float = 0.70
    gmc_method: str = "sparse_optical_flow"  # or 'orb', 'sift'
    gmc_max_features: int = 400
    scene_change_threshold: float = 0.65  # Histogram distance threshold

@dataclass
class TrajectoryConfig:
    process_noise_pos: float = 1.0
    process_noise_vel: float = 5.0
    process_noise_acc: float = 10.0
    measurement_noise: float = 4.0
    gravity: float = 9.81  # Sub-prior for vertical axis acceleration
    max_occlusion_frames: int = 15
    bounce_peak_prominence: float = 3.0
    bounce_peak_distance: int = 10

@dataclass
class VisualizerConfig:
    trajectory_max_length: int = 60
    tail_fading: bool = True
    tail_glow: bool = True
    tail_start_color: Tuple[int, int, int] = (0, 165, 255)   # BGR: Orange
    tail_end_color: Tuple[int, int, int] = (255, 255, 0)     # BGR: Cyan
    bounce_color: Tuple[int, int, int] = (0, 0, 255)         # BGR: Red
    pitch_width_meters: float = 3.05
    pitch_length_meters: float = 20.12

@dataclass
class HawkEyeConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    output_dir: str = "output"
