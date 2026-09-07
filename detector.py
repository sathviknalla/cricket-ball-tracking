import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from config import DetectorConfig

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    feature: Optional[np.ndarray] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        if h <= 1e-6:
            return 1.0
        return self.width / h

class CricketBallDetector:
    """
    YOLOv8-based cricket ball detector with motion-blur adaptation
    and false-positive heuristic filtering.
    """
    def __init__(self, config: Optional[DetectorConfig] = None, model=None):
        self.config = config or DetectorConfig()
        self.model = model
        self._model_loaded = False
        if model is not None:
            self._model_loaded = True

    def _lazy_load_model(self):
        if not self._model_loaded:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.config.weights_path)
                self._model_loaded = True
            except Exception as e:
                # Fallback / graceful initialization
                self.model = None
                self._model_loaded = False

    def is_valid_candidate(
        self,
        det: Detection,
        frame_shape: Tuple[int, int],
        prev_ball_pos: Optional[Tuple[float, float]] = None
    ) -> bool:
        """
        Filters false positives based on size, aspect ratio, motion streak heuristics,
        and kinematic proximity to previous ball position.
        """
        # 1. Class ID check
        if det.class_id not in self.config.target_class_ids:
            return False

        # 2. Geometric bounds (size/area/aspect ratio)
        w, h = det.width, det.height
        area = det.area
        ar = det.aspect_ratio

        if area < self.config.min_area or area > self.config.max_area:
            return False

        avg_radius = (w + h) / 4.0
        if avg_radius < self.config.min_radius or avg_radius > self.config.max_radius:
            return False

        if ar < self.config.min_aspect_ratio or ar > self.config.max_aspect_ratio:
            return False

        # 3. Confidence & Motion Blur check
        # For severe motion blur (>140 km/h), the ball elongates into a streak, lowering YOLO confidence.
        is_streak = (ar > 1.3 or ar < 0.77)
        if det.confidence >= self.config.base_conf_threshold:
            conf_ok = True
        elif det.confidence >= self.config.motion_blur_conf_threshold:
            # Allow lower confidence if it displays motion streak geometry or is close to predicted pos
            conf_ok = is_streak or (prev_ball_pos is not None)
        else:
            conf_ok = False

        if not conf_ok:
            return False

        # 4. Kinematic distance gate (reject teleporting false positives)
        if prev_ball_pos is not None:
            cx, cy = det.center
            dist = float(np.hypot(cx - prev_ball_pos[0], cy - prev_ball_pos[1]))
            if dist > self.config.max_speed_pixels_per_frame:
                return False

        return True

    def filter_candidates(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
        prev_ball_pos: Optional[Tuple[float, float]] = None
    ) -> List[Detection]:
        """Applies heuristic filters to a list of raw detections."""
        valid = []
        for det in detections:
            if self.is_valid_candidate(det, frame_shape, prev_ball_pos):
                valid.append(det)
        
        # Sort by confidence descending
        valid.sort(key=lambda d: d.confidence, reverse=True)
        return valid

    def detect(
        self,
        frame: np.ndarray,
        prev_ball_pos: Optional[Tuple[float, float]] = None
    ) -> List[Detection]:
        """
        Runs YOLO detection on a single frame and filters results.
        """
        if frame is None or frame.size == 0:
            return []

        self._lazy_load_model()
        if self.model is None:
            return []

        frame_shape = frame.shape[:2]
        # Run inference with lower motion-blur threshold to capture streak candidates
        results = self.model(frame, conf=self.config.motion_blur_conf_threshold, verbose=False)
        raw_detections: List[Detection] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                raw_detections.append(Detection(
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=conf,
                    class_id=cls_id
                ))

        return self.filter_candidates(raw_detections, frame_shape, prev_ball_pos)
