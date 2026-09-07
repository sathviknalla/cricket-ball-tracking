import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import linear_sum_assignment
from config import TrackerConfig
from core.detector import Detection

class TrackState(Enum):
    Tentative = 1
    Confirmed = 2
    Lost = 3
    Deleted = 4

class GMC:
    """
    Global Motion Compensation (GMC) for aggressive camera panning and zooming.
    Computes affine transformation between consecutive video frames using sparse optical flow.
    """
    def __init__(self, method: str = "sparse_optical_flow", max_features: int = 400):
        self.method = method
        self.max_features = max_features
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[np.ndarray] = None

    def reset(self):
        self.prev_frame_gray = None
        self.prev_keypoints = None

    def apply(self, frame: Optional[np.ndarray]) -> np.ndarray:
        """
        Calculates 2x3 affine motion matrix compensating for camera pan/tilt/zoom.
        Returns 2x3 identity matrix if no motion or first frame.
        """
        identity_transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if frame is None or frame.size == 0:
            return identity_transform

        if len(frame.shape) == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame.copy()

        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_gray
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            return identity_transform

        if self.prev_keypoints is None or len(self.prev_keypoints) < 6:
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                self.prev_frame_gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
            )
            if self.prev_keypoints is None or len(self.prev_keypoints) < 6:
                self.prev_frame_gray = curr_gray
                return identity_transform

        # Optical flow tracking
        curr_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame_gray, curr_gray, self.prev_keypoints, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        valid_prev = self.prev_keypoints[status == 1]
        valid_curr = curr_keypoints[status == 1]

        transform = identity_transform
        if len(valid_prev) >= 6:
            # Estimate partial affine (translation + rotation + uniform scale)
            affine_mat, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_curr, method=cv2.RANSAC)
            if affine_mat is not None:
                transform = affine_mat

        self.prev_frame_gray = curr_gray
        self.prev_keypoints = cv2.goodFeaturesToTrack(
            curr_gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
        )
        return transform

    @staticmethod
    def warp_point(point: Tuple[float, float], transform: np.ndarray) -> Tuple[float, float]:
        """Warps a 2D point [x, y] using 2x3 affine transformation."""
        x, y = point
        vec = np.array([x, y, 1.0], dtype=np.float32)
        warped = transform @ vec
        return float(warped[0]), float(warped[1])

    @staticmethod
    def warp_bbox(bbox: Tuple[float, float, float, float], transform: np.ndarray) -> Tuple[float, float, float, float]:
        """Warps a bounding box [x1, y1, x2, y2] using affine matrix."""
        x1, y1, x2, y2 = bbox
        pts = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x1, y2, 1.0], [x2, y2, 1.0]], dtype=np.float32).T
        warped_pts = (transform @ pts).T
        min_x, min_y = float(np.min(warped_pts[:, 0])), float(np.min(warped_pts[:, 1]))
        max_x, max_y = float(np.max(warped_pts[:, 0])), float(np.max(warped_pts[:, 1]))
        return (min_x, min_y, max_x, max_y)

class SceneChangeDetector:
    """
    Detects instant camera cuts / scene switches in broadcast footage
    using Bhattacharyya distance on color histograms.
    """
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.prev_hist: Optional[np.ndarray] = None

    def reset(self):
        self.prev_hist = None

    def compute_hist(self, frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 3:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        else:
            hist = cv2.calcHist([frame], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def detect_cut(self, frame: Optional[np.ndarray]) -> bool:
        if frame is None or frame.size == 0:
            return False

        curr_hist = self.compute_hist(frame)
        if self.prev_hist is None:
            self.prev_hist = curr_hist
            return False

        dist = cv2.compareHist(self.prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        self.prev_hist = curr_hist
        return bool(dist > self.threshold)

class Track:
    """
    Individual ball track representation with motion model and history.
    """
    _next_id: int = 1

    def __init__(self, initial_det: Detection, frame_idx: int):
        self.track_id: int = Track._next_id
        Track._next_id += 1

        self.state: TrackState = TrackState.Tentative
        self.bbox: Tuple[float, float, float, float] = initial_det.bbox
        self.center: Tuple[float, float] = initial_det.center
        self.confidence: float = initial_det.confidence
        self.hits: int = 1
        self.age: int = 1
        self.time_since_update: int = 0
        self.history: List[Tuple[float, float]] = [self.center]
        self.frame_indices: List[int] = [frame_idx]
        self.velocity: Tuple[float, float] = (0.0, 0.0)

    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 1

    def apply_gmc(self, transform: np.ndarray):
        """Compensates internal position and history with camera motion matrix."""
        if np.allclose(transform, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]):
            return
        self.center = GMC.warp_point(self.center, transform)
        self.bbox = GMC.warp_bbox(self.bbox, transform)

    def predict(self, transform: Optional[np.ndarray] = None):
        """Predicts position using kinematic velocity and GMC."""
        if transform is not None:
            self.apply_gmc(transform)

        # Extrapolate center by current velocity
        cx, cy = self.center
        vx, vy = self.velocity
        new_cx = cx + vx
        new_cy = cy + vy

        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.bbox = (new_cx - w / 2, new_cy - h / 2, new_cx + w / 2, new_cy + h / 2)
        self.center = (new_cx, new_cy)

        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection, frame_idx: int):
        """Updates track with matched detection."""
        old_cx, old_cy = self.center
        new_cx, new_cy = det.center

        # Update velocity with exponential moving average
        inst_vx = new_cx - old_cx
        inst_vy = new_cy - old_cy
        if self.hits == 1:
            self.velocity = (inst_vx, inst_vy)
        else:
            self.velocity = (0.7 * self.velocity[0] + 0.3 * inst_vx,
                             0.7 * self.velocity[1] + 0.3 * inst_vy)

        self.bbox = det.bbox
        self.center = (new_cx, new_cy)
        self.confidence = det.confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.center)
        self.frame_indices.append(frame_idx)

        if self.state == TrackState.Tentative and self.hits >= 2:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Marks frame where detection was not matched."""
        if self.time_since_update > 0:
            # Maintain history projection during occlusion
            self.history.append(self.center)
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > 30:
            self.state = TrackState.Deleted
        else:
            self.state = TrackState.Lost

class BoTSORTTracker:
    """
    BoT-SORT Tracker enhanced with Global Motion Compensation (GMC)
    and broadcast scene cut detection.
    """
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.tracks: List[Track] = []
        self.gmc = GMC(method=self.config.gmc_method, max_features=self.config.gmc_max_features)
        self.scene_detector = SceneChangeDetector(threshold=self.config.scene_change_threshold)
        self.frame_idx: int = 0

    def reset(self):
        """Clears all active tracks and detector state (e.g. on camera cut)."""
        self.tracks.clear()
        self.gmc.reset()
        self.scene_detector.reset()
        Track.reset_id_counter()

    def _compute_cost_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """
        Computes motion and spatial distance cost matrix between tracks and detections.
        """
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            tx, ty = t.center
            for j, d in enumerate(detections):
                dx, dy = d.center
                # Euclidean distance
                dist = np.hypot(tx - dx, ty - dy)

                # Velocity alignment bonus (if track has known velocity)
                vx, vy = t.velocity
                if np.hypot(vx, vy) > 2.0:
                    pred_dx = dx - tx
                    pred_dy = dy - ty
                    norm_pred = np.hypot(pred_dx, pred_dy) + 1e-6
                    norm_v = np.hypot(vx, vy) + 1e-6
                    cos_sim = (vx * pred_dx + vy * pred_dy) / (norm_v * norm_pred)
                    if cos_sim < 0.0:  # Moving in opposite direction
                        dist += getattr(self.config, 'velocity_alignment_penalty', 50.0)

                cost_matrix[i, j] = dist
        return cost_matrix

    def update(self, frame: Optional[np.ndarray], detections: List[Detection]) -> List[Track]:
        """
        Updates tracker on incoming video frame and detections.
        """
        self.frame_idx += 1

        # 1. Check for broadcast scene cut / camera switch
        if frame is not None and self.scene_detector.detect_cut(frame):
            self.reset()
            # Start fresh from new scene
            for det in detections:
                if det.confidence >= self.config.new_track_thresh:
                    self.tracks.append(Track(det, self.frame_idx))
            return [t for t in self.tracks if t.state == TrackState.Confirmed]

        # 2. Apply Global Motion Compensation (GMC)
        transform = self.gmc.apply(frame) if frame is not None else np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        # 3. Predict state of all current tracks
        for t in self.tracks:
            t.predict(transform)

        # Split detections by high/low confidence (BoT-SORT / ByteTrack style)
        high_dets = [d for d in detections if d.confidence >= self.config.track_high_thresh]
        low_dets = [d for d in detections if self.config.track_low_thresh <= d.confidence < self.config.track_high_thresh]

        # 4. First Association (High Confidence detections via Hungarian Algorithm)
        active_tracks = [t for t in self.tracks if t.state in (TrackState.Confirmed, TrackState.Tentative)]
        matched_tracks = set()
        matched_dets = set()

        if active_tracks and high_dets:
            cost_matrix = self._compute_cost_matrix(active_tracks, high_dets)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            high_gate = getattr(self.config, 'high_conf_gating_dist', 120.0)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= high_gate:
                    active_tracks[r].update(high_dets[c], self.frame_idx)
                    matched_tracks.add(active_tracks[r])
                    matched_dets.add(c)

        # 5. Second Association (Low Confidence / Motion Blur streak detections via Hungarian Algorithm)
        unmatched_tracks = [t for t in active_tracks if t not in matched_tracks]
        if unmatched_tracks and low_dets:
            low_cost_mat = self._compute_cost_matrix(unmatched_tracks, low_dets)
            row_ind, col_ind = linear_sum_assignment(low_cost_mat)
            low_gate = getattr(self.config, 'low_conf_gating_dist', 90.0)
            for r, c in zip(row_ind, col_ind):
                if low_cost_mat[r, c] <= low_gate:
                    unmatched_tracks[r].update(low_dets[c], self.frame_idx)
                    matched_tracks.add(unmatched_tracks[r])

        # 6. Mark unmatched tracks as missed
        for t in self.tracks:
            if t not in matched_tracks:
                t.mark_missed()

        # 7. Initialize new tracks from unmatched high-confidence detections
        unmatched_high_dets = [d for idx, d in enumerate(high_dets) if idx not in matched_dets]
        for d in unmatched_high_dets:
            if d.confidence >= self.config.new_track_thresh:
                self.tracks.append(Track(d, self.frame_idx))

        # 8. Clean up deleted tracks and track buffer expiration
        self.tracks = [t for t in self.tracks if t.state != TrackState.Deleted and t.time_since_update <= self.config.track_buffer]

        # Return confirmed tracks or tracks with active hits
        return [t for t in self.tracks if t.state == TrackState.Confirmed or t.hits >= 2]
