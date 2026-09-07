import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from config import TrajectoryConfig

# Robust import with native fallback if scipy binary is restricted by App Control policy
try:
    from scipy.signal import find_peaks as scipy_find_peaks
    from scipy.signal import savgol_filter as scipy_savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def native_find_peaks(
    x: np.ndarray,
    prominence: Optional[float] = None,
    distance: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Pure NumPy implementation of find_peaks compatible with scipy.signal.find_peaks.
    Detects local maxima and calculates topological peak prominence and distance gating.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return np.array([], dtype=int), {"prominences": np.array([])}

    # Identify local maxima
    raw_peaks = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            raw_peaks.append(i)

    if not raw_peaks:
        return np.array([], dtype=int), {"prominences": np.array([])}

    # Calculate prominence for each candidate peak
    valid_peaks = []
    prominences = []
    for p in raw_peaks:
        pv = x[p]
        # Find left base (lowest valley before a higher peak or array start)
        left_min = pv
        for l in range(p - 1, -1, -1):
            if x[l] > pv:
                break
            if x[l] < left_min:
                left_min = x[l]

        # Find right base (lowest valley before a higher peak or array end)
        right_min = pv
        for r in range(p + 1, len(x)):
            if x[r] > pv:
                break
            if x[r] < right_min:
                right_min = x[r]

        prom = pv - max(left_min, right_min)
        if prominence is None or prom >= prominence:
            valid_peaks.append(p)
            prominences.append(prom)

    peaks = np.array(valid_peaks, dtype=int)
    prominences = np.array(prominences, dtype=np.float64)

    # Apply distance constraint (greedy selection of highest peaks)
    if distance is not None and len(peaks) > 1:
        keep = []
        sorted_order = np.argsort(-x[peaks])
        for s_idx in sorted_order:
            p = peaks[s_idx]
            if all(abs(p - k) >= distance for k in keep):
                keep.append(p)
        peaks = np.array(sorted(keep), dtype=int)

    return peaks, {"prominences": prominences}

def find_peaks(x, prominence=None, distance=None):
    if _HAS_SCIPY:
        try:
            return scipy_find_peaks(x, prominence=prominence, distance=distance)
        except Exception:
            return native_find_peaks(x, prominence=prominence, distance=distance)
    return native_find_peaks(x, prominence=prominence, distance=distance)

def native_savgol_filter(x: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Moving polynomial filter smoothing trajectory coords.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window_length or window_length < 3:
        return x
    
    half_w = window_length // 2
    smoothed = x.copy()
    for i in range(half_w, len(x) - half_w):
        window = x[i - half_w : i + half_w + 1]
        t = np.arange(-half_w, half_w + 1)
        poly = np.polyfit(t, window, deg=min(polyorder, len(window) - 1))
        smoothed[i] = np.polyval(poly, 0)
    return smoothed

def savgol_filter(x, window_length, polyorder):
    if _HAS_SCIPY:
        try:
            return scipy_savgol_filter(x, window_length=window_length, polyorder=polyorder)
        except Exception:
            return native_savgol_filter(x, window_length=window_length, polyorder=polyorder)
    return native_savgol_filter(x, window_length=window_length, polyorder=polyorder)

class KalmanTrajectorySmoother:
    """
    6-State Kalman Filter [x, y, vx, vy, ax, ay] for ball trajectory smoothing
    and occlusion bridging.
    """
    def __init__(self, dt: float = 1.0, config: Optional[TrajectoryConfig] = None):
        self.dt = dt
        self.config = config or TrajectoryConfig()

        # State: [x, y, vx, vy, ax, ay]^T
        self.state_dim = 6
        self.meas_dim = 2
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)

        # Transition Matrix F
        self.F = np.eye(self.state_dim, dtype=np.float64)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[0, 4] = 0.5 * dt * dt
        self.F[1, 5] = 0.5 * dt * dt
        self.F[2, 4] = dt
        self.F[3, 5] = dt

        # Measurement Matrix H (we measure x, y)
        self.H = np.zeros((self.meas_dim, self.state_dim), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Covariances
        self.P = np.eye(self.state_dim, dtype=np.float64) * 100.0
        self.Q = np.eye(self.state_dim, dtype=np.float64)
        self.Q[0, 0] = self.config.process_noise_pos
        self.Q[1, 1] = self.config.process_noise_pos
        self.Q[2, 2] = self.config.process_noise_vel
        self.Q[3, 3] = self.config.process_noise_vel
        self.Q[4, 4] = self.config.process_noise_acc
        self.Q[5, 5] = self.config.process_noise_acc

        self.R = np.eye(self.meas_dim, dtype=np.float64) * self.config.measurement_noise

        self.initialized = False

    def init_state(self, initial_pos: Tuple[float, float], initial_vel: Tuple[float, float] = (0.0, 0.0)):
        """Initializes Kalman state."""
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)
        self.x[0, 0] = initial_pos[0]
        self.x[1, 0] = initial_pos[1]
        self.x[2, 0] = initial_vel[0]
        self.x[3, 0] = initial_vel[1]
        self.x[5, 0] = 0.0  # Vertical acceleration prior
        self.P = np.eye(self.state_dim, dtype=np.float64) * 10.0
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        """Predicts state forward by 1 time step."""
        if not self.initialized:
            return (0.0, 0.0)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Updates state with a new measurement (x, y)."""
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)
        if not self.initialized:
            self.init_state(measurement)
            return measurement

        # Prediction step first
        self.predict()

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        y_residual = z - (self.H @ self.x)
        self.x = self.x + K @ y_residual
        I = np.eye(self.state_dim, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def smooth_trajectory(
        self,
        points: List[Optional[Tuple[float, float]]]
    ) -> np.ndarray:
        """
        Smooths a trajectory sequence and fills in missing/occluded frames (where point is None).
        Returns an (N, 2) numpy array of continuous coordinates.
        """
        if not points:
            return np.empty((0, 2), dtype=np.float64)

        # Find first valid point to initialize
        first_valid_idx = None
        for i, pt in enumerate(points):
            if pt is not None:
                first_valid_idx = i
                break

        if first_valid_idx is None:
            return np.zeros((len(points), 2), dtype=np.float64)

        # Estimate initial velocity if possible
        init_vel = (0.0, 0.0)
        for i in range(first_valid_idx + 1, len(points)):
            if points[i] is not None:
                dt_span = max(1.0, float(i - first_valid_idx))
                init_vel = (
                    (points[i][0] - points[first_valid_idx][0]) / dt_span,
                    (points[i][1] - points[first_valid_idx][1]) / dt_span
                )
                break

        self.init_state(points[first_valid_idx], init_vel)

        filtered_states = []

        # Backfill any leading None frames
        leading_pred = points[first_valid_idx]
        for i in range(first_valid_idx):
            filtered_states.append(np.array(leading_pred, dtype=np.float64))

        # Forward filtering pass
        for i in range(first_valid_idx, len(points)):
            pt = points[i]
            if pt is not None:
                smooth_pt = self.update(pt)
            else:
                smooth_pt = self.predict()
            filtered_states.append(np.array(smooth_pt, dtype=np.float64))

        smoothed = np.array(filtered_states, dtype=np.float64)

        # Apply polynomial smoothing refinement if trajectory is sufficiently long
        if len(smoothed) >= 7:
            try:
                window_len = min(7, len(smoothed) if len(smoothed) % 2 == 1 else len(smoothed) - 1)
                smoothed[:, 0] = savgol_filter(smoothed[:, 0], window_length=window_len, polyorder=2)
                smoothed[:, 1] = savgol_filter(smoothed[:, 1], window_length=window_len, polyorder=2)
            except Exception:
                pass

        return smoothed

class BounceDetector:
    """
    Detects the bounce frame and pitch impact coordinates from ball trajectory.
    Uses vertical velocity inflection analysis and peak detection.
    """
    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()

    def detect_bounce(
        self,
        trajectory: np.ndarray,
        fps: float = 30.0
    ) -> Optional[Dict[str, float]]:
        """
        Finds the bounce frame index and (x, y) coordinates from an (N, 2) trajectory array.
        Returns dict with keys: 'frame_idx', 'x', 'y', 'confidence', or None if no bounce detected.
        """
        if trajectory is None or len(trajectory) < 5:
            return None

        y_coords = trajectory[:, 1]
        x_coords = trajectory[:, 0]
        n_pts = len(y_coords)

        # Calculate vertical velocity dy/dt
        vy = np.gradient(y_coords)
        # Calculate vertical acceleration d(vy)/dt = d^2y/dt^2
        ay = np.gradient(vy)

        # In image coordinates, Y increases downwards.
        # As ball falls towards pitch, Y increases (vy > 0).
        # At bounce, Y reaches local maximum, and vy sharply reverses / decreases.
        
        # Method 1: Direct peak in Y coordinates
        peaks, properties = find_peaks(
            y_coords,
            prominence=self.config.bounce_peak_prominence,
            distance=self.config.bounce_peak_distance
        )

        if len(peaks) == 0:
            # Method 2: Peak in deceleration (-ay) where downward speed rapidly stops
            neg_ay = -ay
            peaks, properties = find_peaks(
                neg_ay,
                prominence=1.0,
                distance=self.config.bounce_peak_distance
            )

        if len(peaks) == 0:
            # Method 3: Direct maximum position if trajectory has distinct V shape
            peak_idx = int(np.argmax(y_coords))
            # Verify it is not at the exact boundaries
            if 1 <= peak_idx <= n_pts - 2:
                if y_coords[peak_idx] > y_coords[peak_idx - 1] and y_coords[peak_idx] > y_coords[peak_idx + 1]:
                    peaks = np.array([peak_idx])

        if len(peaks) == 0:
            return None

        # Select the most prominent peak (primary bounce)
        bounce_idx = int(peaks[0])
        if len(peaks) > 1 and 'prominences' in properties and len(properties['prominences']) > 0:
            best_prom_idx = int(np.argmax(properties['prominences']))
            bounce_idx = int(peaks[best_prom_idx])

        bounce_x = float(x_coords[bounce_idx])
        bounce_y = float(y_coords[bounce_idx])

        return {
            "frame_idx": int(bounce_idx),
            "x": bounce_x,
            "y": bounce_y,
            "confidence": 0.95
        }
