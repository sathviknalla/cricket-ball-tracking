import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from config import TrajectoryConfig

# Robust import with native fallback if scipy binary is restricted
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

    raw_peaks = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            raw_peaks.append(i)

    if not raw_peaks:
        return np.array([], dtype=int), {"prominences": np.array([])}

    valid_peaks = []
    prominences = []
    for p in raw_peaks:
        pv = x[p]
        left_min = pv
        for l in range(p - 1, -1, -1):
            if x[l] > pv:
                break
            if x[l] < left_min:
                left_min = x[l]

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
    """Moving polynomial filter smoothing trajectory coords."""
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
    6-State 2D Kalman Filter [x, y, vx, vy, ax, ay] for ball trajectory smoothing
    and occlusion bridging.
    """
    def __init__(self, dt: float = 1.0, config: Optional[TrajectoryConfig] = None):
        self.dt = dt
        self.config = config or TrajectoryConfig()

        self.state_dim = 6
        self.meas_dim = 2
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)

        self.F = np.eye(self.state_dim, dtype=np.float64)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[0, 4] = 0.5 * dt * dt
        self.F[1, 5] = 0.5 * dt * dt
        self.F[2, 4] = dt
        self.F[3, 5] = dt

        self.H = np.zeros((self.meas_dim, self.state_dim), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

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
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)
        self.x[0, 0] = initial_pos[0]
        self.x[1, 0] = initial_pos[1]
        self.x[2, 0] = initial_vel[0]
        self.x[3, 0] = initial_vel[1]
        self.x[5, 0] = 0.0
        self.P = np.eye(self.state_dim, dtype=np.float64) * 10.0
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        if not self.initialized:
            return (0.0, 0.0)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float64)
        if not self.initialized:
            self.init_state(measurement)
            return measurement

        self.predict()

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y_residual = z - (self.H @ self.x)
        self.x = self.x + K @ y_residual
        I = np.eye(self.state_dim, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def smooth_trajectory(
        self,
        points: List[Optional[Tuple[float, float]]]
    ) -> np.ndarray:
        if not points:
            return np.empty((0, 2), dtype=np.float64)

        first_valid_idx = None
        for i, pt in enumerate(points):
            if pt is not None:
                first_valid_idx = i
                break

        if first_valid_idx is None:
            return np.zeros((len(points), 2), dtype=np.float64)

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
        leading_pred = points[first_valid_idx]
        for i in range(first_valid_idx):
            filtered_states.append(np.array(leading_pred, dtype=np.float64))

        for i in range(first_valid_idx, len(points)):
            pt = points[i]
            if i == first_valid_idx:
                smooth_pt = (float(self.x[0, 0]), float(self.x[1, 0]))
            elif pt is not None:
                smooth_pt = self.update(pt)
            else:
                smooth_pt = self.predict()
            filtered_states.append(np.array(smooth_pt, dtype=np.float64))

        smoothed = np.array(filtered_states, dtype=np.float64)

        if len(smoothed) >= 7:
            try:
                window_len = min(7, len(smoothed) if len(smoothed) % 2 == 1 else len(smoothed) - 1)
                smoothed[:, 0] = savgol_filter(smoothed[:, 0], window_length=window_len, polyorder=2)
                smoothed[:, 1] = savgol_filter(smoothed[:, 1], window_length=window_len, polyorder=2)
            except Exception:
                pass

        return smoothed

# Alias
Kalman2DSmoother = KalmanTrajectorySmoother

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
        if trajectory is None or len(trajectory) < 5:
            return None

        y_coords = trajectory[:, 1]
        x_coords = trajectory[:, 0]
        n_pts = len(y_coords)

        vy = np.gradient(y_coords)
        ay = np.gradient(vy)

        peaks, properties = find_peaks(
            y_coords,
            prominence=self.config.bounce_peak_prominence,
            distance=self.config.bounce_peak_distance
        )

        if len(peaks) == 0:
            neg_ay = -ay
            peaks, properties = find_peaks(
                neg_ay,
                prominence=1.0,
                distance=self.config.bounce_peak_distance
            )

        if len(peaks) == 0:
            peak_idx = int(np.argmax(y_coords))
            if 1 <= peak_idx <= n_pts - 2:
                if y_coords[peak_idx] > y_coords[peak_idx - 1] and y_coords[peak_idx] > y_coords[peak_idx + 1]:
                    peaks = np.array([peak_idx])

        if len(peaks) == 0:
            return None

        bounce_idx = int(peaks[0])
        if len(peaks) > 1 and 'prominences' in properties and len(properties['prominences']) > 0:
            best_prom_idx = int(np.argmax(properties['prominences']))
            bounce_idx = int(peaks[best_prom_idx])

        bounce_x = float(x_coords[bounce_idx])
        bounce_y = float(y_coords[bounce_idx])

        prominence_val = float(properties['prominences'][0]) if 'prominences' in properties and len(properties['prominences']) > 0 else 5.0
        confidence = min(0.99, max(0.50, 0.70 + 0.05 * prominence_val))

        return {
            "frame_idx": int(bounce_idx),
            "x": bounce_x,
            "y": bounce_y,
            "confidence": confidence
        }

@dataclass
class Kalman3DConfig:
    dt: float = 1.0 / 30.0  # 30 fps
    gravity: float = 9.81
    process_noise_pos: float = 0.001
    process_noise_vel: float = 0.01
    measurement_noise: float = 0.005
    drag_coeff: float = 0.0070  # Aerodynamic drag gamma = 0.5 * rho * Cd * A / m

class Kalman3DSmoother:
    """
    6-State 3D Kalman Filter [X, Y, Z, vx, vy, vz] for 3D trajectory tracking,
    gravitational physics modeling, multi-frame occlusion bridging,
    bat-edge deflection detection, and seam/spin deviation estimation.
    """
    def __init__(self, config: Optional[Kalman3DConfig] = None):
        self.config = config or Kalman3DConfig()
        self.dt = self.config.dt
        self.g = self.config.gravity

        self.state_dim = 6
        self.meas_dim = 3
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)

        self.F = np.eye(self.state_dim, dtype=np.float64)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        self.Bu = np.zeros((self.state_dim, 1), dtype=np.float64)
        self.Bu[2, 0] = -0.5 * self.g * (self.dt ** 2)
        self.Bu[5, 0] = -self.g * self.dt

        self.H = np.zeros((self.meas_dim, self.state_dim), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.P = np.eye(self.state_dim, dtype=np.float64) * 0.05
        self.Q = np.eye(self.state_dim, dtype=np.float64)
        self.Q[0, 0] = self.config.process_noise_pos
        self.Q[1, 1] = self.config.process_noise_pos
        self.Q[2, 2] = self.config.process_noise_pos
        self.Q[3, 3] = self.config.process_noise_vel
        self.Q[4, 4] = self.config.process_noise_vel
        self.Q[5, 5] = self.config.process_noise_vel

        self.R = np.eye(self.meas_dim, dtype=np.float64) * self.config.measurement_noise
        self.initialized = False

    def init_state(self, initial_pos: Tuple[float, float, float], initial_vel: Tuple[float, float, float] = (0.0, 35.0, -2.0)):
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)
        self.x[0, 0] = initial_pos[0]
        self.x[1, 0] = initial_pos[1]
        self.x[2, 0] = initial_pos[2]
        self.x[3, 0] = initial_vel[0]
        self.x[4, 0] = initial_vel[1]
        self.x[5, 0] = initial_vel[2]
        self.P = np.eye(self.state_dim, dtype=np.float64) * 0.01
        self.initialized = True

    def predict(self) -> Tuple[float, float, float]:
        if not self.initialized:
            return (0.0, 0.0, 0.0)

        self.x = self.F @ self.x + self.Bu
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]))

    def update(self, measurement: Tuple[float, float, float]) -> Tuple[float, float, float]:
        z = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float64)
        if not self.initialized:
            self.init_state(measurement)
            return measurement

        self.predict()

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        residual = z - (self.H @ self.x)
        self.x = self.x + K @ residual
        I = np.eye(self.state_dim, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return (float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]))

    def smooth_3d_trajectory(
        self,
        points: List[Optional[Tuple[float, float, float]]]
    ) -> np.ndarray:
        if not points:
            return np.empty((0, 3), dtype=np.float64)

        valid_indices = [i for i, p in enumerate(points) if p is not None]
        if not valid_indices:
            return np.zeros((len(points), 3), dtype=np.float64)

        first_valid_idx = valid_indices[0]

        if len(valid_indices) >= 2:
            second_idx = valid_indices[min(3, len(valid_indices) - 1)]
            dt_span = (second_idx - first_valid_idx) * self.dt
            p1 = points[first_valid_idx]
            p2 = points[second_idx]
            
            vx_init = (p2[0] - p1[0]) / dt_span
            vy_init = (p2[1] - p1[1]) / dt_span
            vz_init = (p2[2] - p1[2] + 0.5 * self.g * (dt_span ** 2)) / dt_span
            init_vel = (vx_init, vy_init, vz_init)
        else:
            init_vel = (0.0, 35.0, 0.0)

        self.init_state(points[first_valid_idx], init_vel)

        filtered = []
        for i in range(first_valid_idx):
            filtered.append(np.array(points[first_valid_idx], dtype=np.float64))

        for i in range(first_valid_idx, len(points)):
            pt = points[i]
            if i == first_valid_idx:
                smooth_pt = (float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]))
            elif pt is not None:
                smooth_pt = self.update(pt)
            else:
                smooth_pt = self.predict()
            filtered.append(np.array(smooth_pt, dtype=np.float64))

        return np.array(filtered, dtype=np.float64)

    @staticmethod
    def detect_bat_deflection(
        trajectory_3d: np.ndarray,
        threshold_angle_deg: float = 4.5,
        min_y_check: float = 16.0
    ) -> Tuple[bool, Optional[int]]:
        if trajectory_3d is None or len(trajectory_3d) < 4:
            return False, None

        vels = np.diff(trajectory_3d, axis=0)
        n_steps = len(vels)

        for i in range(1, n_steps):
            y_pos = trajectory_3d[i, 1]
            z_pos = trajectory_3d[i, 2]

            if y_pos >= min_y_check and z_pos > 0.08:
                v1 = vels[i - 1]
                v2 = vels[i]
                norm1 = np.linalg.norm(v1) + 1e-6
                norm2 = np.linalg.norm(v2) + 1e-6

                cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_theta))

                if angle_deg >= threshold_angle_deg:
                    return True, i

        return False, None

    @staticmethod
    def compute_seam_spin_deviation(
        pre_bounce_vel: Tuple[float, float, float],
        post_bounce_vel: Tuple[float, float, float]
    ) -> float:
        vx_pre, vy_pre, _ = pre_bounce_vel
        vx_post, vy_post, _ = post_bounce_vel

        angle_pre = np.degrees(np.arctan2(vx_pre, max(0.1, vy_pre)))
        angle_post = np.degrees(np.arctan2(vx_post, max(0.1, vy_post)))
        return float(angle_post - angle_pre)

    @staticmethod
    def project_to_stumps(
        impact_point: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        target_y: float = 20.12,
        fps: float = 30.0,
        gravity: float = 9.81,
        drag_coeff: float = 0.0070
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        x, y, z = impact_point
        vx, vy, vz = velocity
        dt = 1.0 / fps

        if vy <= 1.0:
            vy = 30.0

        pts = [(x, y, z)]
        while y < target_y:
            x += vx * dt
            y += vy * dt
            z += vz * dt - 0.5 * gravity * (dt ** 2)

            v_mag = np.hypot(vx, np.hypot(vy, vz))
            drag_x = -drag_coeff * v_mag * vx
            drag_y = -drag_coeff * v_mag * vy
            drag_z = -drag_coeff * v_mag * vz - gravity

            vx += drag_x * dt
            vy += drag_y * dt
            vz += drag_z * dt

            pts.append((x, min(target_y, y), max(0.0, z)))

        last_pt = pts[-1]
        second_last = pts[-2] if len(pts) >= 2 else (x, y, z)

        y1, y2 = second_last[1], last_pt[1]
        span_y = max(1e-5, y2 - y1)
        interp_factor = (target_y - y1) / span_y

        exact_x = second_last[0] + interp_factor * (last_pt[0] - second_last[0])
        exact_z = second_last[2] + interp_factor * (last_pt[2] - second_last[2])
        stump_impact = (float(exact_x), float(target_y), float(max(0.0, exact_z)))

        pts[-1] = stump_impact
        return np.array(pts, dtype=np.float64), stump_impact
