import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from config import TrajectoryConfig

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
            if pt is not None:
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
