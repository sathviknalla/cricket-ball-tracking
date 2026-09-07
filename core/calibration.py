import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class PitchDimensions:
    length: float = 20.12         # Stump to stump in meters
    width: float = 3.05           # Pitch strip width in meters
    half_width: float = 1.524     # Half width in meters
    bowling_popping_crease: float = 1.22
    batting_popping_crease: float = 18.90
    return_crease_half_width: float = 1.32
    stump_height: float = 0.7112  # 28 inches in meters
    stump_width: float = 0.2286   # 9 inches in meters (outer edge to outer edge)
    stump_half_width: float = 0.1143
    bails_height: float = 0.74    # Top of bails in meters
    ball_radius: float = 0.036    # 7.2 cm diameter (0.036m radius)

class MetricPitchCalibration:
    """
    Handles camera calibration, planar homography, PnP camera pose estimation,
    and 3D metric coordinate reconstruction for cricket pitch broadcast footage.
    """
    def __init__(self, dimensions: Optional[PitchDimensions] = None):
        self.dims = dimensions or PitchDimensions()
        self.homography_img_to_pitch: Optional[np.ndarray] = None
        self.homography_pitch_to_img: Optional[np.ndarray] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None

        self._init_default_synthetic_homography()

    def _init_default_synthetic_homography(self):
        """Initializes a calibrated homography for standard broadcast angle."""
        pitch_pts = np.array([
            [-self.dims.return_crease_half_width, self.dims.bowling_popping_crease],  # Top-left (Bowler end)
            [self.dims.return_crease_half_width, self.dims.bowling_popping_crease],   # Top-right
            [self.dims.return_crease_half_width, self.dims.batting_popping_crease],   # Bottom-right (Batsman end)
            [-self.dims.return_crease_half_width, self.dims.batting_popping_crease]   # Bottom-left
        ], dtype=np.float32)

        img_pts = np.array([
            [380.0, 180.0],
            [900.0, 180.0],
            [1040.0, 620.0],
            [240.0, 620.0]
        ], dtype=np.float32)

        self.compute_homography(img_pts, pitch_pts)

    def compute_homography(self, img_points: np.ndarray, pitch_points: np.ndarray) -> np.ndarray:
        """
        Calculates planar homography mapping 2D image pixels [u, v] to metric pitch [X, Y].
        """
        img_points = np.asarray(img_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)
        
        H, _ = cv2.findHomography(img_points, pitch_points)
        if H is not None:
            self.homography_img_to_pitch = H
            self.homography_pitch_to_img = np.linalg.inv(H)
        return H

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        Maps image pixel coordinates (u, v) to ground plane (X, Y, Z=0) in meters.
        """
        if self.homography_img_to_pitch is None:
            return (0.0, 0.0)

        vec = np.array([u, v, 1.0], dtype=np.float64)
        pitch_vec = self.homography_img_to_pitch @ vec
        w = pitch_vec[2]
        if abs(w) < 1e-8:
            return (0.0, 0.0)
        
        X = float(pitch_vec[0] / w)
        Y = float(pitch_vec[1] / w)
        return (X, Y)

    def ground_to_pixel(self, X: float, Y: float) -> Tuple[float, float]:
        """
        Maps pitch ground coordinates (X, Y) in meters to image pixel (u, v).
        """
        if self.homography_pitch_to_img is None:
            return (0.0, 0.0)

        vec = np.array([X, Y, 1.0], dtype=np.float64)
        img_vec = self.homography_pitch_to_img @ vec
        w = img_vec[2]
        if abs(w) < 1e-8:
            return (0.0, 0.0)

        u = float(img_vec[0] / w)
        v = float(img_vec[1] / w)
        return (u, v)

    def reconstruct_3d_trajectory(
        self,
        pixel_trajectory: List[Tuple[float, float]],
        bounce_frame_idx: int,
        fps: float = 30.0,
        release_height: float = 2.15,
        gravity: float = 9.81
    ) -> np.ndarray:
        """
        Reconstructs metric 3D coordinates [X, Y, Z] from monocular 2D pixel trajectory
        by fusing planar ground homography with projectile physics constraints.
        Returns an (N, 3) numpy array in meters.
        """
        n_pts = len(pixel_trajectory)
        if n_pts == 0:
            return np.empty((0, 3), dtype=np.float64)

        dt = 1.0 / fps
        traj_3d = np.zeros((n_pts, 3), dtype=np.float64)
        r = self.dims.ball_radius

        # 1. Project ground (X, Y) coordinates via homography
        for i, (u, v) in enumerate(pixel_trajectory):
            gx, gy = self.pixel_to_ground(u, v)
            traj_3d[i, 0] = gx
            traj_3d[i, 1] = gy

        # 2. Physics-based vertical Z reconstruction
        b_idx = max(1, min(bounce_frame_idx, n_pts - 2))
        t_bounce = b_idx * dt
        
        # Center of ball reaches height r at bounce
        vz0 = ((r - release_height) + 0.5 * gravity * (t_bounce ** 2)) / t_bounce

        for i in range(b_idx + 1):
            t = i * dt
            z = release_height + vz0 * t - 0.5 * gravity * (t ** 2)
            traj_3d[i, 2] = max(r, float(z))

        # Bounce point center
        traj_3d[b_idx, 2] = r

        # Post-bounce phase
        vz_pre = vz0 - gravity * t_bounce
        restitution_coeff = 0.65
        vz_rebound = -vz_pre * restitution_coeff

        for i in range(b_idx + 1, n_pts):
            t_post = (i - b_idx) * dt
            z = r + vz_rebound * t_post - 0.5 * gravity * (t_post ** 2)
            traj_3d[i, 2] = max(r, float(z))

        return traj_3d
