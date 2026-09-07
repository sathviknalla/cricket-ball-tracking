import numpy as np
import pytest
from core.trajectory import Kalman3DSmoother, Kalman3DConfig

def test_kalman_3d_7_frame_occlusion():
    """
    Inject a 7-frame complete occlusion in a parabolic 3D flight path.
    Assert reconstructed path error < 0.03m (3cm) RMSE against ground truth.
    """
    dt = 1.0 / 30.0  # 30 fps
    g = 9.81
    n_frames = 35
    
    # Generate Ground Truth 3D parabolic trajectory
    # Ball bowled from X=0.1, Y=1.5, Z=2.15 with vx=-0.015, vy=36.0 m/s (~130 km/h), vz0=-1.5 m/s
    gt_traj = []
    vx, vy, vz0 = -0.015, 36.0, -1.5
    for i in range(n_frames):
        t = i * dt
        x = 0.1 + vx * t
        y = 1.5 + vy * t
        z = 2.15 + vz0 * t - 0.5 * g * (t ** 2)
        gt_traj.append((x, y, z))

    # Inject 7-frame complete occlusion from frame 14 to 20
    occluded_traj = []
    for i in range(n_frames):
        if 14 <= i <= 20:  # 7-frame gap
            occluded_traj.append(None)
        else:
            # Add small sensor measurement noise (0.3 cm standard dev)
            jitter_x = 0.002 if i % 2 == 0 else -0.002
            jitter_y = 0.003 if i % 3 == 0 else -0.003
            jitter_z = 0.002 if i % 2 == 1 else -0.002
            gt_pt = gt_traj[i]
            occluded_traj.append((gt_pt[0] + jitter_x, gt_pt[1] + jitter_y, gt_pt[2] + jitter_z))

    smoother = Kalman3DSmoother(Kalman3DConfig(dt=dt, gravity=g))
    reconstructed = smoother.smooth_3d_trajectory(occluded_traj)

    assert len(reconstructed) == n_frames
    assert not np.isnan(reconstructed).any()

    # Calculate RMSE over the 7-frame occlusion window (frames 14 to 20)
    occlusion_gt = np.array(gt_traj[14:21])
    occlusion_pred = reconstructed[14:21]

    errors = occlusion_pred - occlusion_gt
    rmse_x = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(errors[:, 1] ** 2)))
    rmse_z = float(np.sqrt(np.mean(errors[:, 2] ** 2)))
    total_rmse = float(np.sqrt(np.mean(errors ** 2)))

    print(f"7-Frame Occlusion RMSE: Total={total_rmse*100:.2f} cm (X={rmse_x*100:.2f} cm, Y={rmse_y*100:.2f} cm, Z={rmse_z*100:.2f} cm)")

    # Assert total RMSE is strictly below 3 cm (0.03m)
    assert total_rmse < 0.03, f"Total RMSE {total_rmse:.4f}m exceeds 0.03m (3 cm) threshold"
    assert rmse_x < 0.03
    assert rmse_y < 0.03
    assert rmse_z < 0.03
