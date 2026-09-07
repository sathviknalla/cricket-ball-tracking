import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Dict
from config import VisualizerConfig

class HawkEyeVisualizer:
    """
    Visualizer module for trajectory tail rendering on video frames
    and 2D cricket pitch-map generation.
    """
    def __init__(self, config: Optional[VisualizerConfig] = None):
        self.config = config or VisualizerConfig()

    def draw_trajectory(
        self,
        frame: np.ndarray,
        history: List[Tuple[float, float]],
        bounce_point: Optional[Tuple[float, float]] = None,
        track_id: Optional[int] = None,
        current_speed_kmh: Optional[float] = 142.5
    ) -> np.ndarray:
        """
        Renders a glowing, fading trajectory tail and bounce marker on a video frame.
        """
        if frame is None or len(history) < 2:
            return frame

        vis_frame = frame.copy()
        overlay = frame.copy()
        n_pts = len(history)
        max_tail = min(self.config.trajectory_max_length, n_pts)
        pts = history[-max_tail:]

        # Draw fading trajectory segments
        for i in range(len(pts) - 1):
            alpha = (i + 1) / len(pts)
            thickness = max(2, int(2 + 4 * alpha))

            # Color interpolation between start (orange) and end (cyan)
            c_start = np.array(self.config.tail_start_color, dtype=np.float32)
            c_end = np.array(self.config.tail_end_color, dtype=np.float32)
            bgr = tuple(map(int, (1 - alpha) * c_start + alpha * c_end))

            pt1 = (int(round(pts[i][0])), int(round(pts[i][1])))
            pt2 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))

            # Glow line
            if self.config.tail_glow:
                cv2.line(overlay, pt1, pt2, bgr, thickness=thickness + 3, lineType=cv2.LINE_AA)
            cv2.line(vis_frame, pt1, pt2, bgr, thickness=thickness, lineType=cv2.LINE_AA)

        # Blend glow overlay
        if self.config.tail_glow:
            cv2.addWeighted(overlay, 0.4, vis_frame, 0.6, 0, vis_frame)

        # Draw current head position (glowing ball)
        head_x, head_y = int(round(pts[-1][0])), int(round(pts[-1][1]))
        cv2.circle(vis_frame, (head_x, head_y), 7, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis_frame, (head_x, head_y), 10, self.config.tail_end_color, 2, lineType=cv2.LINE_AA)

        # Draw bounce impact marker if detected
        if bounce_point is not None:
            bx, by = int(round(bounce_point[0])), int(round(bounce_point[1]))
            # Concentric target rings
            cv2.circle(vis_frame, (bx, by), 12, self.config.bounce_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(vis_frame, (bx, by), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                vis_frame, "PITCH BOUNCE", (bx + 15, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
            )

        # Render Hawk-Eye HUD overlay
        hud_bg = vis_frame[10:90, 10:240]
        if hud_bg.shape[0] > 0 and hud_bg.shape[1] > 0:
            black_rect = np.zeros_like(hud_bg)
            cv2.addWeighted(hud_bg, 0.3, black_rect, 0.7, 0, hud_bg)
            vis_frame[10:90, 10:240] = hud_bg

        cv2.putText(vis_frame, "HAWK-EYE LITE", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        if track_id is not None:
            cv2.putText(vis_frame, f"TRACK ID: #{track_id}", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        if current_speed_kmh is not None:
            cv2.putText(vis_frame, f"SPEED: {current_speed_kmh:.1f} km/h", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 1, cv2.LINE_AA)

        return vis_frame

    def generate_pitch_map(
        self,
        trajectory: np.ndarray,
        bounce_point: Optional[Dict[str, float]] = None,
        save_path: str = "output/pitch_map.png"
    ) -> plt.Figure:
        """
        Generates a 2D top-down Cricket Pitch Map showing ball trajectory, length zones, and bounce point.
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 12), facecolor='#111827')
        ax.set_facecolor('#1e293b')

        # Dimensions in meters
        pitch_length = self.config.pitch_length_meters  # 20.12m
        pitch_width = self.config.pitch_width_meters    # 3.05m

        # Draw Pitch Boundaries
        pitch_rect = patches.Rectangle(
            (-pitch_width / 2, 0), pitch_width, pitch_length,
            linewidth=2, edgecolor='#94a3b8', facecolor='#334155', alpha=0.9
        )
        ax.add_patch(pitch_rect)

        # Pitch Zones (Color Shading)
        # Short (4 - 10m), Good Length (10 - 14m), Full (14 - 18m), Yorker (18 - 20.12m)
        zones = [
            (4.0, 10.0, '#3b82f6', 'Short / Bouncer (4-10m)'),
            (10.0, 14.0, '#10b981', 'Good Length (10-14m)'),
            (14.0, 18.0, '#f59e0b', 'Full Length (14-18m)'),
            (18.0, 20.12, '#ef4444', 'Yorker (18-20.12m)')
        ]

        for y_start, y_end, color, label in zones:
            zone_rect = patches.Rectangle(
                (-pitch_width / 2, y_start), pitch_width, y_end - y_start,
                color=color, alpha=0.18, label=label
            )
            ax.add_patch(zone_rect)
            ax.axhline(y_end, color='#64748b', linestyle='--', linewidth=0.8, alpha=0.6)

        # Crease Markings
        # Bowling Crease (0m) & Popping Crease (1.22m)
        ax.axhline(0, color='#f8fafc', linewidth=2.5)
        ax.axhline(1.22, color='#f8fafc', linewidth=1.5, linestyle='-')

        # Batting Popping Crease (18.90m) & Bowling Crease (20.12m)
        ax.axhline(18.90, color='#f8fafc', linewidth=1.5, linestyle='-')
        ax.axhline(20.12, color='#f8fafc', linewidth=2.5)

        # Stumps (Bowler End: y=0, Batsman End: y=20.12)
        stump_x = [-0.11, 0.0, 0.11]
        for sx in stump_x:
            ax.plot(sx, 0, 'wo', markersize=4)
            ax.plot(sx, 20.12, 'wo', markersize=4)

        # Map trajectory points to pitch coordinates
        if trajectory is not None and len(trajectory) > 0:
            # Normalize trajectory coordinates to pitch space
            img_xs = trajectory[:, 0]
            img_ys = trajectory[:, 1]

            # Linear mapping from image Y span to pitch length [0, 20.12]
            min_y, max_y = np.min(img_ys), np.max(img_ys)
            span_y = max(1.0, max_y - min_y)
            pitch_y = ((img_ys - min_y) / span_y) * (pitch_length - 2.0) + 1.0

            # Normalize image X to pitch width [-1.2, 1.2]
            mean_x = np.mean(img_xs)
            span_x = max(1.0, np.ptp(img_xs))
            pitch_x = ((img_xs - mean_x) / span_x) * (pitch_width * 0.4)

            # Draw trajectory path
            ax.plot(pitch_x, pitch_y, color='#38bdf8', linewidth=3.5, label='Ball Trajectory', zorder=5)
            ax.scatter(pitch_x[0], pitch_y[0], color='#fbbf24', s=70, label='Release Point', zorder=6)

            # Plot Bounce Location
            if bounce_point is not None:
                b_idx = min(len(pitch_y) - 1, int(bounce_point.get('frame_idx', len(pitch_y) // 2)))
                bx = pitch_x[b_idx]
                by = pitch_y[b_idx]
                ax.scatter(bx, by, color='#ef4444', marker='*', s=250, edgecolor='white', linewidth=1.5, label='Bounce Point', zorder=7)
                ax.annotate(
                    f"Bounce: {by:.2f}m", (bx, by),
                    textcoords="offset points", xytext=(15, 5),
                    color='#ef4444', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1e293b", ec="#ef4444", lw=1)
                )

        ax.set_xlim(-pitch_width / 2 - 0.5, pitch_width / 2 + 0.5)
        ax.set_ylim(-1.0, pitch_length + 1.5)
        ax.set_aspect('equal')

        ax.set_title("HAWK-EYE 2D PITCH MAP", color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Pitch Width (meters)", color='#94a3b8', fontsize=10)
        ax.set_ylabel("Pitch Length (meters)", color='#94a3b8', fontsize=10)
        ax.tick_params(colors='#94a3b8')

        # Add text indicators
        ax.text(0, -0.6, "BOWLER'S END", color='#cbd5e1', ha='center', fontsize=9, fontweight='semibold')
        ax.text(0, 20.7, "BATSMAN'S END", color='#cbd5e1', ha='center', fontsize=9, fontweight='semibold')

        legend = ax.legend(loc='lower left', facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return fig
