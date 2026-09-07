import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict
from config import VisualizerConfig
from core.calibration import PitchDimensions
from core.drs_engine import DRSVerdict

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
        ax.axhline(0, color='#f8fafc', linewidth=2.5)
        ax.axhline(1.22, color='#f8fafc', linewidth=1.5, linestyle='-')
        ax.axhline(18.90, color='#f8fafc', linewidth=1.5, linestyle='-')
        ax.axhline(20.12, color='#f8fafc', linewidth=2.5)

        # Stumps (Bowler End: y=0, Batsman End: y=20.12)
        stump_x = [-0.11, 0.0, 0.11]
        for sx in stump_x:
            ax.plot(sx, 0, 'wo', markersize=4)
            ax.plot(sx, 20.12, 'wo', markersize=4)

        # Map trajectory points to pitch coordinates
        if trajectory is not None and len(trajectory) > 0:
            img_xs = trajectory[:, 0]
            img_ys = trajectory[:, 1]

            min_y, max_y = np.min(img_ys), np.max(img_ys)
            span_y = max(1.0, max_y - min_y)
            pitch_y = ((img_ys - min_y) / span_y) * (pitch_length - 2.0) + 1.0

            mean_x = np.mean(img_xs)
            span_x = max(1.0, np.ptp(img_xs))
            pitch_x = ((img_xs - mean_x) / span_x) * (pitch_width * 0.4)

            ax.plot(pitch_x, pitch_y, color='#38bdf8', linewidth=3.5, label='Ball Trajectory', zorder=5)
            ax.scatter(pitch_x[0], pitch_y[0], color='#fbbf24', s=70, label='Release Point', zorder=6)

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

        ax.text(0, -0.6, "BOWLER'S END", color='#cbd5e1', ha='center', fontsize=9, fontweight='semibold')
        ax.text(0, 20.7, "BATSMAN'S END", color='#cbd5e1', ha='center', fontsize=9, fontweight='semibold')

        legend = ax.legend(loc='lower left', facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return fig

class Plotly3DVisualizer:
    """
    Renders interactive 3D cricket pitch and ball trajectory visualizations using Plotly.
    """
    def __init__(self, dims: Optional[PitchDimensions] = None):
        self.dims = dims or PitchDimensions()

    def build_3d_pitch_figure(
        self,
        pre_bounce_traj: np.ndarray,
        post_bounce_traj: Optional[np.ndarray] = None,
        projected_traj: Optional[np.ndarray] = None,
        bounce_point: Optional[Tuple[float, float, float]] = None,
        impact_point: Optional[Tuple[float, float, float]] = None,
        stump_impact: Optional[Tuple[float, float, float]] = None,
        verdict: Optional[DRSVerdict] = None,
        confidence_cone_radii: Optional[List[Tuple[float, float]]] = None
    ) -> go.Figure:
        """
        Constructs an interactive 3D scene containing the metric pitch surface,
        both sets of wickets, crease lines, segmented ball trajectories, and 3D confidence cone.
        """
        fig = go.Figure()
        pw = self.dims.half_width  # 1.524m
        pl = self.dims.length      # 20.12m

        # 1. Pitch Surface (Ground plane Z=0)
        fig.add_trace(go.Mesh3d(
            x=[-pw, pw, pw, -pw],
            y=[0, 0, pl, pl],
            z=[0, 0, 0, 0],
            color='#334155',
            opacity=0.75,
            name='Pitch Strip',
            hoverinfo='name'
        ))

        # 2. Crease Lines
        # Bowling Crease (y=0) & Popping Crease (y=1.22)
        fig.add_trace(go.Scatter3d(
            x=[-1.32, 1.32, 1.32, -1.32, -1.32],
            y=[0, 0, 1.22, 1.22, 0],
            z=[0, 0, 0, 0, 0],
            mode='lines',
            line=dict(color='white', width=4),
            name="Bowler Crease"
        ))
        # Batting Crease (y=20.12) & Popping Crease (y=18.90)
        fig.add_trace(go.Scatter3d(
            x=[-1.32, 1.32, 1.32, -1.32, -1.32],
            y=[pl, pl, 18.90, 18.90, pl],
            z=[0, 0, 0, 0, 0],
            mode='lines',
            line=dict(color='white', width=4),
            name="Batting Crease"
        ))

        # 3. Wickets at both ends
        stump_xs = [-0.1143, 0.0, 0.1143]
        sh = self.dims.stump_height
        bh = self.dims.bails_height

        for y_pos, name_prefix in [(0.0, "Bowler"), (pl, "Batsman")]:
            for sx in stump_xs:
                fig.add_trace(go.Scatter3d(
                    x=[sx, sx], y=[y_pos, y_pos], z=[0, sh],
                    mode='lines', line=dict(color='#fbbf24', width=8),
                    showlegend=False, hoverinfo='skip'
                ))
            # Bails
            fig.add_trace(go.Scatter3d(
                x=[-0.12, 0.12], y=[y_pos, y_pos], z=[bh, bh],
                mode='lines', line=dict(color='#f59e0b', width=6),
                name=f"{name_prefix} Stumps" if sx == 0.0 else None,
                showlegend=(name_prefix == "Batsman")
            ))

        # 4. Trajectories
        # Pre-bounce path
        if pre_bounce_traj is not None and len(pre_bounce_traj) > 0:
            fig.add_trace(go.Scatter3d(
                x=pre_bounce_traj[:, 0], y=pre_bounce_traj[:, 1], z=pre_bounce_traj[:, 2],
                mode='lines', line=dict(color='#38bdf8', width=6),
                name='Pre-Bounce Flight'
            ))

        # Post-bounce path
        if post_bounce_traj is not None and len(post_bounce_traj) > 0:
            fig.add_trace(go.Scatter3d(
                x=post_bounce_traj[:, 0], y=post_bounce_traj[:, 1], z=post_bounce_traj[:, 2],
                mode='lines', line=dict(color='#f97316', width=6),
                name='Post-Bounce Path'
            ))

        # Post-impact projected path (DRS projection)
        if projected_traj is not None and len(projected_traj) > 0:
            fig.add_trace(go.Scatter3d(
                x=projected_traj[:, 0], y=projected_traj[:, 1], z=projected_traj[:, 2],
                mode='lines', line=dict(color='#ef4444', width=6, dash='dash'),
                name='DRS Projected Path'
            ))

            # 3D Uncertainty / Confidence Cone Boundary
            if confidence_cone_radii is not None and len(confidence_cone_radii) == len(projected_traj):
                upper_x, upper_z, lower_x, lower_z = [], [], [], []
                for pt, (rx, rz) in zip(projected_traj, confidence_cone_radii):
                    upper_x.append(pt[0] + rx)
                    upper_z.append(pt[2] + rz)
                    lower_x.append(pt[0] - rx)
                    lower_z.append(max(0.0, pt[2] - rz))

                fig.add_trace(go.Scatter3d(
                    x=upper_x, y=projected_traj[:, 1], z=upper_z,
                    mode='lines', line=dict(color='rgba(239, 68, 68, 0.35)', width=2),
                    name='95% Confidence Upper Bound'
                ))
                fig.add_trace(go.Scatter3d(
                    x=lower_x, y=projected_traj[:, 1], z=lower_z,
                    mode='lines', line=dict(color='rgba(239, 68, 68, 0.35)', width=2),
                    name='95% Confidence Lower Bound'
                ))

        # 5. Key Impact Markers
        if bounce_point is not None:
            fig.add_trace(go.Scatter3d(
                x=[bounce_point[0]], y=[bounce_point[1]], z=[bounce_point[2]],
                mode='markers+text', marker=dict(color='#ef4444', size=8, symbol='diamond'),
                text=['Pitch Bounce'], textposition='top center',
                name='Pitch Bounce Point'
            ))

        if impact_point is not None:
            fig.add_trace(go.Scatter3d(
                x=[impact_point[0]], y=[impact_point[1]], z=[impact_point[2]],
                mode='markers+text', marker=dict(color='#eab308', size=8, symbol='circle'),
                text=['Pad Impact'], textposition='top center',
                name='Pad Impact Point'
            ))

        if stump_impact is not None:
            hit_color = '#ef4444' if (verdict and verdict.wickets.value == "HITTING") else '#38bdf8'
            fig.add_trace(go.Scatter3d(
                x=[stump_impact[0]], y=[stump_impact[1]], z=[stump_impact[2]],
                mode='markers+text', marker=dict(color=hit_color, size=10, symbol='cross'),
                text=[f"Wicket Arrival (Z={stump_impact[2]:.2f}m)"], textposition='top center',
                name='Predicted Stump Arrival'
            ))

        # Layout & Dark Theme Styling
        fig.update_layout(
            template='plotly_dark',
            scene=dict(
                xaxis=dict(title='Width X (m)', range=[-2.0, 2.0], backgroundcolor='#0f172a', gridcolor='#334155'),
                yaxis=dict(title='Length Y (m)', range=[-1.0, 22.0], backgroundcolor='#0f172a', gridcolor='#334155'),
                zaxis=dict(title='Height Z (m)', range=[0.0, 3.0], backgroundcolor='#0f172a', gridcolor='#334155'),
                aspectratio=dict(x=1, y=3.5, z=1),
                camera=dict(
                    eye=dict(x=-2.2, y=-1.5, z=1.8),
                    center=dict(x=0, y=0.4, z=0)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(x=0.02, y=0.95, bgcolor='rgba(15, 23, 42, 0.8)')
        )
        return fig

    @staticmethod
    def build_velocity_timeline_figure(
        y_coords: np.ndarray,
        speeds_kmh: List[float],
        bounce_y: Optional[float] = None,
        impact_y: Optional[float] = None
    ) -> go.Figure:
        """
        Renders a 2D line plot showing ball deceleration along the pitch length (0 to 20.12m).
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_coords,
            y=speeds_kmh,
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            marker=dict(size=4, color='#34d399'),
            name='Ball Speed (km/h)'
        ))

        if bounce_y is not None:
            fig.add_vline(x=bounce_y, line_width=1.5, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Bounce", annotation_position="top left")
        if impact_y is not None:
            fig.add_vline(x=impact_y, line_width=1.5, line_dash="dash", line_color="#ef4444",
                          annotation_text="Pad Impact", annotation_position="top right")

        fig.update_layout(
            template='plotly_dark',
            title="Velocity Telemetry & Drag Decay Curve",
            xaxis_title="Pitch Length Y (meters)",
            yaxis_title="Speed (km/h)",
            height=320,
            margin=dict(l=40, r=20, t=40, b=30),
            paper_bgcolor='#0f172a',
            plot_bgcolor='#1e293b'
        )
        return fig

    @staticmethod
    def build_pitch_heatmap_figure(
        density_matrix: np.ndarray,
        deliveries: List[Tuple[float, float]]
    ) -> go.Figure:
        """
        Renders 2D interactive spatial density heatmap of bowling pitch landing zones.
        """
        fig = go.Figure()
        
        # 2D Heatmap surface
        y_axis = np.linspace(0, 20.12, density_matrix.shape[0])
        x_axis = np.linspace(-1.524, 1.524, density_matrix.shape[1])

        fig.add_trace(go.Heatmap(
            z=density_matrix,
            x=x_axis,
            y=y_axis,
            colorscale='Viridis',
            opacity=0.85,
            showscale=True,
            name='Density Heatmap'
        ))

        # Scatter dots of individual deliveries
        if deliveries:
            dxs = [d[0] for d in deliveries]
            dys = [d[1] for d in deliveries]
            fig.add_trace(go.Scatter(
                x=dxs, y=dys,
                mode='markers',
                marker=dict(size=7, color='#f43f5e', line=dict(color='white', width=1)),
                name='Pitch Bounce Points'
            ))

        # Zone Lines
        for zy, zlabel in [(8.0, "Short"), (10.0, "Back of Length"), (13.5, "Good Length"), (16.5, "Full"), (20.12, "Crease")]:
            fig.add_hline(y=zy, line_width=1, line_dash="dot", line_color="#94a3b8",
                          annotation_text=zlabel, annotation_position="bottom right")

        fig.update_layout(
            template='plotly_dark',
            title="2D Bowling Spell Pitch Heatmap & Length Distribution",
            xaxis_title="Pitch Width X (m)",
            yaxis_title="Pitch Length Y (m)",
            height=450,
            paper_bgcolor='#0f172a',
            plot_bgcolor='#1e293b',
            yaxis=dict(autorange='reversed') # Bowler at top, batsman at bottom
        )
        return fig

