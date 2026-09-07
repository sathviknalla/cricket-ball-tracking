import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict
from core.calibration import PitchDimensions
from core.drs_engine import DRSVerdict

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
        verdict: Optional[DRSVerdict] = None
    ) -> go.Figure:
        """
        Constructs an interactive 3D scene containing the metric pitch surface,
        both sets of wickets, crease lines, and segmented ball trajectories.
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
