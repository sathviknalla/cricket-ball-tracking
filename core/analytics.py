import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from core.calibration import PitchDimensions

@dataclass
class DeliveryRecord:
    delivery_id: int
    speed_kmh: float
    bounce_x: float
    bounce_y: float
    pad_x: float
    pad_y: float
    stump_x: float
    stump_z: float
    verdict: str
    xd_score: float
    length_zone: str
    line_zone: str

class ExpectedDismissalEngine:
    """
    Computes Expected Dismissal (xD) probability score (0.00 to 1.00 / 0-100%)
    for a delivery based on kinematic line, length, deviation, and wicket-zone intersection.
    """
    def __init__(self, dimensions: Optional[PitchDimensions] = None):
        self.dims = dimensions or PitchDimensions()

    def classify_length_zone(self, bounce_y: float) -> str:
        if bounce_y < 8.0:
            return "Bouncer"
        elif bounce_y < 10.0:
            return "Short / Back of Length"
        elif bounce_y <= 13.5:
            return "Good Length"
        elif bounce_y <= 16.5:
            return "Full / Driving Length"
        else:
            return "Yorker"

    def classify_line_zone(self, bounce_x: float, is_rhb: bool = True) -> str:
        stump_edge = self.dims.stump_half_width
        sign = 1.0 if is_rhb else -1.0
        x_rel = bounce_x * sign

        if x_rel < -stump_edge * 2.0:
            return "Wide Outside Off"
        elif x_rel < -stump_edge:
            return "Corridor of Uncertainty / 4th-5th Stump"
        elif abs(x_rel) <= stump_edge:
            return "On The Stumps"
        else:
            return "Down Leg Side"

    def calculate_xd(
        self,
        bounce_point: Tuple[float, float, float],
        impact_point: Tuple[float, float, float],
        stump_impact: Tuple[float, float, float],
        speed_kmh: float = 142.5,
        seam_deviation_deg: float = 0.0,
        is_rhb: bool = True
    ) -> Dict:
        """
        Calculates xD score and factor breakdown.
        """
        bx, by, _ = bounce_point
        ix, iy, iz = impact_point
        sx, sy, sz = stump_impact
        stump_edge = self.dims.stump_half_width
        stump_h = self.dims.stump_height

        length_zone = self.classify_length_zone(by)
        line_zone = self.classify_line_zone(bx, is_rhb=is_rhb)

        # 1. Length baseline threat score
        if length_zone == "Good Length":
            length_threat = 0.42
        elif length_zone == "Full / Driving Length":
            length_threat = 0.35
        elif length_zone == "Yorker":
            length_threat = 0.38
        elif length_zone == "Short / Back of Length":
            length_threat = 0.18
        else:
            length_threat = 0.08

        # 2. Line precision threat score
        dist_to_center = abs(bx)
        if dist_to_center <= stump_edge:
            line_threat = 0.35
        elif dist_to_center <= stump_edge * 1.5:
            line_threat = 0.25
        elif dist_to_center <= stump_edge * 2.5:
            line_threat = 0.12
        else:
            line_threat = 0.03

        # 3. Stump intersection accuracy
        is_hitting_stumps = (abs(sx) <= stump_edge) and (0.05 <= sz <= stump_h)
        is_clipping = (abs(sx) <= stump_edge + self.dims.ball_radius) and (0.0 <= sz <= self.dims.bails_height)

        if is_hitting_stumps:
            # Middle stump hit gives peak boost
            closeness_to_middle = 1.0 - (abs(sx) / max(1e-4, stump_edge))
            wicket_threat = 0.20 + 0.10 * closeness_to_middle
        elif is_clipping:
            wicket_threat = 0.10
        else:
            wicket_threat = 0.01

        # 4. Pace and deviation multiplier
        speed_factor = min(1.20, max(0.80, speed_kmh / 135.0))
        deviation_boost = min(0.12, abs(seam_deviation_deg) * 0.03)

        raw_xd = (length_threat + line_threat + wicket_threat + deviation_boost) * speed_factor
        xd_clamped = float(np.clip(raw_xd, 0.02, 0.98))

        return {
            "xd_score": xd_clamped,
            "xd_percentage": round(xd_clamped * 100.0, 1),
            "length_zone": length_zone,
            "line_zone": line_zone,
            "threat_breakdown": {
                "length_threat": length_threat,
                "line_threat": line_threat,
                "wicket_threat": wicket_threat,
                "deviation_boost": deviation_boost,
                "speed_multiplier": speed_factor
            }
        }

class PitchHeatmapAggregator:
    """
    Aggregates delivery landing points on the 2D pitch strip into a 2D spatial density matrix.
    """
    def __init__(self, grid_rows: int = 20, grid_cols: int = 15):
        self.grid_rows = grid_rows # Along Y length (0 to 20.12m)
        self.grid_cols = grid_cols # Across X width (-1.524 to +1.524m)
        self.density_matrix = np.zeros((grid_rows, grid_cols), dtype=np.float64)
        self.deliveries: List[Tuple[float, float]] = []

    def add_bounce(self, x: float, y: float, weight: float = 1.0):
        self.deliveries.append((x, y))
        col = int(np.clip(((x + 1.524) / 3.048) * self.grid_cols, 0, self.grid_cols - 1))
        row = int(np.clip((y / 20.12) * self.grid_rows, 0, self.grid_rows - 1))

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r_idx, c_idx = row + dr, col + dc
                if 0 <= r_idx < self.grid_rows and 0 <= c_idx < self.grid_cols:
                    dist_sq = dr ** 2 + dc ** 2
                    self.density_matrix[r_idx, c_idx] += weight * np.exp(-dist_sq / 1.5)

    def get_normalized_heatmap(self) -> np.ndarray:
        max_val = np.max(self.density_matrix)
        if max_val <= 1e-6:
            return self.density_matrix
        return self.density_matrix / max_val

class DeliverySessionTracker:
    """
    Manages spell session state, bowling spell statistics, and multi-delivery histories.
    """
    def __init__(self):
        self.deliveries: List[DeliveryRecord] = []
        self.heatmap = PitchHeatmapAggregator()
        self.xd_engine = ExpectedDismissalEngine()

    def record_delivery(
        self,
        speed_kmh: float,
        bounce_point: Tuple[float, float, float],
        pad_point: Tuple[float, float, float],
        stump_point: Tuple[float, float, float],
        verdict: str,
        seam_deviation_deg: float = 0.0,
        is_rhb: bool = True
    ) -> DeliveryRecord:
        deliv_id = len(self.deliveries) + 1
        xd_res = self.xd_engine.calculate_xd(
            bounce_point=bounce_point,
            impact_point=pad_point,
            stump_impact=stump_point,
            speed_kmh=speed_kmh,
            seam_deviation_deg=seam_deviation_deg,
            is_rhb=is_rhb
        )

        rec = DeliveryRecord(
            delivery_id=deliv_id,
            speed_kmh=speed_kmh,
            bounce_x=bounce_point[0],
            bounce_y=bounce_point[1],
            pad_x=pad_point[0],
            pad_y=pad_point[1],
            stump_x=stump_point[0],
            stump_z=stump_point[2],
            verdict=verdict,
            xd_score=xd_res['xd_score'],
            length_zone=xd_res['length_zone'],
            line_zone=xd_res['line_zone']
        )
        self.deliveries.append(rec)
        self.heatmap.add_bounce(bounce_point[0], bounce_point[1], weight=1.0)
        return rec

    def get_summary_stats(self) -> Dict:
        if not self.deliveries:
            return {
                "total_deliveries": 0,
                "avg_speed_kmh": 0.0,
                "avg_xd_score": 0.0,
                "good_length_percentage": 0.0,
                "lbw_appeals": 0,
                "wickets_hitting": 0
            }

        speeds = [d.speed_kmh for d in self.deliveries]
        xds = [d.xd_score for d in self.deliveries]
        good_lengths = sum(1 for d in self.deliveries if d.length_zone == "Good Length")
        hitting_count = sum(1 for d in self.deliveries if d.verdict in ["OUT", "HITTING"])

        return {
            "total_deliveries": len(self.deliveries),
            "avg_speed_kmh": float(np.mean(speeds)),
            "avg_xd_score": float(np.mean(xds)),
            "good_length_percentage": float((good_lengths / len(self.deliveries)) * 100.0),
            "lbw_appeals": len(self.deliveries),
            "wickets_hitting": hitting_count
        }
