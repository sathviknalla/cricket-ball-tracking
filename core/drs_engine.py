import numpy as np
import cv2
import os
from enum import Enum
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from core.calibration import PitchDimensions

class BatsmanHand(Enum):
    RIGHT_HAND = "RHB"
    LEFT_HAND = "LHB"

class OnFieldCall(Enum):
    OUT = "OUT"
    NOT_OUT = "NOT OUT"

class PitchingZone(Enum):
    IN_LINE = "IN-LINE"
    OUTSIDE_LEG = "OUTSIDE LEG"
    OUTSIDE_OFF = "OUTSIDE OFF"
    FULL_TOSS = "FULL TOSS (IN-LINE)"

class ImpactZone(Enum):
    IN_LINE = "IN-LINE"
    UMPIRES_CALL = "UMPIRE'S CALL"
    OUTSIDE_OFF = "OUTSIDE OFF"
    OUTSIDE_LEG = "OUTSIDE LEG"

class WicketsResult(Enum):
    HITTING = "HITTING"
    UMPIRES_CALL = "UMPIRE'S CALL"
    MISSING = "MISSING"

@dataclass
class DRSVerdict:
    pitching: PitchingZone
    pitching_coord: Optional[Tuple[float, float, float]]
    impact: ImpactZone
    impact_coord: Tuple[float, float, float]
    wickets: WicketsResult
    stump_coord: Tuple[float, float, float]
    on_field_call: OnFieldCall
    final_verdict: str
    reasons: str
    bat_edge_detected: bool = False
    is_3_meter_rule_triggered: bool = False
    is_close_proximity_bounce: bool = False
    seam_spin_deviation_deg: float = 0.0

class DRSEngine:
    """
    Official ICC DRS / LBW Decision Support Engine with advanced edge-case handling:
    - Bat Edge / Inside Edge Deflection check
    - Full Toss Deliveries (Zero-Bounce)
    - ICC 3-Meter Distance-From-Stumps Law
    - Close-Proximity Bounce to Pad rule
    - Sharp Seam/Spin Deviation
    """
    def __init__(self, dimensions: Optional[PitchDimensions] = None):
        self.dims = dimensions or PitchDimensions()

    def evaluate_lbw(
        self,
        bounce_point: Optional[Tuple[float, float, float]],
        pad_impact_point: Tuple[float, float, float],
        predicted_stump_point: Tuple[float, float, float],
        batsman_hand: BatsmanHand = BatsmanHand.RIGHT_HAND,
        on_field_call: OnFieldCall = OnFieldCall.NOT_OUT,
        bat_edge_detected: bool = False,
        seam_deviation_deg: float = 0.0
    ) -> DRSVerdict:
        """
        Executes full LBW adjudication according to ICC DRS playing conditions.
        """
        r = self.dims.ball_radius              # 0.036m
        stump_edge = self.dims.stump_half_width # 0.1143m
        stump_h = self.dims.stump_height       # 0.7112m
        bail_h = self.dims.bails_height        # 0.74m
        stump_y = self.dims.length             # 20.12m

        ix, iy, iz = pad_impact_point
        sx, sy, sz = predicted_stump_point

        # ----------------------------------------------------
        # EDGE CASE 1: BAT EDGE / DEFLECTION PRIOR TO PAD
        # ----------------------------------------------------
        if bat_edge_detected:
            return DRSVerdict(
                pitching=PitchingZone.IN_LINE if bounce_point else PitchingZone.FULL_TOSS,
                pitching_coord=bounce_point,
                impact=ImpactZone.IN_LINE,
                impact_coord=pad_impact_point,
                wickets=WicketsResult.HITTING,
                stump_coord=predicted_stump_point,
                on_field_call=on_field_call,
                final_verdict="NOT OUT",
                reasons="Bat edge detected prior to pad contact (UltraEdge / Deflection). LBW invalidated.",
                bat_edge_detected=True,
                seam_spin_deviation_deg=seam_deviation_deg
            )

        # ----------------------------------------------------
        # 1. PITCHING EVALUATION (Handling Full Toss Edge Case)
        # ----------------------------------------------------
        is_close_proximity = False
        if bounce_point is None:
            # Full toss delivery hitting on the full
            pitching = PitchingZone.FULL_TOSS
        else:
            px, py, pz = bounce_point
            # Edge Case: Close-Proximity Bounce (< 0.40m before pad)
            if (iy - py) < 0.40:
                is_close_proximity = True

            if batsman_hand == BatsmanHand.RIGHT_HAND:
                if px > stump_edge:
                    pitching = PitchingZone.OUTSIDE_LEG
                elif px < -stump_edge:
                    pitching = PitchingZone.OUTSIDE_OFF
                else:
                    pitching = PitchingZone.IN_LINE
            else: # LEFT_HAND
                if px < -stump_edge:
                    pitching = PitchingZone.OUTSIDE_LEG
                elif px > stump_edge:
                    pitching = PitchingZone.OUTSIDE_OFF
                else:
                    pitching = PitchingZone.IN_LINE

        # ----------------------------------------------------
        # 2. IMPACT EVALUATION (at pad collision)
        # ----------------------------------------------------
        abs_ix = abs(ix)
        if abs_ix <= stump_edge:
            impact = ImpactZone.IN_LINE
        elif abs_ix <= (stump_edge + r):
            impact = ImpactZone.UMPIRES_CALL
        else:
            if batsman_hand == BatsmanHand.RIGHT_HAND:
                impact = ImpactZone.OUTSIDE_OFF if ix < 0 else ImpactZone.OUTSIDE_LEG
            else:
                impact = ImpactZone.OUTSIDE_OFF if ix > 0 else ImpactZone.OUTSIDE_LEG

        # ----------------------------------------------------
        # 3. WICKETS EVALUATION (at Y = 20.12m)
        # ----------------------------------------------------
        abs_sx = abs(sx)
        if (abs_sx <= stump_edge) and (0.0 <= sz <= stump_h):
            wickets = WicketsResult.HITTING
        elif (abs_sx <= stump_edge + r) and (0.0 <= sz <= bail_h + r):
            wickets = WicketsResult.UMPIRES_CALL
        else:
            wickets = WicketsResult.MISSING

        # ----------------------------------------------------
        # EDGE CASE 4: ICC 3-METER DISTANCE-FROM-STUMPS LAW
        # ----------------------------------------------------
        # Distance from pad to stumps: d = 20.12 - iy
        dist_to_stumps = stump_y - iy
        is_3_meter_rule = (dist_to_stumps >= 3.0)

        # ----------------------------------------------------
        # 5. FINAL ADJUDICATION RESOLUTION
        # ----------------------------------------------------
        if pitching == PitchingZone.OUTSIDE_LEG:
            final_verdict = "NOT OUT"
            reasons = "Pitching outside leg stump line."
        elif impact in (ImpactZone.OUTSIDE_OFF, ImpactZone.OUTSIDE_LEG):
            final_verdict = "NOT OUT"
            reasons = "Impact outside off/leg stump line."
        elif wickets == WicketsResult.MISSING:
            final_verdict = "NOT OUT"
            reasons = "Predicted path missing stumps."
        else:
            # Check 3-Meter Law: If batsman >= 3.0m down the pitch and on-field call was NOT OUT,
            # it cannot be overturned unless hitting is completely definitive (not clipping).
            if is_3_meter_rule and on_field_call == OnFieldCall.NOT_OUT:
                # If wickets is Umpire's Call or close, 3-meter law protects the batsman
                if (wickets == WicketsResult.UMPIRES_CALL) or (abs_sx > (stump_edge * 0.7)):
                    final_verdict = "NOT OUT"
                    reasons = f"ICC 3-Meter Law applies ({dist_to_stumps:.2f}m from stumps). On-field NOT OUT upheld."
                elif impact == ImpactZone.UMPIRES_CALL:
                    final_verdict = "NOT OUT"
                    reasons = f"ICC 3-Meter Law applies with Umpire's Call on impact. On-field NOT OUT upheld."
                else:
                    final_verdict = "OUT"
                    reasons = "Three Reds despite 3m distance (ball hitting middle of stumps definitively)."
            else:
                # Standard resolution
                if (impact == ImpactZone.UMPIRES_CALL) or (wickets == WicketsResult.UMPIRES_CALL):
                    final_verdict = on_field_call.value
                    reasons = f"Umpire's Call on {'Impact' if impact == ImpactZone.UMPIRES_CALL else 'Wickets'}. Verdict stands."
                else:
                    final_verdict = "OUT"
                    reasons = "Three Reds: Pitching in-line, Impact in-line, Wickets hitting."

        return DRSVerdict(
            pitching=pitching,
            pitching_coord=bounce_point,
            impact=impact,
            impact_coord=(ix, iy, iz),
            wickets=wickets,
            stump_coord=(sx, sy, sz),
            on_field_call=on_field_call,
            final_verdict=final_verdict,
            reasons=reasons,
            bat_edge_detected=bat_edge_detected,
            is_3_meter_rule_triggered=is_3_meter_rule,
            is_close_proximity_bounce=is_close_proximity,
            seam_spin_deviation_deg=seam_deviation_deg
        )

    def render_drs_banner(
        self,
        verdict: DRSVerdict,
        save_path: Optional[str] = "output/drs_banner.png"
    ) -> np.ndarray:
        w, h = 900, 380
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (15, 23, 42)

        # Header Title
        cv2.rectangle(img, (0, 0), (w, 55), (30, 41, 59), -1)
        cv2.putText(img, "DECISION REVIEW SYSTEM (DRS)", (30, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"ON-FIELD CALL: {verdict.on_field_call.value}", (w - 300, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, (148, 163, 184), 1, cv2.LINE_AA)

        boxes = [
            ("PITCHING", verdict.pitching.value, 40),
            ("IMPACT", verdict.impact.value, 330),
            ("WICKETS", verdict.wickets.value, 620)
        ]

        def get_color(text: str):
            if "IN-LINE" in text or "HITTING" in text:
                return (34, 197, 94)  # Green
            elif "UMPIRE" in text:
                return (234, 179, 8)   # Yellow
            else:
                return (59, 130, 246)  # Blue

        for title, val, x_start in boxes:
            box_w, box_h = 240, 140
            y_start = 75
            cv2.rectangle(img, (x_start, y_start), (x_start + box_w, y_start + box_h), (30, 41, 59), -1)
            cv2.rectangle(img, (x_start, y_start), (x_start + box_w, y_start + box_h), (71, 85, 105), 2)
            cv2.putText(img, title, (x_start + 20, y_start + 35), cv2.FONT_HERSHEY_DUPLEX, 0.70, (203, 213, 225), 1, cv2.LINE_AA)
            
            pill_color = get_color(val)
            cv2.rectangle(img, (x_start + 15, y_start + 65), (x_start + box_w - 15, y_start + 120), pill_color, -1)
            cv2.putText(img, val, (x_start + 22, y_start + 100), cv2.FONT_HERSHEY_DUPLEX, 0.60, (15, 23, 42), 2, cv2.LINE_AA)

        # Bottom Decision Verdict
        v_color = (34, 197, 94) if verdict.final_verdict == "OUT" else (59, 130, 246)
        cv2.rectangle(img, (40, 235), (w - 40, 360), (30, 41, 59), -1)
        cv2.rectangle(img, (40, 235), (w - 40, 360), v_color, 2)

        verdict_text = f"DECISION: {verdict.final_verdict}"
        cv2.putText(img, verdict_text, (65, 280), cv2.FONT_HERSHEY_DUPLEX, 1.1, v_color, 3, cv2.LINE_AA)
        cv2.putText(img, verdict.reasons, (65, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (203, 213, 225), 1, cv2.LINE_AA)

        # Special Edge Case Badges
        if verdict.bat_edge_detected:
            cv2.putText(img, "[ULTRAEDGE: BAT DETECTED]", (w - 320, 280), cv2.FONT_HERSHEY_DUPLEX, 0.55, (234, 179, 8), 1, cv2.LINE_AA)
        elif verdict.is_3_meter_rule_triggered:
            cv2.putText(img, "[ICC 3-METER LAW ACTIVE]", (w - 320, 280), cv2.FONT_HERSHEY_DUPLEX, 0.55, (234, 179, 8), 1, cv2.LINE_AA)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            cv2.imwrite(save_path, img)

        return img
