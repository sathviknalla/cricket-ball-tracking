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
    pitching_coord: Tuple[float, float, float]
    impact: ImpactZone
    impact_coord: Tuple[float, float, float]
    wickets: WicketsResult
    stump_coord: Tuple[float, float, float]
    on_field_call: OnFieldCall
    final_verdict: str
    reasons: str

class DRSEngine:
    """
    Official ICC DRS / LBW Decision Support Engine.
    Evaluates 3D trajectory against official Pitching, Impact, and Wicket criteria.
    """
    def __init__(self, dimensions: Optional[PitchDimensions] = None):
        self.dims = dimensions or PitchDimensions()

    def evaluate_lbw(
        self,
        bounce_point: Tuple[float, float, float],
        pad_impact_point: Tuple[float, float, float],
        predicted_stump_point: Tuple[float, float, float],
        batsman_hand: BatsmanHand = BatsmanHand.RIGHT_HAND,
        on_field_call: OnFieldCall = OnFieldCall.NOT_OUT
    ) -> DRSVerdict:
        """
        Executes full LBW adjudication according to ICC DRS playing conditions.
        """
        r = self.dims.ball_radius              # 0.036m
        stump_edge = self.dims.stump_half_width # 0.1143m
        stump_h = self.dims.stump_height       # 0.7112m
        bail_h = self.dims.bails_height        # 0.74m

        px, py, pz = bounce_point
        ix, iy, iz = pad_impact_point
        sx, sy, sz = predicted_stump_point

        # ----------------------------------------------------
        # 1. PITCHING EVALUATION
        # ----------------------------------------------------
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
            # Center of ball is outside, but sphere overlaps stump line (>50% margin)
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
        # Check horizontal and vertical bounds
        # Solid hit: ball center inside width and within stump height
        if (abs_sx <= stump_edge) and (0.0 <= sz <= stump_h):
            wickets = WicketsResult.HITTING
        # Umpire's Call clipping: ball sphere intersects outer edges or bails
        elif (abs_sx <= stump_edge + r) and (0.0 <= sz <= bail_h + r):
            wickets = WicketsResult.UMPIRES_CALL
        else:
            wickets = WicketsResult.MISSING

        # ----------------------------------------------------
        # 4. FINAL ADJUDICATION RESOLUTION
        # ----------------------------------------------------
        # Ball pitching outside leg cannot be out LBW
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
            # Check for Umpire's Call conditions
            if (impact == ImpactZone.UMPIRES_CALL) or (wickets == WicketsResult.UMPIRES_CALL):
                final_verdict = on_field_call.value
                reasons = f"Umpire's Call on {'Impact' if impact == ImpactZone.UMPIRES_CALL else 'Wickets'}. Verdict stands."
            else:
                final_verdict = "OUT"
                reasons = "Three Reds: Pitching in-line, Impact in-line, Wickets hitting."

        return DRSVerdict(
            pitching=pitching,
            pitching_coord=(px, py, pz),
            impact=impact,
            impact_coord=(ix, iy, iz),
            wickets=wickets,
            stump_coord=(sx, sy, sz),
            on_field_call=on_field_call,
            final_verdict=final_verdict,
            reasons=reasons
        )

    def render_drs_banner(
        self,
        verdict: DRSVerdict,
        save_path: Optional[str] = "output/drs_banner.png"
    ) -> np.ndarray:
        """
        Renders an official broadcast DRS decision graphic with 3 sequential status lights.
        """
        w, h = 900, 360
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (15, 23, 42)  # Dark slate background

        # Header Title
        cv2.rectangle(img, (0, 0), (w, 55), (30, 41, 59), -1)
        cv2.putText(img, "DECISION REVIEW SYSTEM (DRS)", (30, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"ON-FIELD CALL: {verdict.on_field_call.value}", (w - 300, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, (148, 163, 184), 1, cv2.LINE_AA)

        # 3 Sequential Lights: Pitching, Impact, Wickets
        boxes = [
            ("PITCHING", verdict.pitching.value, 40),
            ("IMPACT", verdict.impact.value, 330),
            ("WICKETS", verdict.wickets.value, 620)
        ]

        def get_color(text: str):
            if text in ("IN-LINE", "HITTING"):
                return (34, 197, 94)  # Green / Red (in cricket DRS hitting is red/green indicator)
            elif "UMPIRE" in text:
                return (234, 179, 8)   # Yellow
            else:
                return (59, 130, 246)  # Blue (Missing / Outside)

        for title, val, x_start in boxes:
            box_w, box_h = 240, 150
            y_start = 80
            # Outer card
            cv2.rectangle(img, (x_start, y_start), (x_start + box_w, y_start + box_h), (30, 41, 59), -1)
            cv2.rectangle(img, (x_start, y_start), (x_start + box_w, y_start + box_h), (71, 85, 105), 2)
            
            # Category title
            cv2.putText(img, title, (x_start + 20, y_start + 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (203, 213, 225), 1, cv2.LINE_AA)
            
            # Status light pill
            pill_color = get_color(val)
            cv2.rectangle(img, (x_start + 15, y_start + 70), (x_start + box_w - 15, y_start + 130), pill_color, -1)
            cv2.putText(img, val, (x_start + 25, y_start + 107), cv2.FONT_HERSHEY_DUPLEX, 0.65, (15, 23, 42), 2, cv2.LINE_AA)

        # Bottom Decision Verdict
        v_color = (34, 197, 94) if verdict.final_verdict == "OUT" else (59, 130, 246)
        cv2.rectangle(img, (40, 260), (w - 40, 335), (30, 41, 59), -1)
        cv2.rectangle(img, (40, 260), (w - 40, 335), v_color, 2)

        verdict_text = f"DECISION: {verdict.final_verdict}"
        cv2.putText(img, verdict_text, (65, 310), cv2.FONT_HERSHEY_DUPLEX, 1.1, v_color, 3, cv2.LINE_AA)
        cv2.putText(img, verdict.reasons, (420, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (203, 213, 225), 1, cv2.LINE_AA)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            cv2.imwrite(save_path, img)

        return img
