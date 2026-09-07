import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class UltraEdgeType(Enum):
    CLEAN_PASS = "CLEAN PASS (NO CONTACT)"
    BAT_EDGE = "BAT EDGE DETECTED"
    PAD_CONTACT = "PAD IMPACT"
    GLOVE_CONTACT = "GLOVE CONTACT"

@dataclass
class NoBallVerdict:
    is_no_ball: bool
    heel_position_y: float
    popping_crease_y: float
    margin_cm: float
    verdict_text: str
    confidence: float

class NoBallDetector:
    """
    Evaluates bowler front-foot landing position relative to bowling popping crease.
    Law 21.5: Bowler's front foot must land with some part of the foot grounded behind the popping crease.
    """
    def __init__(self, popping_crease_y: float = 1.22):
        self.popping_crease_y = popping_crease_y

    def evaluate_front_foot(
        self,
        foot_landing_y: float,
        foot_length_m: float = 0.28,
        is_grounded: bool = True
    ) -> NoBallVerdict:
        """
        foot_landing_y: Y position in pitch coordinates (where 0 is stumps, 1.22 is popping crease).
        Heel lands at foot_landing_y, toe extends to foot_landing_y + foot_length_m.
        """
        # Behind popping crease means Y <= 1.22m
        # Overstepping occurs when entire heel is ahead of 1.22m (i.e. foot_landing_y > 1.22)
        margin_m = foot_landing_y - self.popping_crease_y
        margin_cm = margin_m * 100.0

        is_no_ball = (margin_m > 0.0)

        if is_no_ball:
            verdict_text = f"NO BALL — Front foot overstepped by {abs(margin_cm):.1f} cm"
        else:
            verdict_text = f"LEGAL DELIVERY — Heel grounded behind crease by {abs(margin_cm):.1f} cm"

        return NoBallVerdict(
            is_no_ball=is_no_ball,
            heel_position_y=foot_landing_y,
            popping_crease_y=self.popping_crease_y,
            margin_cm=margin_cm,
            verdict_text=verdict_text,
            confidence=0.96 if is_grounded else 0.82
        )

class UltraEdgeWaveformSimulator:
    """
    Generates synthetic synchronized acoustic sound waveforms (UltraEdge / Snickometer)
    matching bat-edge or pad impact events.
    """
    def __init__(self, sample_rate_hz: int = 1000):
        self.sample_rate = sample_rate_hz

    def generate_waveform(
        self,
        n_frames: int = 60,
        fps: float = 30.0,
        contact_type: UltraEdgeType = UltraEdgeType.BAT_EDGE,
        contact_frame: int = 35
    ) -> Dict:
        """
        Generates continuous audio amplitude time-series with high-frequency harmonic spikes.
        """
        duration_sec = n_frames / fps
        n_samples = int(duration_sec * self.sample_rate)
        time_arr = np.linspace(0, duration_sec, n_samples)
        
        # Ambient crowd noise background
        ambient = np.random.normal(0, 0.04, n_samples)
        waveform = ambient.copy()

        contact_time = contact_frame / fps
        spike_window_sec = 0.08

        if contact_type == UltraEdgeType.BAT_EDGE:
            # Sharp resonant high-frequency metallic spike (~250-400Hz acoustic resonance)
            mask = np.abs(time_arr - contact_time) < spike_window_sec
            t_rel = (time_arr[mask] - contact_time) / spike_window_sec
            envelope = np.exp(-12.0 * (t_rel ** 2))
            spike = 0.95 * envelope * np.sin(2 * np.pi * 320.0 * (time_arr[mask] - contact_time))
            waveform[mask] += spike
        elif contact_type == UltraEdgeType.PAD_CONTACT:
            # Low-frequency dull thud (~60-120Hz)
            mask = np.abs(time_arr - contact_time) < spike_window_sec * 1.5
            t_rel = (time_arr[mask] - contact_time) / (spike_window_sec * 1.5)
            envelope = np.exp(-6.0 * (t_rel ** 2))
            spike = 0.45 * envelope * np.sin(2 * np.pi * 90.0 * (time_arr[mask] - contact_time))
            waveform[mask] += spike
        elif contact_type == UltraEdgeType.GLOVE_CONTACT:
            # Medium frequency flutter
            mask = np.abs(time_arr - contact_time) < spike_window_sec
            t_rel = (time_arr[mask] - contact_time) / spike_window_sec
            envelope = np.exp(-8.0 * (t_rel ** 2))
            spike = 0.65 * envelope * np.sin(2 * np.pi * 180.0 * (time_arr[mask] - contact_time))
            waveform[mask] += spike

        peak_amplitude = float(np.max(np.abs(waveform)))

        return {
            "time_axis": time_arr,
            "waveform": waveform,
            "contact_type": contact_type.value,
            "contact_time_sec": contact_time,
            "peak_amplitude": peak_amplitude,
            "is_spike_detected": (peak_amplitude > 0.60)
        }
