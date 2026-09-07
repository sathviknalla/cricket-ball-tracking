import os
import cv2
import time
import argparse
import queue
import threading
import numpy as np
from typing import Optional, List, Tuple
from core.calibration import MetricPitchCalibration, PitchDimensions
from core.trajectory import Kalman3DSmoother
from core.drs_engine import DRSEngine, BatsmanHand, OnFieldCall
from core.visualizer import Plotly3DVisualizer
from pipeline import HawkEyePipeline

class ThreadedVideoReader:
    """Producer-consumer frame reader for max video decoding throughput."""
    def __init__(self, video_path: str, queue_size: int = 64):
        self.video_path = video_path
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.cap = cv2.VideoCapture(video_path)
        self.thread = threading.Thread(target=self._read_frames, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read_frames(self):
        while not self.stopped and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put(None)
                break
            self.queue.put(frame)
        self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        return self.queue.get()

    def stop(self):
        self.stopped = True

def run_cli_pipeline(
    input_video: Optional[str] = None,
    output_dir: str = "output",
    enable_drs: bool = True,
    plot_3d: bool = True,
    threaded: bool = True,
    handedness: str = "RHB",
    umpire_call: str = "NOT OUT",
    simulate: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("   HAWK-EYE LITE 3D & DRS DECISION SUPPORT ENGINE")
    print("=" * 60)

    # 1. Run synthetic simulation if simulate or no input
    if simulate or input_video is None:
        print("[CLI] Running synthetic 3D broadcast simulation...")
        pipeline = HawkEyePipeline()
        res = pipeline.run_synthetic_simulation()

        # 3D Calibration & DRS adjudication
        calib = MetricPitchCalibration()
        drs_engine = DRSEngine()
        
        # Evaluate LBW
        b_pt = (res['bounce_info']['x'], res['bounce_info']['y']) if res['bounce_info'] else (640.0, 560.0)
        gx, gy = calib.pixel_to_ground(b_pt[0], b_pt[1])
        bounce_3d = (gx, gy, 0.0)
        pad_3d = (gx + 0.05, 18.85, 0.46)

        proj_arr, stump_3d = Kalman3DSmoother.project_to_stumps(
            impact_point=pad_3d,
            velocity=(0.04 * 38.0, 38.0 * 0.85, -0.2 * 38.0),
            target_y=20.12
        )

        batsman_hand = BatsmanHand.RIGHT_HAND if handedness == "RHB" else BatsmanHand.LEFT_HAND
        on_field = OnFieldCall.OUT if umpire_call == "OUT" else OnFieldCall.NOT_OUT

        verdict = drs_engine.evaluate_lbw(
            bounce_point=bounce_3d,
            pad_impact_point=pad_3d,
            predicted_stump_point=stump_3d,
            batsman_hand=batsman_hand,
            on_field_call=on_field
        )

        banner_path = os.path.join(output_dir, "drs_banner.png")
        drs_engine.render_drs_banner(verdict, save_path=banner_path)

        print(f"\n[DRS Adjudication Result]")
        print(f" -> Pitching: {verdict.pitching.value}")
        print(f" -> Impact:   {verdict.impact.value}")
        print(f" -> Wickets:  {verdict.wickets.value}")
        print(f" -> FINAL VERDICT: {verdict.final_verdict} ({verdict.reasons})")
        print(f" -> DRS Banner Saved: {banner_path}")
        return

    # 2. Process real input video file
    print(f"[CLI] Processing video: {input_video} (Threaded={threaded})")
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    pipeline = HawkEyePipeline()
    reader = ThreadedVideoReader(input_video).start() if threaded else None
    cap_fallback = cv2.VideoCapture(input_video) if not threaded else None

    frame_idx = 0
    t0 = time.time()

    while True:
        frame = reader.get_frame() if threaded else cap_fallback.read()[1]
        if frame is None:
            break
        frame_idx += 1
        vis_frame, history, bounce = pipeline.process_frame(frame)
        if frame_idx % 30 == 0:
            elapsed = time.time() - t0
            print(f" -> Processed {frame_idx}/{total_frames} frames ({frame_idx/elapsed:.1f} FPS)...")

    if cap_fallback:
        cap_fallback.release()

    print(f"[CLI] Completed processing {frame_idx} frames.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hawk-Eye 3D Cricket Ball Tracking & DRS")
    parser.add_argument("--input", type=str, default=None, help="Path to input cricket video")
    parser.add_argument("--output", type=str, default="output", help="Directory for output artifacts")
    parser.add_argument("--drs", action="store_true", default=True, help="Enable DRS LBW adjudication")
    parser.add_argument("--plot-3d", action="store_true", default=True, help="Generate interactive 3D plot")
    parser.add_argument("--threaded", action="store_true", default=True, help="Use multi-threaded frame queue")
    parser.add_argument("--handedness", type=str, default="RHB", choices=["RHB", "LHB"], help="Batsman Handedness")
    parser.add_argument("--umpire-call", type=str, default="NOT OUT", choices=["OUT", "NOT OUT"], help="On-field Umpire Call")
    parser.add_argument("--simulate", action="store_true", help="Run full synthetic broadcast demo")
    args = parser.parse_args()

    run_cli_pipeline(
        input_video=args.input,
        output_dir=args.output,
        enable_drs=args.drs,
        plot_3d=args.plot_3d,
        threaded=args.threaded,
        handedness=args.handedness,
        umpire_call=args.umpire_call,
        simulate=args.simulate
    )
