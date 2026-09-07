import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BenchmarkResults:
    detection_ap: float
    mota: float
    motp: float
    id_switches: int
    false_positives: int
    false_negatives: int
    total_gt: int
    fps: float
    latency_breakdown_ms: Dict[str, float]

class HawkEyeBenchmark:
    """
    Benchmark suite calculating Detection AP, Tracking MOTA/MOTP, FPS,
    and generating visual performance comparison charts.
    """
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """Calculates Intersection over Union between two bboxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return float(iou)

    def evaluate_detections(
        self,
        predicted_boxes: List[List[Tuple[float, float, float, float]]],
        predicted_scores: List[List[float]],
        gt_boxes: List[List[Tuple[float, float, float, float]]],
        iou_threshold: float = 0.5
    ) -> float:
        """
        Calculates Detection Average Precision (AP) at given IoU threshold.
        """
        total_gt = sum(len(gts) for gts in gt_boxes)
        if total_gt == 0:
            return 1.0

        all_preds = []
        for f_idx, (boxes, scores) in enumerate(zip(predicted_boxes, predicted_scores)):
            for b, s in zip(boxes, scores):
                all_preds.append((s, f_idx, b))

        if not all_preds:
            return 0.0

        # Sort by confidence descending
        all_preds.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        matched_gt = {f_idx: set() for f_idx in range(len(gt_boxes))}

        for p_idx, (score, f_idx, pred_b) in enumerate(all_preds):
            frame_gts = gt_boxes[f_idx]
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gt_b in enumerate(frame_gts):
                iou = self.compute_iou(pred_b, gt_b)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx not in matched_gt[f_idx]:
                tp[p_idx] = 1.0
                matched_gt[f_idx].add(best_gt_idx)
            else:
                fp[p_idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / total_gt
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        # 11-point interpolation or area under PR curve
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p_over_t = precisions[recalls >= t]
            if len(p_over_t) > 0:
                ap += np.max(p_over_t)
        ap /= 11.0
        return float(ap)

    def evaluate_tracking(
        self,
        predicted_tracks: List[List[Tuple[int, Tuple[float, float, float, float]]]],  # [frame][(track_id, bbox)]
        gt_tracks: List[List[Tuple[int, Tuple[float, float, float, float]]]],          # [frame][(gt_id, bbox)]
        iou_threshold: float = 0.3
    ) -> Tuple[float, float, int, int, int, int]:
        """
        Calculates MOTA, MOTP, ID Switches, FP, FN, and total GT count.
        """
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_iou_matches = 0
        sum_iou = 0.0

        last_gt_to_track_mapping: Dict[int, int] = {}

        n_frames = max(len(predicted_tracks), len(gt_tracks))
        for f in range(n_frames):
            preds = predicted_tracks[f] if f < len(predicted_tracks) else []
            gts = gt_tracks[f] if f < len(gt_tracks) else []
            total_gt += len(gts)

            # Match predictions to GT via IoU / spatial overlap
            matched_p = set()
            matched_g = set()

            for g_idx, (g_id, g_box) in enumerate(gts):
                best_iou = 0.0
                best_p_idx = -1
                for p_idx, (p_id, p_box) in enumerate(preds):
                    if p_idx in matched_p:
                        continue
                    iou = self.compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_p_idx = p_idx

                if best_iou >= iou_threshold and best_p_idx >= 0:
                    matched_p.add(best_p_idx)
                    matched_g.add(g_idx)
                    p_id = preds[best_p_idx][0]
                    sum_iou += best_iou
                    total_iou_matches += 1

                    # Check for ID switch
                    if g_id in last_gt_to_track_mapping:
                        if last_gt_to_track_mapping[g_id] != p_id:
                            total_idsw += 1
                    last_gt_to_track_mapping[g_id] = p_id

            frame_fn = len(gts) - len(matched_g)
            frame_fp = len(preds) - len(matched_p)
            total_fn += frame_fn
            total_fp += frame_fp

        if total_gt == 0:
            mota = 1.0
        else:
            mota = 1.0 - (total_fn + total_fp + total_idsw) / float(total_gt)

        motp = (sum_iou / total_iou_matches) if total_iou_matches > 0 else 0.0
        return float(mota), float(motp), int(total_idsw), int(total_fp), int(total_fn), int(total_gt)

    def generate_benchmark_charts(
        self,
        results: BenchmarkResults,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plots performance comparison charts (MOTA breakdown, module latency, FPS).
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "benchmark_metrics.png")

        fig, axs = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0f172a')

        # Chart 1: Key Metrics Bar Chart
        ax1 = axs[0]
        ax1.set_facecolor('#1e293b')
        metrics = ['Detection AP', 'Tracking MOTA', 'Tracking MOTP']
        values = [results.detection_ap * 100, max(0.0, results.mota * 100), results.motp * 100]
        colors = ['#38bdf8', '#34d399', '#a78bfa']

        bars = ax1.bar(metrics, values, color=colors, width=0.55, edgecolor='white', alpha=0.9)
        ax1.set_ylim(0, 110)
        ax1.set_ylabel('Score (%)', color='#cbd5e1', fontsize=11)
        ax1.set_title('Accuracy & Tracking Metrics', color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
        ax1.tick_params(colors='#94a3b8')

        for bar, val in zip(bars, values):
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{val:.1f}%", ha='center', va='bottom', color='#f8fafc', fontweight='bold')

        # Chart 2: Tracking Error Breakdown
        ax2 = axs[1]
        ax2.set_facecolor('#1e293b')
        err_labels = ['False Positives', 'False Negatives', 'ID Switches']
        err_counts = [results.false_positives, results.false_negatives, results.id_switches]
        err_colors = ['#f87171', '#fb923c', '#fbbf24']

        ax2.bar(err_labels, err_counts, color=err_colors, width=0.55, edgecolor='white', alpha=0.9)
        ax2.set_title('Tracking Error Composition', color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
        ax2.set_ylabel('Count', color='#cbd5e1', fontsize=11)
        ax2.tick_params(colors='#94a3b8')

        # Chart 3: Pipeline Latency Breakdown
        ax3 = axs[2]
        ax3.set_facecolor('#1e293b')
        modules = list(results.latency_breakdown_ms.keys())
        latencies = list(results.latency_breakdown_ms.values())
        lat_colors = ['#60a5fa', '#f472b6', '#4ade80', '#c084fc']

        wedges, texts, autotexts = ax3.pie(
            latencies, labels=modules, autopct='%1.1f%%',
            colors=lat_colors[:len(modules)], startangle=140,
            textprops=dict(color="#cbd5e1")
        )
        for autotext in autotexts:
            autotext.set_color('#0f172a')
            autotext.set_weight('bold')

        ax3.set_title(f'Module Latency (Total FPS: {results.fps:.1f})', color='#f8fafc', fontsize=13, fontweight='bold', pad=12)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return save_path
