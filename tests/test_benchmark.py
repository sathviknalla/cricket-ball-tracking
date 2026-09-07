import numpy as np
import pytest
from benchmark import HawkEyeBenchmark, BenchmarkResults

def test_benchmark_iou_calculation():
    box1 = (10.0, 10.0, 30.0, 30.0)
    box2 = (10.0, 10.0, 30.0, 30.0)
    assert HawkEyeBenchmark.compute_iou(box1, box2) == pytest.approx(1.0)

    box3 = (30.0, 30.0, 50.0, 50.0)
    assert HawkEyeBenchmark.compute_iou(box1, box3) == pytest.approx(0.0)

def test_benchmark_detection_ap():
    bm = HawkEyeBenchmark()
    
    gt_boxes = [
        [(100.0, 100.0, 120.0, 120.0)],
        [(110.0, 110.0, 130.0, 130.0)]
    ]
    pred_boxes = [
        [(100.0, 100.0, 120.0, 120.0)],
        [(110.0, 110.0, 130.0, 130.0)]
    ]
    scores = [[0.95], [0.90]]

    ap = bm.evaluate_detections(pred_boxes, scores, gt_boxes, iou_threshold=0.5)
    assert ap > 0.90

def test_benchmark_tracking_mota():
    bm = HawkEyeBenchmark()
    
    gt_tracks = [
        [(1, (100.0, 100.0, 120.0, 120.0))],
        [(1, (110.0, 110.0, 130.0, 130.0))],
        [(1, (120.0, 120.0, 140.0, 140.0))]
    ]
    pred_tracks = [
        [(1, (100.0, 100.0, 120.0, 120.0))],
        [(1, (110.0, 110.0, 130.0, 130.0))],
        [(1, (120.0, 120.0, 140.0, 140.0))]
    ]

    mota, motp, idsw, fp, fn, total_gt = bm.evaluate_tracking(pred_tracks, gt_tracks)
    assert mota == pytest.approx(1.0)
    assert idsw == 0
    assert fp == 0
    assert fn == 0
