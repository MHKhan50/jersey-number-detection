import shutil

import numpy as np
import pytest

from jersey_detect.data import load_dataset
from jersey_detect.leakage import leakage_report, source_stem
from jersey_detect.metrics import all_metrics, bootstrap_ci, first_box_metrics
from jersey_detect.synthetic import make_dataset


def test_source_stem_strips_roboflow_suffix():
    assert source_stem("a/frame_12_jpg.rf.0a1b2c.jpg") == "frame_12"
    assert source_stem("plain.png") == "plain"


def test_copied_image_is_flagged(tmp_path):
    root = make_dataset(tmp_path / "d", counts=(12, 6, 6))
    src = next((root / "train" / "images").iterdir())
    shutil.copy(src, root / "test" / "images" / ("copy_" + src.name))
    rep = leakage_report(load_dataset(root))
    assert rep["pairs"]["train-test"]["test_images_with_near_duplicate_in_train"] >= 1
    assert rep["pairs"]["train-val"]["val_images_with_near_duplicate_in_train"] == 0


def test_first_box_protocol_uses_top_prediction_and_skips_negatives():
    gts = [np.array([[0, 0, 0.5, 0.5]]), np.zeros((0, 4)), np.array([[0, 0, 0.5, 0.5]])]
    dets = [np.array([[0.6, 0.6, 1, 1], [0, 0, 0.5, 0.5]]), np.array([[0, 0, 1, 1]]), np.zeros((0, 4))]
    scores = [np.array([0.1, 0.9]), np.array([0.5]), np.zeros(0)]
    m = first_box_metrics(gts, dets, scores)
    assert m["scored_images"] == 2
    assert m["first_box_mean_iou"] == pytest.approx(0.5)
    assert m["first_box_iou_ge_0_5"] == pytest.approx(0.5)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    gts, dets, scores = [], [], []
    for _ in range(40):
        g = np.array([[0.2, 0.2, 0.4, 0.4]])
        gts.append(g)
        dets.append(g + rng.normal(0, 0.03, (1, 4)))
        scores.append(rng.uniform(0, 1, 1))
    point = all_metrics(gts, dets, scores)
    ci = bootstrap_ci(gts, dets, scores, n_boot=200)
    for k in ("ap50_95", "ap50", "first_box_mean_iou"):
        lo, hi = ci[k]
        assert lo <= point[k] + 1e-9 and point[k] <= hi + 1e-9
