import json

import numpy as np
import pytest

from jersey_detect.data import ImageRecord, train_subset
from jersey_detect.metrics import size_metrics
from jersey_detect.rescore import find_checkpoints
from jersey_detect.study import JERSEYNET, YOLO_JN_AUG, plan
from jersey_detect.yolo_baseline import SETTINGS, scaled_train_args

RECORDS = [ImageRecord(f"/data/train/images/{i:04d}.jpg") for i in range(200)]


def test_subsets_are_nested_and_fixed():
    small, large = train_subset(RECORDS, 0.1), train_subset(RECORDS, 0.25)
    assert len(small) == 20 and len(large) == 50
    assert {r.path for r in small} <= {r.path for r in large}
    assert [r.path for r in small] == [r.path for r in train_subset(RECORDS, 0.1)]
    assert len(train_subset(RECORDS, 1.0)) == 200
    with pytest.raises(ValueError):
        train_subset(RECORDS, 0.0)


def test_subset_runs_keep_the_number_of_steps():
    scratch = next(s for s in SETTINGS if s.slug == "yolov8n_from_scratch_256_px")
    args = scaled_train_args(scratch, 0.25)
    assert args["epochs"] == 560 and args["patience"] == 400 and args["close_mosaic"] == 40
    assert scaled_train_args(scratch, 1.0) == {"epochs": 140}


def test_augmentation_matched_setting_has_no_mosaic_or_flip():
    args = scaled_train_args(YOLO_JN_AUG, 1.0)
    assert args["mosaic"] == 0.0 and args["fliplr"] == 0.0 and args["close_mosaic"] == 0
    assert YOLO_JN_AUG.weights == "yolov8n.yaml" and YOLO_JN_AUG.imgsz == 256


def test_plan_finishes_each_seed_first():
    jobs = plan(seeds=(0, 1), fractions=(0.1, 0.25))
    assert len(jobs) == 2 * (1 + 2 * 3)
    assert {s for _, _, s in jobs[:7]} == {0}
    assert jobs[0] == (YOLO_JN_AUG.name, 1.0, 0) and jobs[1] == (JERSEYNET, 0.1, 0)


def test_size_metrics_split_boxes_by_side():
    small = np.array([[0.10, 0.10, 0.14, 0.14]])  # 5 px at 128 px
    large = np.array([[0.50, 0.50, 0.75, 0.75]])  # 32 px
    gts = [np.vstack([small, large])]
    res = size_metrics(gts, [large], [[0.9]])
    assert res["small"]["num_gt"] == 1 and res["small"]["ap50_95"] == 0.0
    assert res["medium"]["num_gt"] == 0
    assert res["large"]["ap50_95"] == pytest.approx(1.0)


def test_find_checkpoints(tmp_path):
    yolo = tmp_path / "out1" / "yolo_results" / "runs" / "yolov8n_from_scratch_384_px_seed2"
    (yolo / "weights").mkdir(parents=True)
    (yolo / "weights" / "best.pt").write_bytes(b"")
    (yolo / "args.yaml").write_text("task: detect\nimgsz: 384\nbatch: 16\n")
    jn = tmp_path / "out2" / "results" / "runs" / "jersey_net_seed1"
    jn.mkdir(parents=True)
    (jn / "best.pt").write_bytes(b"")
    (jn / "config.json").write_text(json.dumps({"image_size": 384, "mosaic": 0.5, "giou_weight": 1.0,
                                                "seed": 1}))
    found = {c["name"]: c for c in find_checkpoints([tmp_path])}
    assert found["yolov8n_from_scratch_384_px"]["imgsz"] == 384
    assert found["yolov8n_from_scratch_384_px"]["seed"] == 2
    assert found["JerseyNet, 384 px, mosaic, GIoU"]["seed"] == 1
    abl = tmp_path / "out3" / "ablation_results" / "runs" / "baseline_seed0"
    abl.mkdir(parents=True)
    (abl / "best.pt").write_bytes(b"")
    (abl / "config.json").write_text(json.dumps({"image_size": 256, "seed": 0}))
    names = {c["name"] for c in find_checkpoints([tmp_path])}
    assert "JerseyNet, 256 px (ablation run)" in names
