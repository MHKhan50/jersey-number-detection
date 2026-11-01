"""The YOLO baseline needs Ultralytics, which is optional. CI runs this in its own step."""

import json
import os

import pytest

from jersey_detect.yolo_baseline import SETTINGS, write_data_yaml


def test_settings_cover_pretraining_and_resolution():
    assert {(s.weights, s.imgsz) for s in SETTINGS} == {
        ("yolov8n.pt", 640), ("yolov8n.pt", 256), ("yolov8n.yaml", 256), ("yolov8n.yaml", 384)}
    assert len({s.slug for s in SETTINGS}) == len(SETTINGS)


def test_data_yaml_uses_absolute_split_paths(synthetic_root, tmp_path):
    text = write_data_yaml(str(synthetic_root), tmp_path / "data.yaml").read_text()
    assert f"val: {synthetic_root / 'valid' / 'images'}" in text
    assert "nc: 1" in text


@pytest.mark.skipif(os.environ.get("RUN_YOLO") != "1", reason="set RUN_YOLO=1 (needs ultralytics)")
def test_yolo_smoke_run_scores_with_our_evaluator(synthetic_root, tmp_path):
    pytest.importorskip("ultralytics")
    from jersey_detect.cli import main

    out = tmp_path / "yolo"
    code = main(["yolo", "--data", str(synthetic_root), "--out", str(out), "--seeds", "0", "--smoke"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["errors"] == [], summary["errors"]
    assert code == 0
    (res,) = summary["results"].values()
    assert {"val", "test", "train"} <= set(res)
    assert "ci95" in res["test"]
    assert "| YOLOv8n" in (out / "results.md").read_text()


def test_unknown_only_setting_is_rejected(synthetic_root, tmp_path):
    from jersey_detect.cli import main

    with pytest.raises(SystemExit):
        main(["yolo", "--data", str(synthetic_root), "--out", str(tmp_path), "--only", "nope"])
