import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from jersey_detect.cli import main  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"


def test_full_pipeline_on_synthetic_data(synthetic_root, tmp_path):
    out = tmp_path / "results"
    code = main(["run", "--data", str(synthetic_root), "--out", str(out), "--config", str(CONFIG)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["errors"] == [], summary["errors"]
    assert code == 0
    assert {"Mean-box baseline", "v1 CNN regressor (1 epochs)",
            "JerseyNet, from scratch (seed 0)", "JerseyNet, from scratch (seed 1)"} <= set(summary["results"])
    test = summary["results"]["JerseyNet, from scratch (seed 0)"]["test"]
    assert set(test["ci95"]) == {"ap50_95", "ap50", "first_box_mean_iou"}
    assert summary["jersey_net_over_seeds"]["seeds"] == [0, 1]
    for name in ("results.md", "dataset.json", "leakage.json", "environment.json", "run.log"):
        assert (out / name).exists()
    assert (out / "runs" / "jersey_net_seed0" / "test_predictions.json").exists()
    assert "| JerseyNet" in (out / "results.md").read_text()


def test_missing_dataset_is_reported_not_raised(tmp_path):
    code = main(["run", "--data", str(tmp_path / "nope"), "--out", str(tmp_path / "o"),
                 "--config", str(CONFIG)])
    summary = json.loads((tmp_path / "o" / "summary.json").read_text())
    assert code == 1 and summary["errors"][0]["stage"] == "load_dataset"


def test_score_command_reads_yolo_predictions(synthetic_root, tmp_path, capsys):
    from jersey_detect.data import load_dataset

    pred = tmp_path / "pred"
    pred.mkdir()
    for r in load_dataset(synthetic_root)["test"]:
        label = Path(r.path).parent.parent / "labels" / f"{r.stem}.txt"
        if label.exists():
            rows = [line + " 0.9" for line in label.read_text().splitlines() if line.strip()]
            (pred / f"{r.stem}.txt").write_text("\n".join(rows))
    assert main(["score", "--data", str(synthetic_root), "--predictions", str(pred)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["ap50_95"] == pytest.approx(1.0, abs=1e-4)
