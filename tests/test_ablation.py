import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jersey_detect.cli import main  # noqa: E402
from jersey_detect.data import load_dataset  # noqa: E402
from jersey_detect.dataset import TrainDataset  # noqa: E402
from jersey_detect.detector import detection_loss, giou  # noqa: E402
from jersey_detect.transforms import centernet_targets  # noqa: E402


def test_giou_known_values():
    a = torch.tensor([0.0, 0.0, 2.0, 2.0])
    assert giou(a, a).item() == pytest.approx(1.0, abs=1e-5)
    far = torch.tensor([3.0, 0.0, 5.0, 2.0])  # disjoint: IoU 0, enclosing 5x2, union 8
    assert giou(a, far).item() == pytest.approx(0 - (10 - 8) / 10, abs=1e-5)


def test_giou_loss_is_zero_for_perfect_boxes_and_positive_otherwise():
    t = centernet_targets(np.array([[10.0, 12.0, 30.0, 40.0]], np.float32), 64, 4, 8)
    batch = {k: torch.from_numpy(v)[None] for k, v in t.items()}
    out = {"heatmap": torch.zeros(1, 1, 16, 16), "wh": torch.zeros(1, 2, 16, 16),
           "offset": torch.zeros(1, 2, 16, 16)}
    i = int(batch["index"][0, 0])
    out["wh"].view(1, 2, -1)[0, :, i] = batch["wh"][0, 0]
    out["offset"].view(1, 2, -1)[0, :, i] = batch["offset"][0, 0]
    assert detection_loss(out, batch, giou_weight=1.0)["giou"].item() == pytest.approx(0.0, abs=1e-5)
    out["wh"].view(1, 2, -1)[0, :, i] = batch["wh"][0, 0] * 0.5
    assert detection_loss(out, batch, giou_weight=1.0)["giou"].item() > 0.1
    assert "giou" not in detection_loss(out, batch)


def test_mosaic_samples_have_valid_boxes_inside_the_canvas(synthetic_root):
    recs = load_dataset(synthetic_root)["train"]
    ds = TrainDataset(recs, 128, mosaic=1.0, seed=3)
    total = 0
    for i in range(12):
        item = ds[i]
        assert item["image"].shape == (3, 128, 128)
        assert int(item["index"].max()) < (128 // 4) ** 2
        assert item["heatmap"].max() <= 1.0
        total += int(item["mask"].sum())
    assert total > 12  # mosaics carry boxes from several images


def test_ablation_smoke_run(synthetic_root, tmp_path):
    out = tmp_path / "abl"
    code = main(["ablate", "--data", str(synthetic_root), "--out", str(out), "--smoke"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["errors"] == [], summary["errors"]
    assert code == 0
    best = summary["best_variant"]
    assert f"{best} (seed 1)" in summary["runs"]
    assert summary["best_over_seeds"]["seeds"] == [0, 1]
    assert "Best variant on validation" in (out / "results.md").read_text()
