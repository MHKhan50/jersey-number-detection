"""Sanity check that JerseyNet actually learns: a short CPU run on synthetic data must clearly
beat the mean-box baseline. Slow (about a minute), so CI runs it as a separate step."""

import json
import os

import pytest

torch = pytest.importorskip("torch")

from jersey_detect.baselines import constant_predictions, mean_box  # noqa: E402
from jersey_detect.data import load_dataset  # noqa: E402
from jersey_detect.engine import DetectorConfig, evaluate_records, load_detector, train_detector  # noqa: E402
from jersey_detect.metrics import all_metrics  # noqa: E402
from jersey_detect.synthetic import make_dataset  # noqa: E402


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run")
def test_jersey_net_learns_synthetic_numbers(tmp_path):
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    root = make_dataset(tmp_path / "d", counts=(160, 30, 30), image_size=(192, 144))
    cfg = DetectorConfig(data_root=str(root), out_dir=str(tmp_path / "r"), image_size=192, epochs=30,
                         batch_size=16, lr=2e-3, num_workers=2, eval_every=10, warmup_epochs=2, amp=False)
    train_detector(cfg, log=lambda m: None)
    model, _ = load_detector(tmp_path / "r" / "best.pt", torch.device("cpu"))
    splits = load_dataset(root)
    ours = evaluate_records(model, splits["test"], 192, torch.device("cpu"), num_workers=0)
    d, s = constant_predictions(mean_box(splits["train"]), splits["test"])
    base = all_metrics([r.boxes for r in splits["test"]], d, s)
    print("LEARNING_CHECK " + json.dumps({"jersey_net_ap50": ours["ap50"], "ap50_95": ours["ap50_95"],
                                          "mean_box_ap50": base["ap50"]}))
    assert ours["ap50"] > base["ap50"] + 0.2
