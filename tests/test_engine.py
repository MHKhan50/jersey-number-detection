import json

import pytest

torch = pytest.importorskip("torch")

from jersey_detect.engine import DetectorConfig, load_detector, train_detector  # noqa: E402

SMALL = dict(image_size=128, batch_size=8, num_workers=0, eval_every=1, warmup_epochs=1, amp=False)


def test_train_writes_checkpoints_and_history(synthetic_root, tmp_path):
    cfg = DetectorConfig(data_root=str(synthetic_root), out_dir=str(tmp_path / "r"), epochs=2, **SMALL)
    info = train_detector(cfg, log=lambda m: None)
    assert info["epochs_run"] == 2 and info["stopped"] == "completed"
    for name in ("best.pt", "last.pt", "config.json", "history.jsonl", "train_summary.json"):
        assert (tmp_path / "r" / name).exists()
    lines = (tmp_path / "r" / "history.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "val" in json.loads(lines[-1])
    model, state = load_detector(tmp_path / "r" / "best.pt", torch.device("cpu"))
    assert state["config"]["image_size"] == 128


def test_resume_continues_from_last_epoch(synthetic_root, tmp_path):
    out = str(tmp_path / "r")
    train_detector(DetectorConfig(data_root=str(synthetic_root), out_dir=out, epochs=3, **SMALL),
                   log=lambda m: None)
    state = torch.load(tmp_path / "r" / "last.pt", weights_only=False)
    state["epoch"] = 1  # pretend the session died after epoch 1
    state["history"] = state["history"][:1]
    torch.save(state, tmp_path / "r" / "last.pt")
    logs = []
    info = train_detector(DetectorConfig(data_root=str(synthetic_root), out_dir=out, epochs=3, **SMALL),
                          log=logs.append)
    assert any("resumed from epoch 1" in m for m in logs)
    assert info["epochs_run"] == 3


def test_changed_settings_start_a_new_run(synthetic_root, tmp_path):
    out = str(tmp_path / "r")
    train_detector(DetectorConfig(data_root=str(synthetic_root), out_dir=out, epochs=1, **SMALL),
                   log=lambda m: None)
    logs = []
    train_detector(DetectorConfig(data_root=str(synthetic_root), out_dir=out, epochs=1, seed=5, **SMALL),
                   log=logs.append)
    assert any("different settings" in m for m in logs)


def test_time_budget_stops_cleanly(synthetic_root, tmp_path):
    cfg = DetectorConfig(data_root=str(synthetic_root), out_dir=str(tmp_path / "r"), epochs=50,
                         max_hours=1e-6, **SMALL)
    info = train_detector(cfg, log=lambda m: None)
    assert info["stopped"] == "time_budget" and info["epochs_run"] == 1
    assert (tmp_path / "r" / "best.pt").exists()


def test_unknown_setting_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        DetectorConfig.from_dict({"data_root": "x", "out_dir": "y", "epoch": 3})
