import pytest

torch = pytest.importorskip("torch")

from jersey_detect.tfrecord import encode_example, write_record  # noqa: E402
from jersey_detect.train import TrainConfig, train  # noqa: E402
from tests.test_data import _example  # noqa: E402


def test_one_epoch_runs_and_writes_checkpoint(tmp_path):
    train_path = tmp_path / "train.tfrecord"
    val_path = tmp_path / "valid.tfrecord"
    for path in (train_path, val_path):
        with open(path, "wb") as fh:
            for _ in range(4):
                write_record(fh, _example(with_box=True))
    cfg = TrainConfig(
        train_record=str(train_path), val_record=str(val_path),
        out_dir=str(tmp_path / "ckpt"), epochs=1, batch_size=2,
    )
    final = train(cfg)
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert "box_mse" in final and "class_accuracy" in final
