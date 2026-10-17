import pytest

torch = pytest.importorskip("torch")

from jersey_detect.tfrecord import write_record  # noqa: E402
from jersey_detect.train import TrainConfig, train  # noqa: E402
from tests.test_data import _example  # noqa: E402


@pytest.mark.parametrize("box_loss", ["l1", "giou"])
@pytest.mark.parametrize("select_by", ["loss", "iou"])
def test_one_epoch_runs_with_each_option(tmp_path, box_loss, select_by):
    train_path = tmp_path / "train.tfrecord"
    val_path = tmp_path / "valid.tfrecord"
    for path in (train_path, val_path):
        with open(path, "wb") as fh:
            for _ in range(4):
                write_record(fh, _example(with_box=True))
    cfg = TrainConfig(
        train_record=str(train_path), val_record=str(val_path),
        out_dir=str(tmp_path / "ckpt"), epochs=1, batch_size=2,
        box_loss=box_loss, select_by=select_by,
    )
    final = train(cfg)
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert "mean_iou_positive" in final
