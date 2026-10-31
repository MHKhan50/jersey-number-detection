import numpy as np
import pytest
from PIL import Image

from jersey_detect.boxes import iou, iou_matrix, yolo_to_xyxy
from jersey_detect.data import find_split_dirs, load_dataset, load_split, split_stats


def _split(root, items):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for name, label in items.items():
        Image.new("RGB", (60, 40)).save(root / "images" / f"{name}.jpg")
        if label is not None:
            (root / "labels" / f"{name}.txt").write_text(label)


def test_yolo_to_xyxy():
    np.testing.assert_allclose(yolo_to_xyxy(0.5, 0.5, 0.2, 0.4), [0.4, 0.3, 0.6, 0.7])


def test_iou_known_values():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert iou([0, 0, 1, 1], [0.5, 0, 1.5, 1]) == pytest.approx(1 / 3)
    assert iou([0, 0, 0.1, 0.1], [0.5, 0.5, 0.9, 0.9]) == 0.0
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_all_boxes_kept_and_negatives_kept(tmp_path):
    _split(tmp_path, {"two": "0 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.1 0.1\n", "empty": "", "nofile": None})
    recs = {r.stem: r for r in load_split(tmp_path)}
    assert len(recs) == 3
    assert recs["two"].boxes.shape == (2, 4)
    np.testing.assert_allclose(recs["two"].boxes[0], [0.4, 0.3, 0.6, 0.7], rtol=1e-6)
    assert not recs["empty"].labelled and recs["empty"].has_label_file
    assert not recs["nofile"].labelled and not recs["nofile"].has_label_file


def test_boxes_are_clipped_and_degenerate_rows_dropped(tmp_path):
    _split(tmp_path, {"a": "0 0.95 0.5 0.2 0.2\n0 0.5 0.5 0 0.1\n"})
    (rec,) = load_split(tmp_path)
    assert rec.boxes.shape == (1, 4)
    assert rec.boxes.max() <= 1.0


def test_malformed_label_line_raises(tmp_path):
    _split(tmp_path, {"a": "0 0.5 0.5\n"})
    with pytest.raises(ValueError):
        load_split(tmp_path)


def test_val_folder_alias(tmp_path):
    for name in ("train", "val", "test"):
        _split(tmp_path / name, {"x": "0 0.5 0.5 0.1 0.1\n"})
    assert find_split_dirs(tmp_path)["val"].name == "val"


def test_missing_split_is_reported(tmp_path):
    _split(tmp_path / "train", {"x": None})
    with pytest.raises(FileNotFoundError, match="missing"):
        find_split_dirs(tmp_path)


def test_synthetic_dataset_stats(synthetic_root):
    splits = load_dataset(synthetic_root)
    stats = split_stats(splits["test"])
    assert stats["images"] == 12
    assert 0 < stats["labelled_images"] < stats["images"]
    assert stats["images_without_label_file"] > 0
    assert stats["boxes"] >= stats["labelled_images"]


def test_find_dataset_searches_nested_folders(tmp_path):
    from jersey_detect.prepare import find_dataset

    root = tmp_path / "input" / "datasets" / "user" / "jersey" / "jersey-2"
    for name in ("train", "valid", "test"):
        _split(root / name, {"x": "0 0.5 0.5 0.1 0.1\n"})
    assert find_dataset([tmp_path / "input"]) == root
    assert find_dataset([tmp_path / "missing"]) is None
