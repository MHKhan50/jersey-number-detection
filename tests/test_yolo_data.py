import numpy as np
from PIL import Image

from jersey_detect.data import IMAGE_SIZE, load_samples, load_yolo_samples


def _write_split(root, items):
    """items: name -> label text, or None for an image with no label file."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for name, label in items.items():
        Image.new("RGB", (60, 40), color=(10, 200, 10)).save(root / "images" / f"{name}.jpg")
        if label is not None:
            (root / "labels" / f"{name}.txt").write_text(label)


def test_yolo_box_is_converted_to_ymin_xmin_ymax_xmax(tmp_path):
    # centre (0.5, 0.5), width 0.2, height 0.4 -> ymin 0.3, xmin 0.4, ymax 0.7, xmax 0.6
    _write_split(tmp_path, {"a": "0 0.5 0.5 0.2 0.4\n"})
    (sample,) = load_yolo_samples(str(tmp_path))
    np.testing.assert_allclose(sample.box, [0.3, 0.4, 0.7, 0.6], rtol=1e-5)
    assert sample.label == 1.0
    assert sample.image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_first_box_is_used_when_image_has_several(tmp_path):
    _write_split(tmp_path, {"a": "0 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.1 0.1\n"})
    (sample,) = load_yolo_samples(str(tmp_path))
    np.testing.assert_allclose(sample.box, [0.3, 0.4, 0.7, 0.6], rtol=1e-5)


def test_unlabelled_and_empty_label_images_are_skipped(tmp_path):
    _write_split(tmp_path, {"labelled": "0 0.5 0.5 0.2 0.4\n", "empty": "", "missing": None})
    samples = load_yolo_samples(str(tmp_path))
    assert len(samples) == 1


def test_load_samples_dispatches_directories_to_yolo_loader(tmp_path):
    _write_split(tmp_path, {"a": "0 0.5 0.5 0.2 0.4\n"})
    assert len(load_samples(str(tmp_path))) == 1
