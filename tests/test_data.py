import io

import numpy as np
from PIL import Image

from jersey_detect.data import IMAGE_SIZE, load_samples, parse_example
from jersey_detect.tfrecord import encode_example, write_record


def _jpeg_bytes(size=(60, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def _example(with_box: bool) -> bytes:
    feats = {"image/encoded": ("bytes", [_jpeg_bytes()])}
    if with_box:
        feats.update({
            "image/object/bbox/ymin": ("float", [0.2, 0.9]),
            "image/object/bbox/xmin": ("float", [0.1, 0.8]),
            "image/object/bbox/ymax": ("float", [0.6, 1.0]),
            "image/object/bbox/xmax": ("float", [0.5, 1.0]),
            "image/object/class/label": ("int64", [1, 1]),
        })
    return encode_example(feats)


def test_parse_keeps_first_box_in_ymin_xmin_ymax_xmax_order():
    sample = parse_example(_example(with_box=True))
    assert sample is not None
    np.testing.assert_allclose(sample.box, [0.2, 0.1, 0.6, 0.5], rtol=1e-5)
    assert sample.label == 1.0
    assert sample.image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert sample.image.dtype == np.float32
    assert 0.0 <= sample.image.min() and sample.image.max() <= 1.0


def test_example_without_box_is_skipped():
    assert parse_example(_example(with_box=False)) is None


def test_load_samples_reads_file(tmp_path):
    path = tmp_path / "train.tfrecord"
    with open(path, "wb") as fh:
        write_record(fh, _example(with_box=True))
        write_record(fh, _example(with_box=False))
    samples = load_samples(str(path))
    assert len(samples) == 1
