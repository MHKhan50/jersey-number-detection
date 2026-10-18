"""Dataset parsing: TFRecord examples to (image, box, label) samples.

Matches the original notebook: the image is resized to 128x128, the first
bounding box is kept (boxes are stored as [ymin, xmin, ymax, xmax], normalised
to 0-1), and images without any box are skipped.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from jersey_detect.tfrecord import decode_example, iter_records

IMAGE_SIZE = 128


@dataclass(frozen=True)
class Sample:
    image: np.ndarray  # float32, shape (128, 128, 3), values in [0, 1]
    box: np.ndarray  # float32, shape (4,), [ymin, xmin, ymax, xmax] in [0, 1]
    label: float  # 1.0 when the labelled object (jersey number '0') is present


def _values(features: dict, key: str) -> list:
    _, values = features.get(key, ("empty", []))
    return values


def parse_example(payload: bytes) -> Sample | None:
    """Parse one serialized example. Returns None when the example has no box."""
    features = decode_example(payload)
    encoded = _values(features, "image/encoded")
    if not encoded:
        raise ValueError("Example has no image/encoded feature")

    xmin = _values(features, "image/object/bbox/xmin")
    xmax = _values(features, "image/object/bbox/xmax")
    ymin = _values(features, "image/object/bbox/ymin")
    ymax = _values(features, "image/object/bbox/ymax")
    labels = _values(features, "image/object/class/label")
    if not (xmin and xmax and ymin and ymax and labels):
        return None

    box = np.array([ymin[0], xmin[0], ymax[0], xmax[0]], dtype=np.float32)
    label = 1.0 if labels[0] == 1 else 0.0

    image = Image.open(io.BytesIO(encoded[0])).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return Sample(image=arr, box=box, label=label)


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _load_yolo_sample(image_path: str, label_path: str) -> Sample | None:
    """Build one sample from a YOLO image and its label file (class cx cy w h, normalised)."""
    with open(label_path, encoding="utf-8") as fh:
        rows = [line.split() for line in fh if line.strip()]
    if not rows:
        return None  # no box, skipped as in the TFRecord path

    cx, cy, w, h = (float(v) for v in rows[0][1:5])  # first box, as in the original notebook
    box = np.array(
        [cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2], dtype=np.float32
    )  # [ymin, xmin, ymax, xmax]

    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # The YOLO export has one class ('number'), so a labelled box means a number is present.
    return Sample(image=arr, box=box, label=1.0)


def load_yolo_samples(split_dir: str) -> list[Sample]:
    """Load a YOLO-format split laid out as <split_dir>/images and <split_dir>/labels."""
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    samples = []
    for name in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in _IMAGE_SUFFIXES:
            continue
        label_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(label_path):
            continue  # unlabelled image: skipped, the same as an example without a box
        sample = _load_yolo_sample(os.path.join(images_dir, name), label_path)
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise ValueError(f"No usable YOLO examples found in {split_dir}")
    return samples


def load_samples(path: str) -> list[Sample]:
    """Load every usable example from a TFRecord file, or a YOLO split directory."""
    if os.path.isdir(path):
        return load_yolo_samples(path)
    samples = []
    for payload in iter_records(path):
        sample = parse_example(payload)
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise ValueError(f"No usable examples found in {path}")
    return samples
