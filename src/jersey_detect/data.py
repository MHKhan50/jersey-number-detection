"""Dataset parsing: TFRecord examples to (image, box, label) samples.

Matches the original notebook: the image is resized to 128x128, the first
bounding box is kept (boxes are stored as [ymin, xmin, ymax, xmax], normalised
to 0-1), and images without any box are skipped.
"""

from __future__ import annotations

import io
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


def load_samples(path: str) -> list[Sample]:
    """Load every usable example from a TFRecord file into memory."""
    samples = []
    for payload in iter_records(path):
        sample = parse_example(payload)
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise ValueError(f"No usable examples found in {path}")
    return samples
