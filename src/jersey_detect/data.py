"""Read YOLO-format splits (Roboflow "YOLOv8" export).

Layout::

    <root>/train/images/*.jpg   <root>/train/labels/*.txt
    <root>/valid/images/...     <root>/valid/labels/...   (``val`` is accepted too)
    <root>/test/images/...      <root>/test/labels/...

Each label line is ``class cx cy w h`` normalised to 0-1. An image with no label file, or an
empty one, has no jersey number in it. Such images are kept: they count as negatives in the
COCO-style evaluation, where any box predicted on them is a false positive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jersey_detect.boxes import clip01, yolo_to_xyxy

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_NAMES = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}


@dataclass
class ImageRecord:
    path: str
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    has_label_file: bool = False

    @property
    def stem(self) -> str:
        return Path(self.path).stem

    @property
    def labelled(self) -> bool:
        return len(self.boxes) > 0


def _read_label_file(path: Path) -> np.ndarray:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{path}:{line_no}: expected 'class cx cy w h', got {line!r}")
        cx, cy, w, h = (float(v) for v in parts[1:5])
        if w <= 0 or h <= 0:
            continue  # degenerate annotation, nothing to score against
        rows.append(yolo_to_xyxy(cx, cy, w, h))
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return clip01(np.asarray(rows, dtype=np.float32))


def load_split(split_dir: str | os.PathLike) -> list[ImageRecord]:
    """Load every image of one split with its boxes, sorted by file name."""
    split_dir = Path(split_dir)
    images_dir, labels_dir = split_dir / "images", split_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No images/ folder in {split_dir}")
    records = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES or img.name.startswith("._"):
            continue  # "._name.jpg" files are macOS metadata, not images
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            records.append(ImageRecord(str(img), _read_label_file(label), True))
        else:
            records.append(ImageRecord(str(img)))
    if not records:
        raise ValueError(f"No images found in {images_dir}")
    return records


def find_split_dirs(root: str | os.PathLike) -> dict[str, Path]:
    """Return {'train': ..., 'val': ..., 'test': ...} for a dataset root."""
    root = Path(root)
    found = {}
    for key, names in SPLIT_NAMES.items():
        for name in names:
            if (root / name / "images").is_dir():
                found[key] = root / name
                break
    missing = [k for k in SPLIT_NAMES if k not in found]
    if missing:
        raise FileNotFoundError(f"{root} is missing split(s) {missing} (need <split>/images)")
    return found


def load_dataset(root: str | os.PathLike) -> dict[str, list[ImageRecord]]:
    return {key: load_split(path) for key, path in find_split_dirs(root).items()}


def split_stats(records: list[ImageRecord]) -> dict:
    """Counts and box-size statistics used in the dataset table of the paper."""
    boxes = [r.boxes for r in records if r.labelled]
    all_boxes = np.concatenate(boxes) if boxes else np.zeros((0, 4))
    w = all_boxes[:, 2] - all_boxes[:, 0]
    h = all_boxes[:, 3] - all_boxes[:, 1]
    rel = np.sqrt(w * h) if len(all_boxes) else np.zeros(0)

    def q(v, p):
        return float(np.quantile(v, p)) if len(v) else None

    return {
        "images": len(records),
        "labelled_images": sum(r.labelled for r in records),
        "images_without_label_file": sum(not r.has_label_file for r in records),
        "boxes": int(len(all_boxes)),
        "max_boxes_per_image": int(max((len(r.boxes) for r in records), default=0)),
        "box_rel_size_median": q(rel, 0.5),
        "box_rel_size_p10": q(rel, 0.1),
        "box_rel_size_p90": q(rel, 0.9),
    }
