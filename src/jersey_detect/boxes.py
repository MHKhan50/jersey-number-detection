"""Box utilities. Every box in this package is [x1, y1, x2, y2].

Ground truth and predictions are stored normalised to the original image (0-1), so results do
not depend on the input size a model was trained at.
"""

from __future__ import annotations

import numpy as np


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Convert a YOLO (centre x, centre y, width, height) box to [x1, y1, x2, y2]."""
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def clip01(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes, 0.0, 1.0)


def area(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N, 4) and b (M, 4). Returns (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area(a)[:, None] + area(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def iou(a, b) -> float:
    return float(iou_matrix(np.asarray(a)[None], np.asarray(b)[None])[0, 0])
