"""Box and classification metrics. Pure Python and NumPy, no deep-learning framework needed."""

from __future__ import annotations

from collections.abc import Sequence


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes given as [ymin, xmin, ymax, xmax]."""
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter = inter_h * inter_w
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mean_iou(pred_boxes: Sequence[Sequence[float]], true_boxes: Sequence[Sequence[float]]) -> float:
    """Mean IoU over matching pairs."""
    if len(pred_boxes) != len(true_boxes):
        raise ValueError("pred_boxes and true_boxes must have the same length")
    if not pred_boxes:
        return 0.0
    return sum(iou(p, t) for p, t in zip(pred_boxes, true_boxes, strict=True)) / len(pred_boxes)


def binary_accuracy(
    probs: Sequence[float], labels: Sequence[float], threshold: float = 0.5
) -> float:
    """Fraction of predictions that match the 0/1 label after thresholding."""
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have the same length")
    if not probs:
        return 0.0
    hits = sum(int((p >= threshold) == (y >= 0.5)) for p, y in zip(probs, labels, strict=True))
    return hits / len(probs)
