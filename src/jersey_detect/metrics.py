"""Evaluation protocols used in the paper.

1. COCO-style AP over every annotated box (``coco_eval``). This is the primary metric.
2. The first-box protocol of the earlier results: on each labelled image, the highest-scoring
   prediction is compared with the first annotated box. It is kept so the new numbers can be
   compared with the v1 tables. An image with no prediction scores IoU 0.

3. AP by number size (``size_metrics``): the same AP restricted to small, medium and large
   numbers, measured by the box side ``sqrt(w * h)`` in pixels of the original 128 px images.

``bootstrap_ci`` resamples images to give a 95% interval for any metric, because the test split
is small (about 150 images).
"""

from __future__ import annotations

import numpy as np

from jersey_detect import coco_eval
from jersey_detect.boxes import iou


def first_box_ious(gts, dets, scores) -> list[float]:
    out = []
    for gt, det, sc in zip(gts, dets, scores, strict=True):
        gt = np.asarray(gt).reshape(-1, 4)
        if len(gt) == 0:
            continue
        det = np.asarray(det).reshape(-1, 4)
        if len(det) == 0:
            out.append(0.0)
            continue
        top = det[int(np.argmax(np.asarray(sc).reshape(-1)))]
        out.append(iou(top, gt[0]))
    return out


def first_box_metrics(gts, dets, scores) -> dict[str, float]:
    ious = first_box_ious(gts, dets, scores)
    if not ious:
        return {"first_box_mean_iou": 0.0, "first_box_iou_ge_0_5": 0.0, "scored_images": 0}
    arr = np.asarray(ious)
    return {
        "first_box_mean_iou": float(arr.mean()),
        "first_box_iou_ge_0_5": float((arr >= 0.5).mean()),
        "scored_images": int(len(arr)),
    }


def all_metrics(gts, dets, scores) -> dict[str, float]:
    out = coco_eval.evaluate(gts, dets, scores)
    out.update(first_box_metrics(gts, dets, scores))
    out["images"] = len(gts)
    return out


# Box side in pixels of the original 128 x 128 images. The median number is about 10 px.
NATIVE_SIDE = 128
SIZE_BINS = {"small": (0.0, 8.0), "medium": (8.0, 16.0), "large": (16.0, 1e5)}


def size_metrics(gts, dets, scores, native_side: int = NATIVE_SIDE) -> dict[str, dict]:
    """AP50:95, AP50 and the number of boxes for each size bin (pycocotools area-range rules)."""
    out = {}
    for name, (lo, hi) in SIZE_BINS.items():
        res = coco_eval.evaluate(gts, dets, scores, side_range=(lo / native_side, hi / native_side))
        out[name] = {"ap50_95": res["ap50_95"], "ap50": res["ap50"], "ar": res["ar"],
                     "num_gt": res["num_gt"]}
    return out


def bootstrap_ci(gts, dets, scores, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap over images for AP50:95, AP50 and first-box mean IoU.

    Per-image matching is computed once and reused, so 1,000 resamples take seconds.
    """
    matches = coco_eval.match_images(gts, dets, scores)
    fb = np.asarray([
        (first_box_ious([g], [d], [s]) or [np.nan])[0] for g, d, s in zip(gts, dets, scores, strict=True)
    ])
    rng = np.random.default_rng(seed)
    n = len(matches)
    draws = {"ap50_95": [], "ap50": [], "first_box_mean_iou": []}
    if n == 0:
        return {k: [0.0, 0.0] for k in draws}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        res = coco_eval.accumulate([matches[i] for i in idx])
        draws["ap50_95"].append(res["ap50_95"])
        draws["ap50"].append(res["ap50"])
        sample = fb[idx]
        sample = sample[~np.isnan(sample)]
        draws["first_box_mean_iou"].append(float(sample.mean()) if len(sample) else 0.0)
    return {k: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
            for k, v in draws.items()}
