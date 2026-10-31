"""COCO-style average precision for a single class, written in NumPy.

It follows the matching and accumulation rules of ``pycocotools.COCOeval`` for one category
and the ``all`` area range: detections are ranked by score inside each image and capped at
``max_dets``, each one is matched to the unmatched ground-truth box with the highest IoU at or
above the threshold, precision is made monotone, and it is sampled at 101 recall points. The
test suite checks the numbers against ``pycocotools`` when it is installed.

Images without ground truth are part of the evaluation: a detection on such an image is a
false positive.
"""

from __future__ import annotations

import numpy as np

from jersey_detect.boxes import iou_matrix

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def _match_image(gt: np.ndarray, det: np.ndarray, scores: np.ndarray, max_dets: int):
    order = np.argsort(-scores, kind="mergesort")[:max_dets]
    det, scores = det[order], scores[order]
    ious = iou_matrix(det, gt)
    matched = np.zeros((len(IOU_THRESHOLDS), len(det)), dtype=bool)
    for ti, t in enumerate(IOU_THRESHOLDS):
        gt_taken = np.zeros(len(gt), dtype=bool)
        for d in range(len(det)):
            best, best_g = min(t, 1 - 1e-10), -1
            for g in range(len(gt)):
                if gt_taken[g] or ious[d, g] < best:
                    continue
                best, best_g = ious[d, g], g
            if best_g >= 0:
                gt_taken[best_g] = True
                matched[ti, d] = True
    return scores, matched


def match_images(gts, dets, scores, max_dets: int = 100) -> list[tuple]:
    """Per-image matching results ``(scores, matched[T, D], n_gt)``.

    Matching only depends on the image itself, so bootstrap resampling can reuse it.
    """
    if not (len(gts) == len(dets) == len(scores)):
        raise ValueError("gts, dets and scores must have one entry per image")
    out = []
    for gt, det, sc in zip(gts, dets, scores, strict=True):
        gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
        det = np.asarray(det, dtype=np.float64).reshape(-1, 4)
        sc = np.asarray(sc, dtype=np.float64).reshape(-1)
        if len(det) != len(sc):
            raise ValueError("each detection needs exactly one score")
        s, m = _match_image(gt, det, sc, max_dets)
        out.append((s, m, len(gt)))
    return out


def accumulate(matches: list[tuple]) -> dict[str, float]:
    n_gt = int(sum(m[2] for m in matches))
    out = {"ap50_95": 0.0, "ap50": 0.0, "ap75": 0.0, "ar": 0.0, "num_gt": n_gt,
           "num_det": int(sum(len(m[0]) for m in matches))}
    if n_gt == 0 or out["num_det"] == 0:
        return out
    scores_cat = np.concatenate([m[0] for m in matches])
    matched_cat = np.concatenate([m[1] for m in matches], axis=1)
    order = np.argsort(-scores_cat, kind="mergesort")
    matched_cat = matched_cat[:, order]

    precision = np.zeros((len(IOU_THRESHOLDS), len(RECALL_POINTS)))
    recall = np.zeros(len(IOU_THRESHOLDS))
    eps = np.spacing(1)
    for ti in range(len(IOU_THRESHOLDS)):
        tp = np.cumsum(matched_cat[ti]).astype(np.float64)
        fp = np.cumsum(~matched_cat[ti]).astype(np.float64)
        rc = tp / n_gt
        pr = tp / (fp + tp + eps)
        recall[ti] = rc[-1]
        pr = np.maximum.accumulate(pr[::-1])[::-1]  # precision envelope
        idx = np.searchsorted(rc, RECALL_POINTS, side="left")
        valid = idx < len(pr)
        precision[ti, valid] = pr[idx[valid]]

    out["ap50_95"] = float(precision.mean())
    out["ap50"] = float(precision[0].mean())
    out["ap75"] = float(precision[5].mean())
    out["ar"] = float(recall.mean())
    return out


def evaluate(gts, dets, scores, max_dets: int = 100) -> dict[str, float]:
    """Return AP50:95, AP50, AP75 and AR (at ``max_dets``) over all images.

    ``gts[i]`` is a (G, 4) array of boxes for image ``i``, ``dets[i]`` (D, 4) and
    ``scores[i]`` (D,). Boxes are [x1, y1, x2, y2] in any consistent unit.
    """
    return accumulate(match_images(gts, dets, scores, max_dets))
