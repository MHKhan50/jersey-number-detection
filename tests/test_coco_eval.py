import numpy as np
import pytest

from jersey_detect.coco_eval import evaluate

BOX = np.array([[0.1, 0.1, 0.4, 0.4]])


def test_perfect_detections():
    res = evaluate([BOX, BOX], [BOX, BOX], [[0.9], [0.8]])
    assert res["ap50_95"] == pytest.approx(1.0)
    assert res["ar"] == pytest.approx(1.0)


def test_no_detections_scores_zero():
    res = evaluate([BOX], [np.zeros((0, 4))], [np.zeros(0)])
    assert res["ap50"] == 0.0 and res["num_gt"] == 1


def test_false_positive_on_negative_image_ranked_first_lowers_ap():
    clean = evaluate([BOX, np.zeros((0, 4))], [BOX, np.zeros((0, 4))], [[0.5], []])
    noisy = evaluate([BOX, np.zeros((0, 4))], [BOX, BOX], [[0.5], [0.9]])
    assert clean["ap50"] == pytest.approx(1.0)
    assert noisy["ap50"] == pytest.approx(0.5, abs=0.01)


def test_low_scored_false_positive_does_not_change_ap():
    res = evaluate([BOX, np.zeros((0, 4))], [BOX, BOX], [[0.9], [0.1]])
    assert res["ap50"] == pytest.approx(1.0)


def test_half_recall():
    two = np.array([[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.9, 0.9]])
    res = evaluate([two], [BOX], [[0.9]])
    assert res["ap50"] == pytest.approx(51 / 101)  # precision 1 up to recall 0.5
    assert res["ar"] == pytest.approx(0.5)


def test_duplicate_detection_counts_once():
    res = evaluate([BOX], [np.vstack([BOX, BOX])], [[0.9, 0.8]])
    assert res["ap50"] == pytest.approx(1.0)  # the duplicate comes after full recall


def test_iou_threshold_boundary():
    shifted = np.array([[0.1, 0.1, 0.4, 0.55]])  # IoU = 0.09 / 0.135 = 0.667
    res = evaluate([BOX], [shifted], [[0.9]])
    assert res["ap50"] == pytest.approx(1.0)
    assert res["ap75"] == 0.0


def test_matches_pycocotools_on_random_cases():
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(0)
    for trial in range(5):
        gts, dets, scores = [], [], []
        for _ in range(30):
            n_gt = int(rng.integers(0, 4))
            g = []
            for _ in range(n_gt):
                x, y = rng.uniform(0, 0.8, 2)
                w, h = rng.uniform(0.03, 0.2, 2)
                g.append([x, y, x + w, y + h])
            g = np.asarray(g).reshape(-1, 4)
            d = [b + rng.normal(0, 0.02, 4) for b in g if rng.uniform() < 0.8]
            for _ in range(int(rng.integers(0, 3))):
                x, y = rng.uniform(0, 0.8, 2)
                d.append([x, y, x + 0.1, y + 0.1])
            d = np.asarray(d).reshape(-1, 4)
            d[:, 2:] = np.maximum(d[:, 2:], d[:, :2] + 1e-3)
            gts.append(g)
            dets.append(d)
            scores.append(rng.uniform(0, 1, len(d)))
        ours = evaluate(gts, dets, scores)

        scale = 1000.0  # COCO works in pixels; areas only matter for area ranges, unused here
        images, anns, results = [], [], []
        for i, (g, d, s) in enumerate(zip(gts, dets, scores, strict=True)):
            images.append({"id": i + 1, "width": 1000, "height": 1000})
            for b in g * scale:
                w, h = b[2] - b[0], b[3] - b[1]
                anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": 1,
                             "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": 0})
            for b, sc in zip(d * scale, s, strict=True):
                results.append({"image_id": i + 1, "category_id": 1,
                                "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(sc)})
        coco = COCO()
        coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "number"}]}
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(results), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        assert ours["ap50_95"] == pytest.approx(ev.stats[0], abs=1e-6), trial
        assert ours["ap50"] == pytest.approx(ev.stats[1], abs=1e-6), trial
        assert ours["ap75"] == pytest.approx(ev.stats[2], abs=1e-6), trial
        assert ours["ar"] == pytest.approx(ev.stats[8], abs=1e-6), trial
