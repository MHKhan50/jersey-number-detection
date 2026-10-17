import pytest

from jersey_detect.metrics import binary_accuracy, iou, mean_iou


def test_identical_boxes_have_iou_one():
    assert iou([0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5]) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero():
    assert iou([0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.9, 0.9]) == 0.0


def test_half_overlap():
    # two unit-height boxes overlapping by half their width
    assert iou([0, 0, 1, 1], [0, 0.5, 1, 1.5]) == pytest.approx(1 / 3)


def test_mean_iou_checks_lengths():
    with pytest.raises(ValueError):
        mean_iou([[0, 0, 1, 1]], [])


def test_binary_accuracy_threshold():
    assert binary_accuracy([0.9, 0.2, 0.6, 0.4], [1, 0, 0, 1]) == pytest.approx(0.5)
