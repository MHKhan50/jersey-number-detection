import numpy as np
import pytest
from PIL import Image

from jersey_detect.transforms import (
    centernet_targets,
    gaussian_radius,
    keep_visible,
    letterbox,
    to_array,
)


def test_letterbox_keeps_aspect_and_round_trips_boxes():
    img = Image.new("RGB", (200, 100))
    canvas, lb = letterbox(img, 128)
    assert canvas.size == (128, 128)
    assert lb.scale == pytest.approx(0.64)
    assert lb.pad_x == 0 and lb.pad_y == 32
    boxes = np.array([[0.1, 0.2, 0.5, 0.9]], dtype=np.float32)
    px = lb.to_input(boxes)
    np.testing.assert_allclose(px, [[12.8, 32 + 12.8, 64.0, 32 + 57.6]], rtol=1e-5)
    np.testing.assert_allclose(lb.to_normalised(px), boxes, atol=1e-6)


def test_letterbox_with_jitter_and_offset_moves_boxes_consistently():
    img = Image.new("RGB", (100, 100))
    canvas, lb = letterbox(img, 64, scale_jitter=1.5, offset=(0.0, 1.0))
    assert canvas.size == (64, 64)
    px = lb.to_input(np.array([[0.5, 0.5, 0.5, 0.5]]))
    assert px[0, 0] == pytest.approx(0.5 * 96 + lb.pad_x)


def test_keep_visible_drops_boxes_outside_canvas():
    boxes = np.array([[10, 10, 20, 20], [-30, 5, -2, 15], [60, 60, 80, 80]], dtype=np.float32)
    kept = keep_visible(boxes, 64)
    assert len(kept) == 1
    np.testing.assert_allclose(kept[0], [10, 10, 20, 20])


def test_to_array_is_chw_float():
    arr = to_array(Image.new("RGB", (8, 6), (255, 0, 0)))
    assert arr.shape == (3, 6, 8) and arr.dtype == np.float32


def test_targets_peak_at_centre():
    boxes = np.array([[16, 16, 32, 40]], dtype=np.float32)  # centre (24, 28) -> cell (6, 7)
    t = centernet_targets(boxes, 64, 4, 8)
    assert t["heatmap"].shape == (1, 16, 16)
    assert t["heatmap"][0, 7, 6] == 1.0
    assert (t["heatmap"] == 1.0).sum() == 1
    assert t["index"][0] == 7 * 16 + 6
    np.testing.assert_allclose(t["wh"][0], [4, 6])
    np.testing.assert_allclose(t["offset"][0], [0, 0])
    assert t["mask"].sum() == 1


def test_targets_handle_box_at_edge_and_max_objects():
    boxes = np.array([[60, 60, 64, 64]] * 10, dtype=np.float32)
    t = centernet_targets(boxes, 64, 4, 4)
    assert t["mask"].sum() == 4
    assert t["index"].max() < 16 * 16


def test_gaussian_radius_positive():
    assert gaussian_radius(10, 10) > 0
