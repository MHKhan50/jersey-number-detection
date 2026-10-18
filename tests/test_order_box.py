import pytest

torch = pytest.importorskip("torch")

from jersey_detect.model import order_box  # noqa: E402


def test_inverted_box_is_reordered():
    inverted = torch.tensor([[0.7, 0.6, 0.3, 0.4]])  # ymin > ymax and xmin > xmax
    ordered = order_box(inverted)
    assert torch.allclose(ordered, torch.tensor([[0.3, 0.4, 0.7, 0.6]]))


def test_correct_box_is_unchanged():
    box = torch.tensor([[0.1, 0.2, 0.5, 0.9]])
    assert torch.allclose(order_box(box), box)
