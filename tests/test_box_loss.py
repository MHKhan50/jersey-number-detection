import pytest

torch = pytest.importorskip("torch")

from jersey_detect.train import TrainConfig, giou_loss  # noqa: E402


def test_identical_boxes_have_zero_loss():
    box = torch.tensor([[0.2, 0.3, 0.6, 0.7]])
    assert giou_loss(box, box).item() == pytest.approx(0.0, abs=1e-5)


def test_disjoint_boxes_have_loss_above_one_and_a_gradient():
    pred = torch.tensor([[0.0, 0.0, 0.1, 0.1]], requires_grad=True)
    target = torch.tensor([[0.5, 0.5, 0.6, 0.6]])
    loss = giou_loss(pred, target)
    assert loss.item() > 1.0
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0


def test_inverted_prediction_is_reordered_and_still_has_gradient():
    # ymin > ymax, the same box as the target once re-ordered.
    pred = torch.tensor([[0.6, 0.7, 0.2, 0.3]], requires_grad=True)
    target = torch.tensor([[0.2, 0.3, 0.6, 0.7]])
    loss = giou_loss(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    shifted = torch.tensor([[0.6, 0.8, 0.2, 0.4]], requires_grad=True)
    giou_loss(shifted, target).backward()
    assert shifted.grad is not None and shifted.grad.abs().sum().item() > 0


def test_partial_overlap_is_between_zero_and_one():
    pred = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    target = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
    loss = giou_loss(pred, target).item()
    assert 0.0 < loss < 1.0


def test_config_rejects_unknown_options():
    with pytest.raises(ValueError):
        TrainConfig(train_record="a", val_record="b", box_loss="huber")
    with pytest.raises(ValueError):
        TrainConfig(train_record="a", val_record="b", select_by="accuracy")
