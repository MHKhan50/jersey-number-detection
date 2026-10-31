import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jersey_detect.detector import STRIDE, JerseyNet, decode, detection_loss  # noqa: E402
from jersey_detect.transforms import centernet_targets  # noqa: E402


def test_output_shapes_at_stride_4():
    model = JerseyNet().eval()
    out = model(torch.zeros(2, 3, 128, 128))
    assert out["heatmap"].shape == (2, 1, 32, 32)
    assert out["wh"].shape == (2, 2, 32, 32)
    assert out["offset"].shape == (2, 2, 32, 32)


def test_model_is_small_and_randomly_initialised():
    n = sum(p.numel() for p in JerseyNet().parameters())
    assert 1_000_000 < n < 10_000_000


def _targets_as_batch(boxes, size=128):
    t = centernet_targets(np.asarray(boxes, np.float32), size, STRIDE, 16)
    return {k: torch.from_numpy(v)[None] for k, v in t.items()}


def test_decode_recovers_boxes_from_ideal_outputs():
    boxes = [[20.0, 30.0, 44.0, 62.0], [80.0, 12.0, 100.0, 40.0]]
    t = _targets_as_batch(boxes)
    hw = 128 // STRIDE
    heat = t["heatmap"].clamp(1e-6, 1 - 1e-6)
    wh = torch.zeros(1, 2, hw * hw)
    off = torch.zeros(1, 2, hw * hw)
    for k in range(2):
        wh[0, :, t["index"][0, k]] = t["wh"][0, k]
        off[0, :, t["index"][0, k]] = t["offset"][0, k]
    out = {"heatmap": torch.logit(heat), "wh": wh.view(1, 2, hw, hw), "offset": off.view(1, 2, hw, hw)}
    pred, scores = decode(out, top_k=2)
    got = sorted(pred[0].tolist())
    for g, b in zip(got, sorted(boxes), strict=True):
        np.testing.assert_allclose(g, b, atol=1e-3)
    assert torch.all(scores > 0.99)


def test_loss_is_finite_and_backpropagates():
    model = JerseyNet()
    batch = _targets_as_batch([[20.0, 30.0, 44.0, 62.0]])
    batch = {k: torch.cat([v, v]) for k, v in batch.items()}
    out = model(torch.randn(2, 3, 128, 128))
    losses = detection_loss(out, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.heatmap[-1].weight.grad is not None


def test_loss_with_no_objects():
    model = JerseyNet()
    batch = _targets_as_batch(np.zeros((0, 4)))
    losses = detection_loss(model(torch.randn(1, 3, 128, 128)), batch)
    assert torch.isfinite(losses["loss"])
