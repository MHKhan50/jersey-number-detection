import pytest

torch = pytest.importorskip("torch")

from jersey_detect.model import JerseyDetector  # noqa: E402


def test_output_shapes():
    model = JerseyDetector().eval()
    box, cls = model(torch.zeros(2, 3, 128, 128))
    assert box.shape == (2, 4)
    assert cls.shape == (2,)


def test_parameter_count_matches_original_architecture():
    # Conv 3->16, 16->32, 32->64 (3x3 + bias), Dense 64->128, heads 128->4 and 128->1
    expected = (
        (3 * 9 * 16 + 16) + (16 * 9 * 32 + 32) + (32 * 9 * 64 + 64)
        + (64 * 128 + 128) + (128 * 4 + 4) + (128 * 1 + 1)
    )
    assert sum(p.numel() for p in JerseyDetector().parameters()) == expected
