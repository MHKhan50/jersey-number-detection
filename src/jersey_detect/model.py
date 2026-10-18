"""The custom CNN, ported from the original Keras model with the same layer sizes.

Conv(16) -> Pool -> Conv(32) -> Pool -> Conv(64) -> GlobalAvgPool -> Dense(128)
then two heads: a 4-value box (sigmoid) and one presence logit.
"""

from __future__ import annotations

import torch
from torch import nn


class JerseyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        self.box_head = nn.Linear(128, 4)
        self.class_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (box_logits, class_logits). Apply sigmoid to get values in [0, 1]."""
        h = self.features(x)
        return self.box_head(h), self.class_head(h).squeeze(-1)


def order_box(box: torch.Tensor) -> torch.Tensor:
    """Return boxes shaped (..., 4) as [ymin, xmin, ymax, xmax] with min <= max on each axis.

    The sigmoid head can output min above max. Such a box has zero area, so its IoU is 0 even
    when the box sits on the right object. GIoU training is invariant to this swap, so the
    model is not pushed away from it. Ordering the output before scoring or drawing fixes that.
    """
    y1, x1, y2, x2 = box.unbind(-1)
    return torch.stack(
        [torch.minimum(y1, y2), torch.minimum(x1, x2), torch.maximum(y1, y2), torch.maximum(x1, x2)],
        dim=-1,
    )
