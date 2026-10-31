"""Reference models for the comparison table.

* ``mean_box``: predicts the average training box on every image. It has no image input and
  marks the floor any detector must beat.
* ``LegacyRegressor``: the original coursework network (three convolutions, global average
  pooling, one box). It is trained exactly as in v1 (128 x 128 resize, Adam 1e-3, batch 8,
  MSE on the first box, 15 epochs) so the v1 numbers can be reproduced with the new evaluator.
  One difference: the weights after the last epoch are used, where v1 kept the epoch with the
  lowest validation loss. This keeps the validation split out of the baseline entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jersey_detect.data import ImageRecord, load_dataset

LEGACY_SIZE = 128


def mean_box(train: list[ImageRecord]) -> np.ndarray:
    first = [r.boxes[0] for r in train if r.labelled]
    if not first:
        raise ValueError("No labelled training images")
    return np.mean(np.stack(first), axis=0).astype(np.float32)


def constant_predictions(box: np.ndarray, records: list[ImageRecord]):
    return [box[None].copy() for _ in records], [np.ones(1, np.float32) for _ in records]


class LegacyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 128), nn.ReLU(inplace=True),
        )
        self.box = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.box(self.features(x)))  # [x1, y1, x2, y2] in 0-1


def _load_small(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB").resize((LEGACY_SIZE, LEGACY_SIZE), Image.BILINEAR),
                         dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class _Small(Dataset):
    def __init__(self, records, with_box: bool):
        self.records = records
        self.with_box = with_box

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        x = _load_small(self.records[i].path)
        if self.with_box:
            return x, torch.from_numpy(self.records[i].boxes[0].astype(np.float32))
        return x, i


def _order(b: np.ndarray) -> np.ndarray:
    return np.stack([np.minimum(b[:, 0], b[:, 2]), np.minimum(b[:, 1], b[:, 3]),
                     np.maximum(b[:, 0], b[:, 2]), np.maximum(b[:, 1], b[:, 3])], axis=1)


@torch.no_grad()
def legacy_predict(model: LegacyRegressor, records, device, batch_size: int = 64):
    model.eval()
    dets = [None] * len(records)
    for x, idx in DataLoader(_Small(records, False), batch_size=batch_size, num_workers=2):
        boxes = _order(model(x.to(device)).cpu().numpy())
        for b, i in zip(boxes, idx.tolist(), strict=True):
            dets[i] = b[None]
    return dets, [np.ones(1, np.float32) for _ in records]


def train_legacy(data_root: str, out_dir: str, device: torch.device, epochs: int = 15,
                 seed: int = 0, log=print) -> LegacyRegressor:
    torch.manual_seed(seed)
    splits = load_dataset(data_root)
    labelled = [r for r in splits["train"] if r.labelled]
    loader = DataLoader(_Small(labelled, True), batch_size=8, shuffle=True, num_workers=2,
                        generator=torch.Generator().manual_seed(seed))
    model = LegacyRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + float(loss), n + 1
        log(json.dumps({"legacy_epoch": epoch, "train_mse": total / max(n, 1)}))
    torch.save({"model": model.state_dict(), "epochs": epochs, "seed": seed}, out / "legacy.pt")
    return model
