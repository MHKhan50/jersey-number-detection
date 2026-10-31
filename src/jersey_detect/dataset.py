"""PyTorch datasets on top of :mod:`jersey_detect.data` and :mod:`jersey_detect.transforms`."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from jersey_detect.data import ImageRecord
from jersey_detect.detector import STRIDE
from jersey_detect.transforms import (
    centernet_targets,
    colour_jitter,
    keep_visible,
    letterbox,
    to_array,
)


class TrainDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        size: int,
        max_objects: int = 64,
        scale_range: tuple[float, float] = (0.6, 1.3),
        hflip: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.records = records
        self.size = size
        self.max_objects = max_objects
        self.scale_range = scale_range
        self.hflip = hflip
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        # One generator per (seed, epoch, item): augmentation is reproducible with any number of
        # data-loader workers.
        rng = np.random.default_rng([self.seed, self.epoch, i])
        rec = self.records[i]
        with Image.open(rec.path) as im:
            image = im.convert("RGB")
        scale = float(rng.uniform(*self.scale_range))
        canvas, lb = letterbox(image, self.size, scale, (float(rng.uniform()), float(rng.uniform())))
        canvas = colour_jitter(canvas, rng)
        boxes = keep_visible(lb.to_input(rec.boxes), self.size)
        if self.hflip and rng.uniform() < self.hflip:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = boxes.copy()
            boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        targets = centernet_targets(boxes, self.size, STRIDE, self.max_objects)
        item = {k: torch.from_numpy(v) for k, v in targets.items()}
        item["image"] = torch.from_numpy(to_array(canvas))
        return item


class EvalDataset(Dataset):
    """Centred letterbox, no augmentation. Returns the image and its index into ``records``."""

    def __init__(self, records: list[ImageRecord], size: int) -> None:
        self.records = records
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        with Image.open(self.records[i].path) as im:
            canvas, lb = letterbox(im, self.size)
        meta = torch.tensor([lb.scale, lb.pad_x, lb.pad_y, lb.orig_w, lb.orig_h], dtype=torch.float64)
        return torch.from_numpy(to_array(canvas)), meta, i
