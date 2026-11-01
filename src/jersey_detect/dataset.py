"""PyTorch datasets on top of :mod:`jersey_detect.data` and :mod:`jersey_detect.transforms`."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from jersey_detect.data import ImageRecord
from jersey_detect.detector import STRIDE
from jersey_detect.transforms import (
    PAD_VALUE,
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
        mosaic: float = 0.0,
    ) -> None:
        self.records = records
        self.size = size
        self.max_objects = max_objects
        self.scale_range = scale_range
        self.hflip = hflip
        self.seed = seed
        self.mosaic = mosaic
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        # One generator per (seed, epoch, item): augmentation is reproducible with any number of
        # data-loader workers.
        rng = np.random.default_rng([self.seed, self.epoch, i])
        if self.mosaic and rng.uniform() < self.mosaic:
            canvas, boxes = self._mosaic(i, rng)
        else:
            canvas, boxes = self._single(self.records[i], rng)
        canvas = colour_jitter(canvas, rng)
        if self.hflip and rng.uniform() < self.hflip:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = boxes.copy()
            boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        targets = centernet_targets(boxes, self.size, STRIDE, self.max_objects)
        item = {k: torch.from_numpy(v) for k, v in targets.items()}
        item["image"] = torch.from_numpy(to_array(canvas))
        return item


    def _single(self, rec: ImageRecord, rng: np.random.Generator):
        with Image.open(rec.path) as im:
            image = im.convert("RGB")
        scale = float(rng.uniform(*self.scale_range))
        canvas, lb = letterbox(image, self.size, scale, (float(rng.uniform()), float(rng.uniform())))
        return canvas, keep_visible(lb.to_input(rec.boxes), self.size)

    def _mosaic(self, i: int, rng: np.random.Generator):
        """YOLO-style mosaic: four images on a 2x2 canvas, then a random crop of one image size.

        Each tile keeps the normal letterbox scale, so objects are not shrunk; the crop shows
        parts of up to four scenes and puts objects near image borders more often.
        """
        size = self.size
        big = Image.new("RGB", (2 * size, 2 * size), PAD_VALUE)
        boxes = []
        others = rng.integers(0, len(self.records), size=3)
        for k, j in enumerate([i, *(int(x) for x in others)]):
            tile, tile_boxes = self._single(self.records[j], rng)
            ox, oy = (k % 2) * size, (k // 2) * size
            big.paste(tile, (ox, oy))
            if len(tile_boxes):
                boxes.append(tile_boxes + np.array([ox, oy, ox, oy], dtype=np.float32))
        x0, y0 = (int(v) for v in rng.integers(0, size + 1, size=2))
        canvas = big.crop((x0, y0, x0 + size, y0 + size))
        all_boxes = np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32)
        shifted = all_boxes - np.array([x0, y0, x0, y0], dtype=np.float32)
        return canvas, keep_visible(shifted, size)


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
