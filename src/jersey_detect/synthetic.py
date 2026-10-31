"""A small synthetic dataset in the same YOLO layout, for tests and smoke runs.

Each image is a noisy background with one to three "shirts", each carrying a printed number.
The labelled box is the number, as in the real export. Some images have no number, some have
no label file at all (kind 0: empty label file, kind 5: no file), so every code path of the loader and the evaluator is exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def make_dataset(root: str | Path, counts=(48, 12, 12), image_size=(160, 120), seed: int = 0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    font = _font(28)
    for split, n in zip(("train", "valid", "test"), counts, strict=True):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            w, h = image_size
            noise = rng.integers(40, 120, size=(h, w, 3), dtype=np.uint8)
            img = Image.fromarray(noise)
            draw = ImageDraw.Draw(img)
            lines = []
            kind = i % 6
            n_obj = 0 if kind in (0, 5) else 1 + (i % 3)
            for _ in range(n_obj):
                sx, sy = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 50))
                colour = tuple(int(c) for c in rng.integers(150, 255, size=3))
                draw.rectangle([sx, sy, sx + 55, sy + 48], fill=colour)
                text = str(int(rng.integers(1, 99)))
                tx, ty = sx + 8, sy + 6
                x1, y1, x2, y2 = draw.textbbox((tx, ty), text, font=font)
                draw.text((tx, ty), text, fill=(10, 10, 10), font=font)
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            name = f"{split}_{i:04d}_jpg.rf.{i:08x}"
            img.save(root / split / "images" / f"{name}.jpg", quality=92)
            if kind == 5:
                continue  # background image with no label file at all
            (root / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return root
