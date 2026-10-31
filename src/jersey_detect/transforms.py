"""Image geometry and CenterNet targets, in NumPy and Pillow only.

The detector sees a square letterboxed image (aspect ratio kept, grey padding). Training adds a
random scale and placement, and mild colour jitter. Horizontal flip is off by default: it mirrors
the digits, and the model should learn what real numbers look like.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance

PAD_VALUE = (114, 114, 114)
MEAN, STD = 0.5, 0.25  # simple fixed normalisation; the network is not ImageNet-pretrained


@dataclass(frozen=True)
class Letterbox:
    """Maps normalised original coordinates to input pixels: p = n * size_orig * scale + pad."""

    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int

    def to_input(self, boxes: np.ndarray) -> np.ndarray:
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * self.orig_w * self.scale + self.pad_x
        b[:, [1, 3]] = b[:, [1, 3]] * self.orig_h * self.scale + self.pad_y
        return b

    def to_normalised(self, boxes: np.ndarray) -> np.ndarray:
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / (self.orig_w * self.scale)
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / (self.orig_h * self.scale)
        return np.clip(b, 0.0, 1.0)


def letterbox(
    image: Image.Image,
    size: int,
    scale_jitter: float = 1.0,
    offset: tuple[float, float] | None = None,
) -> tuple[Image.Image, Letterbox]:
    """Resize to fit a ``size`` square (times ``scale_jitter``) and paste on a grey canvas.

    ``offset`` is the fractional placement in [0, 1] for x and y; ``None`` centres the image.
    """
    image = image.convert("RGB")
    w, h = image.size
    scale = min(size / w, size / h) * scale_jitter
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((nw, nh), Image.BILINEAR)
    fx, fy = (0.5, 0.5) if offset is None else offset
    pad_x = round((size - nw) * fx)
    pad_y = round((size - nh) * fy)
    canvas = Image.new("RGB", (size, size), PAD_VALUE)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, Letterbox(nw / w, pad_x, pad_y, w, h)


def colour_jitter(image: Image.Image, rng: np.random.Generator, strength: float = 0.3):
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        image = enhancer(image).enhance(float(rng.uniform(1 - strength, 1 + strength)))
    return image


def keep_visible(boxes: np.ndarray, size: int, min_px: float = 2.0, min_visible: float = 0.25):
    """Clip input-pixel boxes to the canvas and drop ones that are mostly outside it."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    clipped = np.clip(boxes, 0, size)
    full = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    cw, ch = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
    vis = cw * ch
    ok = (cw >= min_px) & (ch >= min_px) & (vis >= min_visible * np.maximum(full, 1e-9))
    return clipped[ok]


def to_array(image: Image.Image) -> np.ndarray:
    """HWC uint8 image to normalised CHW float32."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1).copy()


# ------------------------------------------------------------------- CenterNet targets


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """Radius from CenterNet (Zhou et al., 2019), itself from CornerNet."""
    a1, b1 = 1, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2**2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = diameter / 6
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    region = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    patch = g[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(region, patch, out=region)


def centernet_targets(boxes: np.ndarray, size: int, stride: int, max_objects: int) -> dict:
    """Heatmap, size, offset, index and mask arrays for input-pixel boxes [x1, y1, x2, y2]."""
    out = size // stride
    heatmap = np.zeros((1, out, out), dtype=np.float32)
    wh = np.zeros((max_objects, 2), dtype=np.float32)
    offset = np.zeros((max_objects, 2), dtype=np.float32)
    index = np.zeros(max_objects, dtype=np.int64)
    mask = np.zeros(max_objects, dtype=np.float32)
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=np.float32)[:max_objects]):
        bw, bh = (x2 - x1) / stride, (y2 - y1) / stride
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        ix, iy = min(int(cx), out - 1), min(int(cy), out - 1)
        radius = max(0, int(gaussian_radius(math.ceil(bh), math.ceil(bw))))
        draw_gaussian(heatmap[0], ix, iy, radius)
        wh[k] = (bw, bh)
        offset[k] = (cx - ix, cy - iy)
        index[k] = iy * out + ix
        mask[k] = 1.0
    return {"heatmap": heatmap, "wh": wh, "offset": offset, "index": index, "mask": mask}
