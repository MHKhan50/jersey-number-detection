"""Check for the same picture appearing in more than one split.

Roboflow exports often contain frames cut from the same match video, and augmented copies of
one source image. Two signals are reported:

* ``source_stem``: Roboflow renames files to ``<original>_jpg.rf.<hash>.jpg``; files that share
  the original name came from the same upload.
* ``near_duplicate``: a 64-bit difference hash (dHash) of the grey image; pairs within a small
  Hamming distance are visually near-identical.

The report is descriptive. It does not drop images, so the official splits stay comparable with
other work. Numbers go into the paper's limitations.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from jersey_detect.data import ImageRecord

_RF_SUFFIX = re.compile(r"_(jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-f]+$", re.IGNORECASE)


def source_stem(path: str) -> str:
    return _RF_SUFFIX.sub("", Path(path).stem)


def dhash(path: str, size: int = 8) -> int:
    with Image.open(path) as im:
        grey = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(grey, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def leakage_report(splits: dict[str, list[ImageRecord]], max_distance: int = 4) -> dict:
    hashes = {name: [dhash(r.path) for r in recs] for name, recs in splits.items()}
    stems = {name: {source_stem(r.path) for r in recs} for name, recs in splits.items()}
    report = {"max_hamming_distance": max_distance, "pairs": {}}
    for a, b in combinations(splits, 2):
        shared_stems = stems[a] & stems[b]
        ha = np.asarray(hashes[a], dtype=np.uint64)
        near = 0
        for h in hashes[b]:  # e.g. test images that look like a training image
            if len(ha) == 0:
                break
            bits = np.unpackbits((ha ^ np.uint64(h)).view(np.uint8)).reshape(len(ha), 64)
            if int(bits.sum(axis=1).min()) <= max_distance:
                near += 1
        report["pairs"][f"{a}-{b}"] = {
            "shared_source_names": len(shared_stems),
            f"{b}_images_with_near_duplicate_in_{a}": near,
            f"{b}_images": len(splits[b]),
        }
    return report
