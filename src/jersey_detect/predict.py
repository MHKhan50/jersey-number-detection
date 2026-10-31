"""Run a trained JerseyNet checkpoint on single images and draw the boxes."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from jersey_detect.data import ImageRecord
from jersey_detect.engine import load_detector, pick_device, predict_records


def predict_images(checkpoint: str, images: list[str], out_dir: str, threshold: float = 0.3) -> list[dict]:
    device = pick_device()
    model, state = load_detector(checkpoint, device)
    size = state["config"]["image_size"]
    records = [ImageRecord(p) for p in images]
    dets, scores = predict_records(model, records, size, device, batch_size=8, num_workers=0)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for path, boxes, sc in zip(images, dets, scores, strict=True):
        keep = sc >= threshold
        with Image.open(path) as im:
            img = im.convert("RGB")
        w, h = img.size
        px = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes[keep]]
        draw = ImageDraw.Draw(img)
        for (x1, y1, x2, y2), s in zip(px, sc[keep], strict=True):
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            draw.text((x1, max(0, y1 - 11)), f"{s:.2f}", fill=(0, 255, 0))
        img.save(out / Path(path).name)
        results.append({"image": path, "boxes_px": [[round(v, 1) for v in b] for b in px],
                        "scores": [round(float(s), 4) for s in sc[keep]]})
    return results
