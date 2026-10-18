"""Run the trained detector on one image and draw the result."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from jersey_detect.data import IMAGE_SIZE
from jersey_detect.model import JerseyDetector, order_box


def load_model(checkpoint: str, device: torch.device | None = None) -> JerseyDetector:
    device = device or torch.device("cpu")
    model = JerseyDetector().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: JerseyDetector, image: Image.Image) -> dict:
    """Return the presence probability and box in pixel coordinates of the original image."""
    width, height = image.size
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    box_logits, cls_logits = model(tensor)
    ymin, xmin, ymax, xmax = order_box(torch.sigmoid(box_logits))[0].tolist()
    prob = torch.sigmoid(cls_logits)[0].item()
    return {
        "probability": prob,
        "present": prob >= 0.5,
        "box_pixels": [ymin * height, xmin * width, ymax * height, xmax * width],
    }


def annotate(image: Image.Image, result: dict) -> Image.Image:
    out = image.convert("RGB").copy()
    if result["present"]:
        ymin, xmin, ymax, xmax = result["box_pixels"]
        draw = ImageDraw.Draw(out)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=3)
        draw.text((xmin, max(0, ymin - 12)), f"0 {result['probability']:.2f}", fill=(0, 255, 0))
    return out


def predict_file(checkpoint: str, image_path: str, out_path: str) -> dict:
    model = load_model(checkpoint)
    image = Image.open(Path(image_path))
    result = predict_image(model, image)
    annotate(image, result).save(out_path)
    return result
