"""Evaluate a saved checkpoint on any split (TFRecord file or YOLO split directory).

The metrics match the ones logged during training, so validation and test numbers are
directly comparable with the training history.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from jersey_detect.data import load_samples
from jersey_detect.model import JerseyDetector
from jersey_detect.train import SampleDataset, evaluate


def evaluate_checkpoint(checkpoint: str, data_path: str, batch_size: int = 8) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device)
    model = JerseyDetector().to(device)
    model.load_state_dict(state["state_dict"])
    loader = DataLoader(
        SampleDataset(load_samples(data_path)), batch_size=batch_size, shuffle=False
    )
    metrics = evaluate(model, loader, device)
    metrics["samples"] = len(loader.dataset)
    return metrics


def main_evaluate(checkpoint: str, data_path: str) -> str:
    return json.dumps(evaluate_checkpoint(checkpoint, data_path))
