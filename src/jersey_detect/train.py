"""Training loop: MSE for the box, binary cross-entropy for presence, Adam optimiser."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jersey_detect.data import Sample, load_samples
from jersey_detect.metrics import binary_accuracy, mean_iou
from jersey_detect.model import JerseyDetector


@dataclass
class TrainConfig:
    train_record: str
    val_record: str
    out_dir: str = "checkpoints"
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 0


class SampleDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = torch.from_numpy(s.image).permute(2, 0, 1)  # HWC -> CHW
        return image, torch.from_numpy(s.box), torch.tensor(s.label)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mse_fn = nn.MSELoss(reduction="sum")
    bce_fn = nn.BCEWithLogitsLoss(reduction="sum")
    box_sse, cls_sum, n = 0.0, 0.0, 0
    probs, labels, pred_boxes, true_boxes = [], [], [], []
    for images, boxes, labs in loader:
        images, boxes, labs = images.to(device), boxes.to(device), labs.to(device)
        box_logits, cls_logits = model(images)
        pred_box = torch.sigmoid(box_logits)
        box_sse += mse_fn(pred_box, boxes).item()
        cls_sum += bce_fn(cls_logits, labs).item()
        n += images.shape[0]
        p = torch.sigmoid(cls_logits).cpu().tolist()
        probs += p
        labels += labs.cpu().tolist()
        pred_boxes += pred_box.cpu().tolist()
        true_boxes += boxes.cpu().tolist()
    positives = [i for i, y in enumerate(labels) if y >= 0.5]
    return {
        "box_mse": box_sse / (4 * n),
        "class_bce": cls_sum / n,
        "class_accuracy": binary_accuracy(probs, labels),
        "mean_iou_positive": mean_iou(
            [pred_boxes[i] for i in positives], [true_boxes[i] for i in positives]
        ),
    }


def train(cfg: TrainConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        SampleDataset(load_samples(cfg.train_record)),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        SampleDataset(load_samples(cfg.val_record)), batch_size=cfg.batch_size, shuffle=False
    )

    model = JerseyDetector().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for images, boxes, labs in train_loader:
            images, boxes, labs = images.to(device), boxes.to(device), labs.to(device)
            box_logits, cls_logits = model(images)
            loss = mse_fn(torch.sigmoid(box_logits), boxes) + bce_fn(cls_logits, labs)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics))
        val_loss = metrics["box_mse"] + metrics["class_bce"]
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return history[-1]
