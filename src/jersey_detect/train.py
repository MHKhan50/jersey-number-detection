"""Training loop: box loss (MSE, L1 or GIoU), binary cross-entropy for presence, Adam optimiser.

The default box loss is MSE, which matches the first results. ``box_loss="giou"`` optimises the
overlap directly, and ``select_by="iou"`` keeps the checkpoint with the best validation IoU.
"""

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
from jersey_detect.model import JerseyDetector, order_box

BOX_LOSSES = ("mse", "l1", "giou")
SELECT_BY = ("loss", "iou")


@dataclass
class TrainConfig:
    train_record: str
    val_record: str
    out_dir: str = "checkpoints"
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 0
    box_loss: str = "mse"
    select_by: str = "loss"

    def __post_init__(self) -> None:
        if self.box_loss not in BOX_LOSSES:
            raise ValueError(f"box_loss must be one of {BOX_LOSSES}, got {self.box_loss!r}")
        if self.select_by not in SELECT_BY:
            raise ValueError(f"select_by must be one of {SELECT_BY}, got {self.select_by!r}")


class SampleDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = torch.from_numpy(s.image).permute(2, 0, 1)  # HWC -> CHW
        return image, torch.from_numpy(s.box), torch.tensor(s.label)


def giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Mean of 1 - GIoU for boxes shaped (N, 4) as [ymin, xmin, ymax, xmax].

    GIoU equals IoU minus the share of the smallest enclosing box that is not covered by the
    two boxes. Unlike IoU it still gives a gradient when the boxes do not overlap, and the loss
    is in [0, 2].
    """
    py1, px1, py2, px2 = pred.unbind(-1)
    # A sigmoid output can put min above max, giving zero area and no gradient through the
    # clamps below. Re-ordering keeps the gradient flowing.
    py1, py2 = torch.minimum(py1, py2), torch.maximum(py1, py2)
    px1, px2 = torch.minimum(px1, px2), torch.maximum(px1, px2)
    ty1, tx1, ty2, tx2 = target.unbind(-1)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter = inter_h * inter_w
    area_p = (py2 - py1).clamp(min=0) * (px2 - px1).clamp(min=0)
    area_t = (ty2 - ty1).clamp(min=0) * (tx2 - tx1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union
    enclose = (torch.max(py2, ty2) - torch.min(py1, ty1)) * (
        torch.max(px2, tx2) - torch.min(px1, tx1)
    )
    enclose = enclose + eps
    giou = iou - (enclose - union) / enclose
    return (1 - giou).mean()


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
        pred_box = order_box(torch.sigmoid(box_logits))
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


def _box_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return nn.functional.mse_loss(pred, target)
    if kind == "l1":
        return nn.functional.l1_loss(pred, target)
    return giou_loss(pred, target)


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
    bce_fn = nn.BCEWithLogitsLoss()

    best_score = float("inf")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for images, boxes, labs in train_loader:
            images, boxes, labs = images.to(device), boxes.to(device), labs.to(device)
            box_logits, cls_logits = model(images)
            box_term = _box_loss(torch.sigmoid(box_logits), boxes, cfg.box_loss)
            loss = box_term + bce_fn(cls_logits, labs)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics))
        if cfg.select_by == "iou":
            score = -metrics["mean_iou_positive"]
        else:
            score = metrics["box_mse"] + metrics["class_bce"]
        if score < best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return history[-1]
