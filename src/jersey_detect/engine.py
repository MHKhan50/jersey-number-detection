"""Training and inference for JerseyNet.

Long runs are expected to be interrupted (Kaggle sessions, preemption), so the loop:

* writes ``last.pt`` (model, optimiser, scheduler, scaler, history) atomically after every
  epoch and resumes from it when restarted with the same settings;
* keeps ``best.pt``, chosen on validation AP50:95 only (the test split is never seen here);
* stops cleanly before a wall-clock budget runs out, after a final validation pass;
* skips steps with a non-finite loss and gives up only if that keeps happening.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from jersey_detect.data import ImageRecord, load_dataset
from jersey_detect.dataset import EvalDataset, TrainDataset
from jersey_detect.detector import JerseyNet, decode, detection_loss
from jersey_detect.metrics import all_metrics
from jersey_detect.transforms import Letterbox

# Settings that change what is being trained. A checkpoint is only resumed if these match.
_RUN_IDENTITY = ("image_size", "epochs", "batch_size", "lr", "weight_decay", "seed", "hflip",
                 "scale_min", "scale_max", "warmup_epochs")


@dataclass
class DetectorConfig:
    data_root: str
    out_dir: str
    image_size: int = 256
    epochs: int = 140
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01
    grad_clip: float = 10.0
    seed: int = 0
    num_workers: int = 4
    eval_every: int = 5
    patience: int = 0  # validation passes without improvement before stopping; 0 turns it off
    amp: bool = True
    max_hours: float | None = None
    hflip: float = 0.0
    scale_min: float = 0.6
    scale_max: float = 1.3
    max_objects: int = 64
    top_k: int = 100
    resume: bool = True
    device: str = "auto"

    @classmethod
    def from_dict(cls, d: dict) -> DetectorConfig:
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown detector settings: {sorted(unknown)}")
        return cls(**d)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    """CUDA if it is present and actually runs kernels (old GPUs can be unsupported), else CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        try:
            torch.ones(8, device="cuda").mul_(2).sum().item()
            return torch.device("cuda")
        except RuntimeError as exc:
            print(f"[warn] CUDA is present but unusable ({exc}); falling back to CPU")
    return torch.device("cpu")


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# ------------------------------------------------------------------- inference


@torch.no_grad()
def predict_records(
    model: torch.nn.Module,
    records: list[ImageRecord],
    size: int,
    device: torch.device,
    batch_size: int = 32,
    top_k: int = 100,
    num_workers: int = 2,
    min_score: float = 0.001,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Boxes (normalised to each original image) and scores for every record, in order."""
    model.eval()
    loader = DataLoader(EvalDataset(records, size), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    dets: list[np.ndarray] = [np.zeros((0, 4), np.float32)] * len(records)
    scores: list[np.ndarray] = [np.zeros(0, np.float32)] * len(records)
    use_amp = device.type == "cuda"
    for images, metas, idx in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(images.to(device, non_blocking=True))
        boxes, sc = decode(out, top_k)
        boxes, sc = boxes.cpu().numpy(), sc.cpu().numpy()
        for b, s, m, i in zip(boxes, sc, metas.numpy(), idx.tolist(), strict=True):
            lb = Letterbox(float(m[0]), float(m[1]), float(m[2]), int(m[3]), int(m[4]))
            keep = s >= min_score
            nb = lb.to_normalised(b[keep])
            valid = (nb[:, 2] > nb[:, 0]) & (nb[:, 3] > nb[:, 1])
            dets[i], scores[i] = nb[valid], s[keep][valid].astype(np.float32)
    return dets, scores


def evaluate_records(model, records, size, device, **kw) -> dict:
    dets, scores = predict_records(model, records, size, device, **kw)
    return all_metrics([r.boxes for r in records], dets, scores)


def load_detector(checkpoint: str | os.PathLike, device: torch.device | None = None):
    device = device or pick_device()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = JerseyNet(**state.get("arch", {})).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


# ------------------------------------------------------------------- training


def _lr_lambda(total: int, warmup: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


def train_detector(cfg: DetectorConfig, log=print) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    started = time.time()
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    splits = load_dataset(cfg.data_root)
    train_set = TrainDataset(splits["train"], cfg.image_size, cfg.max_objects,
                             (cfg.scale_min, cfg.scale_max), cfg.hflip, cfg.seed)
    batch_size = min(cfg.batch_size, len(train_set))
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_set) > batch_size,
        generator=torch.Generator().manual_seed(cfg.seed),
        # Workers are rebuilt every epoch so they see the new epoch number for augmentation.
        persistent_workers=False,
    )
    arch: dict = {}
    model = JerseyNet(**arch).to(device)
    optimiser = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr)
    steps = len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, _lr_lambda(cfg.epochs * steps, cfg.warmup_epochs * steps, cfg.min_lr_ratio)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best, best_epoch, since_best, history = 0, -1.0, 0, 0, []
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if cfg.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        same = all(state["config"].get(k) == getattr(cfg, k) for k in _RUN_IDENTITY)
        if same:
            model.load_state_dict(state["model"])
            optimiser.load_state_dict(state["optimiser"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, best, best_epoch = state["epoch"], state["best"], state["best_epoch"]
            since_best, history = state["since_best"], state["history"]
            log(f"resumed from epoch {start_epoch} (best val AP50:95 {best:.4f})")
        else:
            log("last.pt was written with different settings; starting a new run")
            os.replace(last_path, out_dir / "last_previous_settings.pt")

    def validate() -> dict:
        return evaluate_records(model, splits["val"], cfg.image_size, device,
                                batch_size=batch_size, top_k=cfg.top_k,
                                num_workers=min(2, cfg.num_workers))

    budget = cfg.max_hours * 3600 if cfg.max_hours else None
    stopped = "completed"
    epoch_time = eval_time = 0.0
    bad_steps = 0
    epoch = start_epoch
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        t0 = time.time()
        train_set.set_epoch(epoch)
        model.train()
        sums = {"loss": 0.0, "heatmap": 0.0, "wh": 0.0, "offset": 0.0}
        n = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch["image"])
            losses = detection_loss(out, batch)
            loss = losses["loss"]
            optimiser.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                bad_steps += 1
                if bad_steps > 20:
                    raise RuntimeError("loss was not finite for more than 20 steps")
                scheduler.step()
                continue
            bad_steps = 0
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            for k in sums:
                sums[k] += float(losses[k])
            n += 1
        epoch_time = time.time() - t0
        record = {"epoch": epoch, "lr": optimiser.param_groups[0]["lr"], "seconds": round(epoch_time, 1)}
        record.update({f"train_{k}": v / max(n, 1) for k, v in sums.items()})

        elapsed = time.time() - started
        out_of_time = budget is not None and elapsed + 1.5 * epoch_time + eval_time + 60 > budget
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs or out_of_time:
            t1 = time.time()
            val = validate()
            eval_time = time.time() - t1
            record["val"] = val
            if val["ap50_95"] > best:
                best, best_epoch, since_best = val["ap50_95"], epoch, 0
                _save_atomic({"model": model.state_dict(), "arch": arch, "epoch": epoch,
                              "val": val, "config": asdict(cfg)}, best_path)
            else:
                since_best += 1
        history.append(record)
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        log(json.dumps({k: record[k] for k in ("epoch", "train_loss", "seconds")}
                       | ({"val_ap50_95": record["val"]["ap50_95"], "val_ap50": record["val"]["ap50"]}
                          if "val" in record else {})))
        _save_atomic({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "epoch": epoch, "best": best, "best_epoch": best_epoch,
                      "since_best": since_best, "history": history, "config": asdict(cfg)},
                     last_path)
        if out_of_time:
            stopped = "time_budget"
            break
        if cfg.patience and since_best >= cfg.patience:
            stopped = "early_stopping"
            break

    if not best_path.exists():  # e.g. resumed after the last epoch, or zero epochs requested
        val = validate()
        _save_atomic({"model": model.state_dict(), "arch": arch, "epoch": epoch, "val": val,
                      "config": asdict(cfg)}, best_path)
        best, best_epoch = val["ap50_95"], epoch
    summary = {"best_epoch": best_epoch, "best_val_ap50_95": best, "epochs_run": epoch,
               "stopped": stopped, "device": str(device), "amp": use_amp,
               "hours": round((time.time() - started) / 3600, 3)}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
