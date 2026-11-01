"""YOLOv8n baselines, scored with exactly the same evaluator as JerseyNet.

Three settings separate the two things a pretrained detector brings:

* COCO-pretrained at 640 px: the earlier comparison (50 epochs);
* COCO-pretrained at 256 px: same input size as JerseyNet, so only pretraining differs;
* from scratch at 256 px: no pretraining either, trained as long as JerseyNet (140 epochs).

Ultralytics selects ``best.pt`` on the validation split. Predictions on validation and test are
then made with a low confidence threshold (0.001) and at most 100 boxes per image, the COCO
convention, and scored by :mod:`jersey_detect.metrics`. The test split is used only here.

Ultralytics is AGPL-3.0 and is installed separately (``pip install ultralytics``); it is not
part of this package.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jersey_detect.data import find_split_dirs, load_dataset
from jersey_detect.metrics import all_metrics, bootstrap_ci


@dataclass(frozen=True)
class YoloSetting:
    name: str
    weights: str  # "yolov8n.pt" (COCO-pretrained) or "yolov8n.yaml" (random initialisation)
    imgsz: int
    epochs: int

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")


SETTINGS = (
    YoloSetting("YOLOv8n, COCO-pretrained, 640 px", "yolov8n.pt", 640, 50),
    YoloSetting("YOLOv8n, COCO-pretrained, 256 px", "yolov8n.pt", 256, 50),
    YoloSetting("YOLOv8n, from scratch, 256 px", "yolov8n.yaml", 256, 140),
)
SMOKE_SETTINGS = (YoloSetting("YOLOv8n, from scratch, 64 px", "yolov8n.yaml", 64, 1),)


def write_data_yaml(data_root: str, path: Path) -> Path:
    """Ultralytics dataset file with absolute split paths (the data folder may be read-only)."""
    splits = find_split_dirs(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            f"train: {splits['train'] / 'images'}",
            f"val: {splits['val'] / 'images'}",
            f"test: {splits['test'] / 'images'}",
            "nc: 1",
            "names: ['number']",
            "",
        ])
    )
    return path


def predict(model, records, imgsz: int, batch: int = 32):
    """Boxes (normalised xyxy) and scores for each record, in the order of ``records``."""
    index = {str(Path(r.path).resolve()): i for i, r in enumerate(records)}
    dets = [np.zeros((0, 4), np.float32) for _ in records]
    scores = [np.zeros(0, np.float32) for _ in records]
    paths = [r.path for r in records]
    for start in range(0, len(paths), batch):
        for res in model.predict(paths[start:start + batch], imgsz=imgsz, conf=0.001, iou=0.7,
                                 max_det=100, verbose=False):
            i = index[str(Path(res.path).resolve())]
            dets[i] = res.boxes.xyxyn.cpu().numpy().astype(np.float32)
            scores[i] = res.boxes.conf.cpu().numpy().astype(np.float32)
    return dets, scores


def _score(records, dets, scores, n_boot: int) -> dict:
    gts = [r.boxes for r in records]
    m = all_metrics(gts, dets, scores)
    if n_boot:
        m["ci95"] = bootstrap_ci(gts, dets, scores, n_boot=n_boot)
    return m


def _table(summary: dict) -> str:
    rows = ["# YOLOv8n baselines", "",
            "Same evaluator as JerseyNet. Test intervals are 95% bootstrap intervals over images.", "",
            "| Model | Split | AP50:95 | AP50 | AP75 | AR@100 | First-box mean IoU | First-box IoU ≥ 0.5 |",
            "|---|---|---|---|---|---|---|---|"]
    for name, res in summary["results"].items():
        for split in ("val", "test"):
            m = res[split]
            ci = m.get("ci95", {})

            def f(k, m=m, ci=ci):
                s = f"{m[k]:.3f}"
                return s + (f" [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else "")

            rows.append(f"| {name} | {split} | {f('ap50_95')} | {f('ap50')} | {m['ap75']:.3f} | "
                        f"{m['ar']:.3f} | {f('first_box_mean_iou')} | "
                        f"{m['first_box_iou_ge_0_5'] * 100:.1f}% |")
    rows += ["", "## Over seeds (test)", "", "| Setting | Seeds | AP50:95 | AP50 | First-box mean IoU |",
             "|---|---|---|---|---|"]
    for name, agg in summary.get("over_seeds", {}).items():
        cells = " | ".join(f"{agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}"
                           for k in ("ap50_95", "ap50", "first_box_mean_iou"))
        rows.append(f"| {name} | {agg['seeds']} | {cells} |")
    if summary["errors"]:
        rows += ["", "## Runs that failed", ""] + [f"- {e['run']}: {e['error']}" for e in summary["errors"]]
    return "\n".join(rows) + "\n"


def run_yolo(data_root: str, out_dir: str, seeds=(0, 1, 2), time_budget_hours: float = 9.5,
             settings=SETTINGS, batch: int = 16, workers: int = 4, n_boot: int = 1000,
             device=None, log=print) -> dict:
    from ultralytics import YOLO  # imported here: optional dependency

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    splits = load_dataset(data_root)
    data_yaml = write_data_yaml(data_root, out / "data.yaml")
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.setdefault("results", {})
    summary["errors"] = []
    summary["skipped"] = []

    def save():
        by_setting: dict = {}
        for s in settings:
            per_seed = {seed: summary["results"].get(f"{s.name} (seed {seed})") for seed in seeds}
            done = {k: v["test"] for k, v in per_seed.items() if v}
            if done:
                agg = {"seeds": sorted(done)}
                for k in ("ap50_95", "ap50", "first_box_mean_iou"):
                    v = np.array([m[k] for m in done.values()])
                    agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
                by_setting[s.name] = agg
        summary["over_seeds"] = by_setting
        summary["elapsed_hours"] = round((time.time() - started) / 3600, 3)
        summary_path.write_text(json.dumps(summary, indent=2))
        (out / "results.md").write_text(_table(summary))

    budget = time_budget_hours * 3600
    for seed in seeds:  # every setting once before any second seed
        for s in settings:
            key = f"{s.name} (seed {seed})"
            run_dir = out / "runs" / f"{s.slug}_seed{seed}"
            if key in summary["results"] and (run_dir / "metrics.json").exists():
                log(f"already done: {key}")
                continue
            if time.time() - started > budget:
                log(f"skipping {key}: time budget used")
                summary["skipped"].append(key)
                save()
                continue
            log(f"=== {key}")
            t0 = time.time()
            try:
                if run_dir.exists():
                    shutil.rmtree(run_dir)  # an unfinished earlier attempt
                model = YOLO(s.weights)
                kwargs = dict(data=str(data_yaml), epochs=s.epochs, imgsz=s.imgsz, batch=batch,
                              seed=seed, deterministic=True, workers=workers, project=str(out / "runs"),
                              name=run_dir.name, exist_ok=True, plots=False, verbose=False)
                if device is not None:
                    kwargs["device"] = device
                model.train(**kwargs)
                best = run_dir / "weights" / "best.pt"
                trained = YOLO(str(best))
                res = {"train": {"weights": s.weights, "imgsz": s.imgsz, "epochs": s.epochs,
                                 "batch": batch, "minutes": round((time.time() - t0) / 60, 1)}}
                for split in ("val", "test"):
                    d, sc = predict(trained, splits[split], s.imgsz)
                    res[split] = _score(splits[split], d, sc, n_boot if split == "test" else 0)
                (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
                summary["results"][key] = res
                log(json.dumps({"run": key, "test_ap50_95": res["test"]["ap50_95"],
                                "test_ap50": res["test"]["ap50"], "minutes": res["train"]["minutes"]}))
            except Exception as exc:  # noqa: BLE001 - one failed run must not stop the others
                log(traceback.format_exc())
                summary["errors"].append({"run": key, "error": f"{type(exc).__name__}: {exc}"})
            save()
    save()
    return summary
