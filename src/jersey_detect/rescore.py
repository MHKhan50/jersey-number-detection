"""Re-score saved checkpoints on the test split, with AP by number size.

The main runs were scored before size-stratified AP was added. Their outputs (Kaggle notebook
versions attached as inputs) still hold every selected checkpoint, so nothing has to be
retrained:

* JerseyNet: ``best.pt`` written by :func:`jersey_detect.engine.train_detector`, next to a
  ``config.json`` that gives the input size and recipe;
* YOLOv8n: ``runs/<setting>_seed<k>/weights/best.pt`` written by Ultralytics, with the input
  size in ``args.yaml``.

Each checkpoint is predicted on the test split and scored with the same evaluator. The overall
AP50:95 is reported next to the size bins so that it can be checked against the original
results before the size numbers are used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from jersey_detect.data import load_dataset
from jersey_detect.metrics import all_metrics, size_metrics


def _jerseynet_name(cfg: dict) -> str:
    parts = [f"{cfg.get('image_size', 256)} px"]
    if cfg.get("mosaic"):
        parts.append("mosaic")
    if cfg.get("giou_weight"):
        parts.append("GIoU")
    if cfg.get("train_fraction", 1.0) < 1:
        parts.append(f"fraction {cfg['train_fraction']:g}")
    return "JerseyNet, " + ", ".join(parts)


def find_checkpoints(roots) -> list[dict]:
    """Every JerseyNet and YOLOv8n checkpoint below ``roots``, with name, seed and input size."""
    found, seen = [], set()
    for root in roots:
        for best in sorted(Path(root).rglob("best.pt")):
            real = best.resolve()
            if real in seen:
                continue
            seen.add(real)
            if best.parent.name == "weights" and (best.parent.parent / "args.yaml").exists():
                run = best.parent.parent
                args = (run / "args.yaml").read_text()
                imgsz = int(re.search(r"^imgsz:\s*(\d+)", args, re.M).group(1))
                m = re.match(r"(.+)_seed(\d+)$", run.name)
                if not m:
                    continue
                found.append({"kind": "yolo", "path": str(best), "imgsz": imgsz,
                              "name": m.group(1), "seed": int(m.group(2)), "run": str(run)})
            elif (best.parent / "config.json").exists():
                cfg = json.loads((best.parent / "config.json").read_text())
                name = _jerseynet_name(cfg)
                if not best.parent.name.startswith("jersey_net_seed"):
                    name += " (ablation run)"  # keeps the seed-0 ablation baseline apart from the main runs
                found.append({"kind": "jerseynet", "path": str(best), "imgsz": cfg["image_size"],
                              "name": name, "seed": cfg.get("seed", 0),
                              "run": str(best.parent)})
    return found


def rescore(data_root: str, roots, out_dir: str, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_dataset(data_root)["test"]
    gts = [r.boxes for r in records]
    results: dict = {}
    for ck in find_checkpoints(roots):
        key = f"{ck['name']} (seed {ck['seed']})"
        if key in results:  # the same run attached twice under different inputs
            continue
        log(f"=== {key}: {ck['path']}")
        try:
            if ck["kind"] == "yolo":
                from ultralytics import YOLO

                from jersey_detect.yolo_baseline import predict

                dets, scores = predict(YOLO(ck["path"]), records, ck["imgsz"])
            else:
                from jersey_detect.engine import load_detector, pick_device, predict_records

                device = pick_device()
                model, _ = load_detector(ck["path"], device)
                dets, scores = predict_records(model, records, ck["imgsz"], device)
            m = all_metrics(gts, dets, scores)
            m["by_size"] = size_metrics(gts, dets, scores)
            results[key] = {**ck, "test": m}
            log(json.dumps({"run": key, "ap50_95": round(m["ap50_95"], 4),
                            **{b: round(v["ap50_95"], 4) for b, v in m["by_size"].items()}}))
        except Exception as exc:  # noqa: BLE001 - one bad checkpoint must not stop the rest
            log(f"failed: {key}: {type(exc).__name__}: {exc}")
    summary = {"results": results, "aggregate": _aggregate(results)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "results.md").write_text(_table(summary))
    return summary


def _aggregate(results: dict) -> dict:
    groups: dict = {}
    for r in results.values():
        groups.setdefault(r["name"], []).append(r)
    agg = {}
    for name, runs in sorted(groups.items()):
        row = {"seeds": sorted(r["seed"] for r in runs)}
        cols = {"ap50_95": [r["test"]["ap50_95"] for r in runs]}
        for b in ("small", "medium", "large"):
            cols[f"ap_{b}"] = [r["test"]["by_size"][b]["ap50_95"] for r in runs]
            row[f"num_gt_{b}"] = runs[0]["test"]["by_size"][b]["num_gt"]
        for k, v in cols.items():
            v = np.asarray(v)
            row[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        agg[name] = row
    return agg


def _table(summary: dict) -> str:
    agg = summary["aggregate"]
    first = next(iter(agg.values()), None)
    counts = (f" Test boxes: small {first['num_gt_small']}, medium {first['num_gt_medium']}, "
              f"large {first['num_gt_large']}.") if first else ""
    rows = ["# Test AP by number size", "",
            "Number side in pixels of the 128 px images: small < 8, medium 8-16, large > 16." + counts,
            "Mean ± standard deviation over seeds.", "",
            "| Model | Seeds | AP50:95 | AP small | AP medium | AP large |", "|---|---|---|---|---|---|"]
    for name, r in agg.items():
        cells = " | ".join(f"{r[k]['mean']:.3f} ± {r[k]['std']:.3f}"
                           for k in ("ap50_95", "ap_small", "ap_medium", "ap_large"))
        rows.append(f"| {name} | {r['seeds']} | {cells} |")
    return "\n".join(rows) + "\n"
