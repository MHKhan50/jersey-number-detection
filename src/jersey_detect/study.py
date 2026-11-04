"""Follow-up study: data efficiency and an augmentation-matched YOLOv8n.

Two questions from the main results are tested here.

1. **Does pretraining matter more with less data?** JerseyNet (baseline recipe, 256 px),
   YOLOv8n from scratch and YOLOv8n from COCO weights (both 256 px) are trained on 10% and
   25% of the training images. The subsets are nested and identical for every model
   (:func:`jersey_detect.data.train_subset`). Each run takes the same number of optimisation
   steps as the full-data run, so a smaller subset is not also a shorter schedule. The
   full-data points are the main results.
2. **How much of the gap is augmentation?** YOLOv8n from scratch at 256 px with JerseyNet's
   augmentation (no mosaic, no flip, the same scale and colour ranges).

Every run is selected on validation and scored on the test split with the same evaluator,
including AP by number size. Runs are ordered so that each seed is complete before the next
one starts; a time budget stops new runs from starting, and finished runs are kept, so the
study can be resumed by running it again on the same output folder.
"""

from __future__ import annotations

import dataclasses
import json
import time
import traceback
from pathlib import Path

import numpy as np

from jersey_detect.data import load_dataset
from jersey_detect.pipeline import _Log, _score, environment
from jersey_detect.yolo_baseline import SETTINGS as YOLO_SETTINGS
from jersey_detect.yolo_baseline import run_yolo

FRACTIONS = (0.1, 0.25)
_YOLO = {s.slug: s for s in YOLO_SETTINGS}
YOLO_SCRATCH = _YOLO["yolov8n_from_scratch_256_px"]
YOLO_COCO = _YOLO["yolov8n_coco_pretrained_256_px"]
YOLO_JN_AUG = _YOLO["yolov8n_from_scratch_256_px_jerseynet_augmentation"]
JERSEYNET = "JerseyNet, from scratch, 256 px"


def plan(seeds, fractions, aug_matched: bool = True) -> list[tuple[str, float, int]]:
    """(model, fraction, seed) in the order they are run."""
    jobs = []
    for seed in seeds:
        if aug_matched:
            jobs.append((YOLO_JN_AUG.name, 1.0, seed))
        for f in fractions:
            jobs += [(JERSEYNET, f, seed), (YOLO_SCRATCH.name, f, seed), (YOLO_COCO.name, f, seed)]
    return jobs


def _key(model: str, fraction: float, seed: int) -> str:
    return f"{model} | {fraction:g} | seed {seed}"


def _table(summary: dict) -> str:
    rows = ["# Data efficiency and augmentation-matched YOLOv8n", "",
            "Test split, mean ± standard deviation over the seeds that finished. "
            "AP by size: number side in pixels of the 128 px images (small < 8, medium 8-16, "
            "large > 16).", "",
            "| Model | Train fraction | Seeds | AP50:95 | AP50 | AP small | AP medium | AP large |",
            "|---|---|---|---|---|---|---|---|"]
    for agg in summary.get("aggregate", {}).values():
        def f(k, agg=agg):
            return f"{agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}"

        rows.append(f"| {agg['model']} | {agg['fraction']:g} | {agg['seeds']} | {f('ap50_95')} | "
                    f"{f('ap50')} | {f('ap_small')} | {f('ap_medium')} | {f('ap_large')} |")
    rows += ["", "## Runs", "", "| Run | Train images | Val AP50:95 | Test AP50:95 [95% CI] | Minutes |",
             "|---|---|---|---|---|"]
    for key, res in summary["runs"].items():
        ci = res["test"].get("ci95", {}).get("ap50_95")
        test = f"{res['test']['ap50_95']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        rows.append(f"| {key} | {res['train_images']} | {res['val']['ap50_95']:.3f} | {test} | "
                    f"{res['minutes']:.0f} |")
    sections = (("Runs that failed", summary["errors"]), ("Skipped (time budget)", summary["skipped"]))
    for title, items in sections:
        if items:
            rows += ["", f"## {title}", ""] + [f"- {e}" for e in items]
    return "\n".join(rows) + "\n"


def _aggregate(summary: dict) -> dict:
    groups: dict = {}
    for res in summary["runs"].values():
        groups.setdefault((res["model"], res["fraction"]), []).append(res)
    out = {}
    for (model, fraction), runs in groups.items():
        t = [r["test"] for r in runs]
        vals = {"ap50_95": [m["ap50_95"] for m in t], "ap50": [m["ap50"] for m in t],
                **{f"ap_{b}": [m["by_size"][b]["ap50_95"] for m in t] for b in ("small", "medium", "large")}}
        agg = {"model": model, "fraction": fraction, "seeds": sorted(r["seed"] for r in runs)}
        for k, v in vals.items():
            v = np.asarray(v)
            agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        out[f"{model} | {fraction:g}"] = agg
    return out


def run_study(data_root: str, out_dir: str, jn_base: dict, seeds=(0, 1, 2), fractions=FRACTIONS,
              aug_matched: bool = True, time_budget_hours: float = 10.0, n_boot: int = 1000,
              yolo_kwargs: dict | None = None, yolo_override: dict | None = None,
              min_hours_left: float = 0.0, log=None) -> dict:
    from jersey_detect.engine import (
        DetectorConfig,
        load_detector,
        pick_device,
        predict_records,
        train_detector,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = log or _Log(out / "run.log")
    started = time.time()
    splits = load_dataset(data_root)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.setdefault("runs", {})
    summary.update(errors=[], skipped=[], environment=environment(), jn_base=jn_base,
                   fractions=list(fractions), seeds=list(seeds))
    log(json.dumps(summary["environment"]))
    yolo_kwargs = yolo_kwargs or {}

    def save():
        summary["aggregate"] = _aggregate(summary)
        summary["elapsed_hours"] = round((time.time() - started) / 3600, 3)
        summary_path.write_text(json.dumps(summary, indent=2))
        (out / "results.md").write_text(_table(summary))

    def jerseynet(fraction: float, seed: int, remaining: float) -> dict:
        run_dir = out / "jerseynet" / f"f{fraction:g}_seed{seed}"
        scale = 1 / fraction
        cfg = DetectorConfig.from_dict({
            **jn_base, "data_root": data_root, "out_dir": str(run_dir), "seed": seed,
            "train_fraction": fraction,
            "epochs": int(round(jn_base.get("epochs", 140) * scale)),
            "warmup_epochs": int(round(jn_base.get("warmup_epochs", 3) * scale)),
            "eval_every": int(round(jn_base.get("eval_every", 5) * scale)),
            "max_hours": remaining - 0.1})
        info = train_detector(cfg, log=log)
        device = pick_device(cfg.device)
        model, _ = load_detector(run_dir / "best.pt", device)
        res = {"train": info}
        for split in ("val", "test"):
            d, s = predict_records(model, splits[split], cfg.image_size, device, batch_size=cfg.batch_size,
                                   top_k=cfg.top_k, num_workers=min(2, cfg.num_workers))
            res[split] = _score(JERSEYNET, splits[split], d, s, n_boot if split == "test" else 0)
        res["train_images"] = int(np.ceil(fraction * len(splits["train"])))
        res["minutes"] = info["hours"] * 60
        return res

    def yolo(model: str, fraction: float, seed: int, remaining: float) -> dict:
        setting = next(s for s in YOLO_SETTINGS if s.name == model)
        if yolo_override:  # smoke tests only
            setting = dataclasses.replace(setting, **yolo_override)
        sub = out / "yolo" / f"{setting.slug}_f{fraction:g}"
        ysum = run_yolo(data_root, str(sub), seeds=(seed,), time_budget_hours=remaining,
                        settings=(setting,), n_boot=n_boot, log=log, train_fraction=fraction,
                        **yolo_kwargs)
        r = ysum["results"].get(f"{setting.name} (seed {seed})")
        if r is None:
            raise RuntimeError(f"YOLOv8n run failed: {ysum['errors']}")
        return {"train": r["train"], "val": r["val"], "test": r["test"],
                "train_images": r["train"]["train_images"], "minutes": r["train"]["minutes"]}

    for model, fraction, seed in plan(seeds, fractions, aug_matched):
        key = _key(model, fraction, seed)
        if key in summary["runs"]:
            log(f"already done: {key}")
            continue
        remaining = time_budget_hours - (time.time() - started) / 3600
        if remaining < min_hours_left:
            log(f"skipping {key}: {remaining:.2f} h left")
            summary["skipped"].append(key)
            save()
            continue
        log(f"=== {key}")
        try:
            run = jerseynet if model == JERSEYNET else (lambda f, s, r, m=model: yolo(m, f, s, r))
            res = run(fraction, seed, remaining)
            res.update(model=model, fraction=fraction, seed=seed)
            summary["runs"][key] = res
            log(json.dumps({"run": key, "val_ap50_95": round(res["val"]["ap50_95"], 4),
                            "test_ap50_95": round(res["test"]["ap50_95"], 4),
                            "minutes": round(res["minutes"], 1)}))
        except Exception as exc:  # noqa: BLE001 - one failed run must not stop the study
            log(traceback.format_exc())
            summary["errors"].append(f"{key}: {type(exc).__name__}: {exc}")
        save()
    log("=== done")
    save()
    return summary
