"""Ablation study: which parts of a modern detector recipe help JerseyNet.

Stage 1 trains every variant once (seed 0). Stage 2 picks the variant with the best
**validation** AP50:95 and trains it with two more seeds, so the reported improvement has a
mean and a standard deviation. The test split never takes part in the choice.

Variants (each on top of the baseline recipe in ``configs/jersey_net.json``):

* ``baseline``: CenterNet recipe as in the main results;
* ``mosaic``: 4-image mosaic augmentation on half of the training samples;
* ``giou``: an additional GIoU box loss (weight 1);
* ``384px``: input 384 instead of 256 px (3x upsampling of the 128 px images);
* ``mosaic+giou`` and ``mosaic+giou+384px``: the combinations.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from jersey_detect.data import load_dataset
from jersey_detect.pipeline import _Log, _predictions_json, _score, environment

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "mosaic": {"mosaic": 0.5},
    "giou": {"giou_weight": 1.0},
    "384px": {"image_size": 384},
    "mosaic+giou": {"mosaic": 0.5, "giou_weight": 1.0},
    "mosaic+giou+384px": {"mosaic": 0.5, "giou_weight": 1.0, "image_size": 384},
}
SMOKE_VARIANTS = {"baseline": {}, "mosaic+giou": {"mosaic": 0.5, "giou_weight": 1.0}}


def _table(summary: dict) -> str:
    rows = ["# JerseyNet ablation", "",
            "Stage 1: every variant with seed 0. Stage 2: the variant with the best validation "
            "AP50:95 with seeds 1 and 2. Test intervals are 95% bootstrap intervals over images.", "",
            "| Variant | Seed | Val AP50:95 | Test AP50:95 | Test AP50 | Test AP75 "
            "| Test first-box IoU | Minutes |",
            "|---|---|---|---|---|---|---|---|"]
    for key, res in summary["runs"].items():
        v, t = res["val"], res["test"]
        ci = t.get("ci95", {}).get("ap50_95")
        test_ap = f"{t['ap50_95']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        rows.append(f"| {res['variant']} | {res['seed']} | {v['ap50_95']:.3f} | {test_ap} | "
                    f"{t['ap50']:.3f} | {t['ap75']:.3f} | {t['first_box_mean_iou']:.3f} | "
                    f"{res['train']['hours'] * 60:.0f} |")
    best = summary.get("best_variant")
    if best:
        rows += ["", f"Best variant on validation (seed 0): **{best}**", ""]
    agg = summary.get("best_over_seeds")
    if agg:
        rows += [f"## {best} over seeds {agg['seeds']} (test)", "",
                 "| Metric | Mean | Std |", "|---|---|---|"]
        rows += [f"| {k} | {agg[k]['mean']:.3f} | {agg[k]['std']:.3f} |"
                 for k in ("ap50_95", "ap50", "ap75", "first_box_mean_iou")]
    if summary["errors"]:
        rows += ["", "## Runs that failed", ""] + [f"- {e['run']}: {e['error']}" for e in summary["errors"]]
    return "\n".join(rows) + "\n"


def run_ablation(data_root: str, out_dir: str, base: dict, variants: dict[str, dict] = VARIANTS,
                 extra_seeds=(1, 2), time_budget_hours: float = 10.0, n_boot: int = 1000,
                 log=None) -> dict:
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
    summary.update(base=base, variants=variants, errors=[], skipped=[], environment=environment())
    log(json.dumps(summary["environment"]))

    def save():
        summary["elapsed_hours"] = round((time.time() - started) / 3600, 3)
        summary_path.write_text(json.dumps(summary, indent=2))
        (out / "results.md").write_text(_table(summary))

    def run(variant: str, seed: int):
        key = f"{variant} (seed {seed})"
        run_dir = out / "runs" / f"{variant.replace('+', '_')}_seed{seed}"
        if key in summary["runs"] and (run_dir / "metrics.json").exists():
            log(f"already done: {key}")
            return
        remaining = time_budget_hours * 3600 - (time.time() - started)
        if remaining < 0.25 * 3600:
            log(f"skipping {key}: {remaining / 3600:.2f} h left")
            summary["skipped"].append(key)
            return
        log(f"=== {key}")
        try:
            cfg = DetectorConfig.from_dict({**base, **variants[variant], "data_root": data_root,
                                            "out_dir": str(run_dir), "seed": seed,
                                            "max_hours": remaining / 3600 - 0.1})
            info = train_detector(cfg, log=log)
            device = pick_device(cfg.device)
            model, _ = load_detector(run_dir / "best.pt", device)
            res = {"variant": variant, "seed": seed, "settings": variants[variant], "train": info}
            for split in ("val", "test"):
                d, s = predict_records(model, splits[split], cfg.image_size, device,
                                       batch_size=cfg.batch_size, top_k=cfg.top_k,
                                       num_workers=min(2, cfg.num_workers))
                res[split] = _score(variant, splits[split], d, s, n_boot if split == "test" else 0)
                if split == "test":
                    (run_dir / "test_predictions.json").write_text(
                        json.dumps(_predictions_json(splits[split], d, s)))
            (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
            summary["runs"][key] = res
            log(json.dumps({"run": key, "val_ap50_95": res["val"]["ap50_95"],
                            "test_ap50_95": res["test"]["ap50_95"]}))
        except Exception as exc:  # noqa: BLE001 - one failed run must not stop the study
            log(traceback.format_exc())
            summary["errors"].append({"run": key, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            save()

    for variant in variants:  # stage 1
        run(variant, 0)

    seed0 = {v: summary["runs"][f"{v} (seed 0)"]["val"]["ap50_95"]
             for v in variants if f"{v} (seed 0)" in summary["runs"]}
    if seed0:
        best = max(seed0, key=seed0.get)
        summary["best_variant"] = best
        log(f"best variant on validation: {best} ({seed0[best]:.4f})")
        for seed in extra_seeds:  # stage 2
            run(best, seed)
        done = [summary["runs"][k]["test"] for k in summary["runs"] if summary["runs"][k]["variant"] == best]
        seeds = sorted(summary["runs"][k]["seed"] for k in summary["runs"]
                       if summary["runs"][k]["variant"] == best)
        agg = {"seeds": seeds}
        for k in ("ap50_95", "ap50", "ap75", "first_box_mean_iou"):
            v = np.array([m[k] for m in done])
            agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        summary["best_over_seeds"] = agg
    log("=== done")
    save()
    return summary
