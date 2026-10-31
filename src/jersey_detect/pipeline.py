"""End-to-end experiment: dataset report, leakage check, baselines, JerseyNet over several seeds.

Every stage runs inside its own error guard and ``summary.json`` is rewritten after each one,
so a failure late in a long run never loses the results already produced. The test split is
only used after a model is fully trained and its checkpoint has been fixed on validation.

Output layout (``out_dir``)::

    summary.json   results.md   dataset.json   leakage.json   environment.json   run.log
    runs/mean_box/metrics.json
    runs/legacy_regressor/{legacy.pt, metrics.json}
    runs/jersey_net_seed<k>/{best.pt, last.pt, history.jsonl, config.json, metrics.json,
                             test_predictions.json}
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from jersey_detect import __version__
from jersey_detect.data import load_dataset, split_stats
from jersey_detect.metrics import all_metrics, bootstrap_ci


@dataclass
class PipelineConfig:
    data_root: str
    out_dir: str
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    detector: dict = field(default_factory=dict)  # overrides for DetectorConfig
    time_budget_hours: float = 10.5
    run_legacy: bool = True
    legacy_epochs: int = 15
    leakage: bool = True
    bootstrap: int = 1000


class _Log:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def environment() -> dict:
    info = {"package_version": __version__, "python": platform.python_version(),
            "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    try:
        repo = Path(__file__).resolve().parents[2]
        info["git_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    return info


def _predictions_json(records, dets, scores) -> list[dict]:
    return [{"image": Path(r.path).name, "boxes_xyxy_norm": np.round(d, 5).tolist(),
             "scores": np.round(s, 5).tolist()} for r, d, s in zip(records, dets, scores, strict=True)]


def _score(name, records, dets, scores, n_boot) -> dict:
    gts = [r.boxes for r in records]
    m = all_metrics(gts, dets, scores)
    if n_boot:
        m["ci95"] = bootstrap_ci(gts, dets, scores, n_boot=n_boot)
    return m


def _fmt(v, ci=None) -> str:
    if v is None:
        return "–"
    s = f"{v:.3f}"
    if ci:
        s += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return s


def results_table(summary: dict) -> str:
    rows = ["| Model | Split | AP50:95 | AP50 | AP75 | AR@100 | First-box mean IoU | First-box IoU ≥ 0.5 |",
            "|---|---|---|---|---|---|---|---|"]
    for model, splits in summary.get("results", {}).items():
        for split in ("val", "test"):
            m = splits.get(split)
            if not m:
                continue
            ci = m.get("ci95", {}) if split == "test" else {}
            rows.append(
                f"| {model} | {split} | {_fmt(m['ap50_95'], ci.get('ap50_95'))} | "
                f"{_fmt(m['ap50'], ci.get('ap50'))} | {_fmt(m['ap75'])} | {_fmt(m['ar'])} | "
                f"{_fmt(m['first_box_mean_iou'], ci.get('first_box_mean_iou'))} | "
                f"{m['first_box_iou_ge_0_5'] * 100:.1f}% |"
            )
    agg = summary.get("jersey_net_over_seeds")
    lines = ["# Results", "", "Test intervals are 95% bootstrap intervals over images.", "",
             *rows, ""]
    if agg:
        lines += ["## JerseyNet over seeds (test)", "",
                  f"Seeds: {agg['seeds']}", "",
                  "| Metric | Mean | Std |", "|---|---|---|"]
        for k in ("ap50_95", "ap50", "first_box_mean_iou"):
            lines.append(f"| {k} | {agg[k]['mean']:.3f} | {agg[k]['std']:.3f} |")
        lines.append("")
    if summary.get("errors"):
        lines += ["## Stages that failed", ""] + [f"- {e['stage']}: {e['error']}" for e in summary["errors"]]
    return "\n".join(lines) + "\n"


def run_pipeline(cfg: PipelineConfig) -> dict:
    out = Path(cfg.out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    log = _Log(out / "run.log")
    started = time.time()
    summary: dict = {"config": asdict(cfg), "results": {}, "stages": {}, "errors": []}

    def save() -> None:
        summary["elapsed_hours"] = round((time.time() - started) / 3600, 3)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        (out / "results.md").write_text(results_table(summary))

    def stage(name, fn):
        log(f"=== {name}")
        t0 = time.time()
        try:
            res = fn()
            summary["stages"][name] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
            return res
        except Exception as exc:  # noqa: BLE001 - one failing stage must not stop the others
            log(traceback.format_exc())
            summary["stages"][name] = {"status": "failed", "seconds": round(time.time() - t0, 1)}
            summary["errors"].append({"stage": name, "error": f"{type(exc).__name__}: {exc}"})
            return None
        finally:
            save()

    env = environment()
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    summary["environment"] = env
    log(json.dumps(env))

    splits = stage("load_dataset", lambda: load_dataset(cfg.data_root))
    if splits is None:
        return summary

    def dataset_report():
        stats = {k: split_stats(v) for k, v in splits.items()}
        (out / "dataset.json").write_text(json.dumps(stats, indent=2))
        summary["dataset"] = stats
        log(json.dumps(stats))

    stage("dataset_report", dataset_report)

    if cfg.leakage:
        def leakage():
            from jersey_detect.leakage import leakage_report

            rep = leakage_report(splits)
            (out / "leakage.json").write_text(json.dumps(rep, indent=2))
            summary["leakage"] = rep
            log(json.dumps(rep))

        stage("leakage_check", leakage)

    def mean_box_stage():
        from jersey_detect.baselines import constant_predictions, mean_box

        box = mean_box(splits["train"])
        res = {"box": box.tolist()}
        for split in ("val", "test"):
            d, s = constant_predictions(box, splits[split])
            res[split] = _score("mean_box", splits[split], d, s, cfg.bootstrap if split == "test" else 0)
        run_dir = out / "runs" / "mean_box"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
        summary["results"]["Mean-box baseline"] = res

    stage("mean_box_baseline", mean_box_stage)

    if cfg.run_legacy:
        def legacy_stage():
            from jersey_detect.baselines import legacy_predict, train_legacy
            from jersey_detect.engine import pick_device

            device = pick_device(cfg.detector.get("device", "auto"))
            run_dir = out / "runs" / "legacy_regressor"
            model = train_legacy(cfg.data_root, str(run_dir), device, cfg.legacy_epochs, log=log)
            res = {}
            for split in ("val", "test"):
                d, s = legacy_predict(model, splits[split], device)
                res[split] = _score("legacy", splits[split], d, s, cfg.bootstrap if split == "test" else 0)
            (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
            summary["results"][f"v1 CNN regressor ({cfg.legacy_epochs} epochs)"] = res

        stage("legacy_regressor", legacy_stage)

    budget_s = cfg.time_budget_hours * 3600
    reserve_s = min(0.15 * 3600, 0.1 * budget_s)  # final evaluation and writing results
    min_run_s = min(0.2 * 3600, 0.1 * budget_s)  # not worth starting a run with less than this
    test_by_seed = {}
    for i, seed in enumerate(cfg.seeds):
        remaining = budget_s - (time.time() - started) - reserve_s
        if remaining < min_run_s:
            log(f"skipping seed {seed}: {remaining / 3600:.2f} h left in the budget")
            summary["stages"][f"jersey_net_seed{seed}"] = {"status": "skipped_time_budget"}
            save()
            continue
        share = remaining / (len(cfg.seeds) - i)

        def detector_stage(seed=seed, share=share):
            import torch

            from jersey_detect.engine import (
                DetectorConfig,
                load_detector,
                pick_device,
                predict_records,
                train_detector,
            )

            t_start = time.time()
            run_dir = out / "runs" / f"jersey_net_seed{seed}"
            dcfg = DetectorConfig.from_dict({**cfg.detector, "data_root": cfg.data_root,
                                             "out_dir": str(run_dir), "seed": seed,
                                             "max_hours": share / 3600})
            try:
                train_info = train_detector(dcfg, log=log)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                dcfg.batch_size = max(2, dcfg.batch_size // 2)
                dcfg.max_hours = max(0.1, (share - (time.time() - t_start)) / 3600)
                log(f"out of GPU memory; retrying seed {seed} with batch size {dcfg.batch_size}")
                train_info = train_detector(dcfg, log=log)
                train_info["note"] = f"batch size reduced to {dcfg.batch_size} after OOM"
            device = pick_device(dcfg.device)
            model, state = load_detector(run_dir / "best.pt", device)
            res = {"train": train_info}
            for split in ("val", "test"):
                d, s = predict_records(model, splits[split], dcfg.image_size, device,
                                       batch_size=dcfg.batch_size, top_k=dcfg.top_k,
                                       num_workers=min(2, dcfg.num_workers))
                res[split] = _score("jersey_net", splits[split], d, s,
                                    cfg.bootstrap if split == "test" else 0)
                if split == "test":
                    (run_dir / "test_predictions.json").write_text(
                        json.dumps(_predictions_json(splits[split], d, s)))
            (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
            summary["results"][f"JerseyNet, from scratch (seed {seed})"] = res
            test_by_seed[seed] = res["test"]

        stage(f"jersey_net_seed{seed}", detector_stage)

    if test_by_seed:
        agg = {"seeds": sorted(test_by_seed)}
        for k in ("ap50_95", "ap50", "first_box_mean_iou"):
            vals = np.array([m[k] for m in test_by_seed.values()])
            agg[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
        summary["jersey_net_over_seeds"] = agg
    log("=== done")
    save()
    return summary
