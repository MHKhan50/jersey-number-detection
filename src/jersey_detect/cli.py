"""Command-line interface.

    jersey-detect run       --data DIR --out DIR [--config configs/kaggle.json]
    jersey-detect train     --data DIR --out DIR [--config configs/jersey_net.json] [--seed N]
    jersey-detect evaluate  --checkpoint best.pt --data DIR [--split test]
    jersey-detect score     --data DIR --split test --predictions DIR   (YOLO txt with confidence)
    jersey-detect predict   --checkpoint best.pt --images a.jpg b.jpg --out DIR
    jersey-detect stats     --data DIR
    jersey-detect synthetic --out DIR
    jersey-detect yolo      --data DIR --out DIR [--hours 9.5]   (needs `pip install ultralytics`)
    jersey-detect ablate    --data DIR --out DIR [--config configs/jersey_net.json] [--hours 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_yolo_predictions(pred_dir: str, records) -> tuple[list, list]:
    """Read ``class cx cy w h conf`` files (Ultralytics ``save_txt=True, save_conf=True``)."""
    from jersey_detect.boxes import yolo_to_xyxy

    dets, scores = [], []
    for r in records:
        f = Path(pred_dir) / f"{r.stem}.txt"
        rows = []
        if f.exists():
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) >= 6:
                    rows.append([*yolo_to_xyxy(*map(float, p[1:5])), float(p[5])])
        arr = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
        dets.append(np.clip(arr[:, :4], 0, 1))
        scores.append(arr[:, 4])
    return dets, scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jersey-detect")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="full experiment: baselines + JerseyNet over seeds")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="JSON with PipelineConfig fields")
    p.add_argument("--hours", type=float, help="overall time budget in hours")

    p = sub.add_parser("train", help="train one JerseyNet run")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="JSON with DetectorConfig fields")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)

    p = sub.add_parser("evaluate", help="evaluate a JerseyNet checkpoint on a split")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])

    p = sub.add_parser("score", help="score saved YOLO-format predictions with the same protocol")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--predictions", required=True)

    p = sub.add_parser("predict", help="draw detections on images")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.3)

    p = sub.add_parser("stats", help="dataset statistics and cross-split duplicate check")
    p.add_argument("--data", required=True)

    p = sub.add_parser("yolo", help="YOLOv8n baselines scored with the same evaluator")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hours", type=float, default=9.5, help="stop starting new runs after this")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--only", nargs="+", metavar="SLUG",
                   help="run only these settings, e.g. yolov8n_from_scratch_384_px")
    p.add_argument("--smoke", action="store_true", help="one tiny run, for testing the setup")

    p = sub.add_parser("ablate", help="ablation study of JerseyNet training choices")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="JSON with the baseline DetectorConfig settings")
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--smoke", action="store_true", help="tiny settings, for testing the setup")

    p = sub.add_parser("study", help="data-efficiency study and augmentation-matched YOLOv8n")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="JSON with the JerseyNet DetectorConfig settings")
    p.add_argument("--hours", type=float, default=10.0, help="no new run starts after this")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25])
    p.add_argument("--smoke", action="store_true", help="tiny settings, for testing the setup")

    p = sub.add_parser("rescore", help="re-score saved checkpoints on test, with AP by number size")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True, help="folders searched for best.pt")
    p.add_argument("--out", required=True)

    p = sub.add_parser("synthetic", help="write a small synthetic dataset for smoke tests")
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.command == "run":
        from jersey_detect.pipeline import PipelineConfig, run_pipeline

        settings = _load_json(args.config)
        if args.hours is not None:
            settings["time_budget_hours"] = args.hours
        summary = run_pipeline(PipelineConfig(data_root=args.data, out_dir=args.out, **settings))
        print(json.dumps({"errors": summary["errors"], "stages": summary["stages"]}, indent=2))
        return 1 if summary["errors"] else 0

    if args.command == "train":
        from jersey_detect.engine import DetectorConfig, train_detector

        settings = _load_json(args.config)
        settings.update(data_root=args.data, out_dir=args.out)
        for key in ("seed", "epochs"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        print(json.dumps(train_detector(DetectorConfig.from_dict(settings)), indent=2))
        return 0

    if args.command == "evaluate":
        from jersey_detect.data import load_dataset
        from jersey_detect.engine import evaluate_records, load_detector, pick_device

        device = pick_device()
        model, state = load_detector(args.checkpoint, device)
        records = load_dataset(args.data)[args.split]
        print(json.dumps(evaluate_records(model, records, state["config"]["image_size"], device), indent=2))
        return 0

    if args.command == "score":
        from jersey_detect.data import load_dataset
        from jersey_detect.metrics import all_metrics, bootstrap_ci

        records = load_dataset(args.data)[args.split]
        dets, scores = read_yolo_predictions(args.predictions, records)
        gts = [r.boxes for r in records]
        res = all_metrics(gts, dets, scores)
        res["ci95"] = bootstrap_ci(gts, dets, scores)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "predict":
        from jersey_detect.predict import predict_images

        print(json.dumps(predict_images(args.checkpoint, args.images, args.out, args.threshold), indent=2))
        return 0

    if args.command == "stats":
        from jersey_detect.data import load_dataset, split_stats
        from jersey_detect.leakage import leakage_report

        splits = load_dataset(args.data)
        print(json.dumps({"splits": {k: split_stats(v) for k, v in splits.items()},
                          "leakage": leakage_report(splits)}, indent=2))
        return 0

    if args.command == "ablate":
        from jersey_detect.ablation import SMOKE_VARIANTS, run_ablation

        base = _load_json(args.config)
        kwargs = {}
        if args.smoke:
            base.update(image_size=128, epochs=1, batch_size=8, num_workers=0, eval_every=1,
                        warmup_epochs=1, amp=False)
            kwargs = {"variants": SMOKE_VARIANTS, "extra_seeds": (1,), "n_boot": 20}
        summary = run_ablation(args.data, args.out, base, time_budget_hours=args.hours, **kwargs)
        print(json.dumps({"errors": summary["errors"], "best": summary.get("best_variant")}, indent=2))
        return 1 if summary["errors"] else 0

    if args.command == "yolo":
        from jersey_detect.yolo_baseline import SETTINGS, SMOKE_SETTINGS, run_yolo

        smoke = {"batch": 4, "workers": 0, "n_boot": 20, "device": "cpu"} if args.smoke else {}
        settings = SMOKE_SETTINGS if args.smoke else SETTINGS
        if args.only:
            unknown = set(args.only) - {s.slug for s in settings}
            if unknown:
                parser.error(f"unknown setting(s) {sorted(unknown)}; choose from "
                             f"{[s.slug for s in settings]}")
            settings = tuple(s for s in settings if s.slug in args.only)
        summary = run_yolo(args.data, args.out, seeds=args.seeds, time_budget_hours=args.hours,
                           settings=settings, **smoke)
        print(json.dumps({"errors": summary["errors"], "done": sorted(summary["results"])}, indent=2))
        return 1 if summary["errors"] else 0

    if args.command == "study":
        from jersey_detect.study import run_study

        base = _load_json(args.config)
        kwargs = {}
        if args.smoke:
            base.update(image_size=128, epochs=1, batch_size=8, num_workers=0, eval_every=1,
                        warmup_epochs=1, amp=False)
            kwargs = {"n_boot": 20, "yolo_kwargs": {"batch": 4, "workers": 0, "device": "cpu"},
                      "yolo_override": {"imgsz": 64, "epochs": 1}}
        summary = run_study(args.data, args.out, base, seeds=args.seeds, fractions=args.fractions,
                            time_budget_hours=args.hours, **kwargs)
        print(json.dumps({"errors": summary["errors"], "done": sorted(summary["runs"])}, indent=2))
        return 1 if summary["errors"] else 0

    if args.command == "rescore":
        from jersey_detect.rescore import rescore

        summary = rescore(args.data, args.checkpoints, args.out)
        print((Path(args.out) / "results.md").read_text())
        return 0 if summary["results"] else 1

    from jersey_detect.synthetic import make_dataset

    print(make_dataset(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
