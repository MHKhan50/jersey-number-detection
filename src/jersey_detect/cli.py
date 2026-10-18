"""Command-line interface: `jersey-detect train ...` and `jersey-detect predict ...`."""

from __future__ import annotations

import argparse
import json

from jersey_detect.train import TrainConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jersey-detect")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train the detector on TFRecord files")
    train_p.add_argument("--config", help="JSON file with default training settings")
    train_p.add_argument("--train-record")
    train_p.add_argument("--val-record")
    train_p.add_argument("--out-dir")
    train_p.add_argument("--epochs", type=int)
    train_p.add_argument("--batch-size", type=int)
    train_p.add_argument("--learning-rate", type=float)
    train_p.add_argument("--seed", type=int)
    train_p.add_argument("--box-loss", choices=["mse", "l1", "giou"])
    train_p.add_argument("--select-by", choices=["loss", "iou"])

    pred_p = sub.add_parser("predict", help="detect on one image and save an annotated copy")
    pred_p.add_argument("--checkpoint", required=True)
    pred_p.add_argument("--image", required=True)
    pred_p.add_argument("--out", required=True)

    eval_p = sub.add_parser(
        "evaluate", help="score a checkpoint on a TFRecord file or a YOLO split directory"
    )
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--data", required=True, help="TFRecord file or YOLO split directory")

    args = parser.parse_args(argv)
    if args.command == "evaluate":
        from jersey_detect.evaluate import evaluate_checkpoint

        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.data)))
        return 0
    if args.command == "train":
        from jersey_detect.train import train

        settings = {}
        if args.config:
            with open(args.config, encoding="utf-8") as fh:
                settings = json.load(fh)
        for key in ("train_record", "val_record", "out_dir", "epochs", "batch_size",
                    "learning_rate", "seed", "box_loss", "select_by"):
            value = getattr(args, key.replace("-", "_"), None)
            if value is not None:
                settings[key] = value
        if "train_record" not in settings or "val_record" not in settings:
            parser.error("train needs --train-record and --val-record (or them in --config)")
        final = train(TrainConfig(**settings))
        print(json.dumps({"final": final}))
        return 0

    from jersey_detect.predict import predict_file

    result = predict_file(args.checkpoint, args.image, args.out)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
