# Changelog

## 2.1.0 (2026-11-07)

- AP by number size (small < 8 px, medium 8-16 px, large > 16 px at the original 128 px),
  following the area-range rules of `pycocotools`; added to every new run.
- `jersey-detect study` and `notebooks/kaggle_study.ipynb`: data-efficiency study (10% and 25%
  of the training images, nested subsets, same number of training steps) for JerseyNet and
  YOLOv8n from scratch and from COCO weights, plus YOLOv8n with JerseyNet's augmentation.
- `jersey-detect rescore` and `notebooks/kaggle_rescore.ipynb`: AP by size for saved checkpoints.
- `train_fraction` and `subset_seed` training settings.
- Results of the follow-up runs (YOLOv8n from scratch at 384 px, AP by size, data efficiency,
  augmentation-matched YOLOv8n) archived in `docs/experiments/` and summarised in the README.
- LaTeX preprint in `paper/`, with a script that builds its figures from the archived results.

## 2.0.0 (2026-10-31)

- New model: JerseyNet, an anchor-free CenterNet-style detector trained from scratch, which can
  predict any number of boxes per image.
- New evaluation: COCO-style AP over every annotated box (tested against `pycocotools`), images
  without labels counted as negatives, 95% bootstrap intervals, and the v1 first-box IoU kept for
  comparison.
- New pipeline and Kaggle notebook: dataset statistics, cross-split duplicate check, mean-box and
  v1 baselines, three seeds, resumable training with a time budget.
- `jersey-detect score` scores saved YOLO-format predictions, so pretrained detectors are
  evaluated with exactly the same code.
- Boxes are now `[x1, y1, x2, y2]` throughout.
- Removed the TFRecord reader and the single-box training code; both remain in 1.0.0.
- The 2025 coursework notebook and slides moved to `legacy/` (notebook outputs cleared).

## 1.0.0

- PyTorch port of the 2025 coursework model: one box and one presence score per image.
