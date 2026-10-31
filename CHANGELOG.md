# Changelog

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
