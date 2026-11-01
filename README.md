# Jersey Number Localisation

[![CI](https://github.com/MHKhan50/jersey-number-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/MHKhan50/jersey-number-detection/actions/workflows/ci.yml)

Find the jersey numbers in a sports image. The main model, **JerseyNet**, is an anchor-free
detector in the style of CenterNet, trained from random initialisation (no ImageNet or COCO
weights). The repository also contains the baselines, the evaluation protocol and a Kaggle
notebook that runs the whole experiment unattended.

The project started as 2025 coursework (a TensorFlow notebook, kept in [`legacy/`](legacy/)).
Version 1 ported that network to PyTorch. Version 2 replaces it with a real detector, because
the v1 network regressed a single box from globally pooled features and barely beat a constant
prediction (see [Results](#results)).

## Task

Input: an image. Output: a box and a confidence score for every jersey number in it. The digits
are not read; this is the localisation stage that a recognition system (for example a scene-text
recogniser such as PARSeq) would run on.

## Results

Test split (147 images, 234 boxes), all models scored with the same evaluator. Each row is the
mean ± standard deviation over three seeds (0, 1, 2); both runs on a Kaggle T4 GPU.

| Model | Pretrained | Input | AP50:95 | AP50 | AP75 | First-box mean IoU | Train time per seed |
|---|---|---|---|---|---|---|---|
| Mean-box baseline | – | – | 0.000 | 0.000 | 0.000 | 0.016 | – |
| v1 CNN regressor (seed 0) | no | 128 | 0.000 | 0.000 | 0.000 | 0.051 | < 1 min |
| **JerseyNet** (this repository) | no | 256 | 0.241 ± 0.006 | 0.495 ± 0.004 | 0.227 ± 0.008 | 0.476 ± 0.023 | 22 min |
| YOLOv8n | no | 256 | 0.352 ± 0.013 | 0.629 ± 0.019 | 0.372 ± 0.005 | 0.544 ± 0.017 | 34 min |
| YOLOv8n | COCO | 256 | 0.384 ± 0.003 | 0.675 ± 0.004 | 0.394 ± 0.017 | 0.576 ± 0.016 | 12 min |
| YOLOv8n | COCO | 640 | 0.419 ± 0.012 | 0.701 ± 0.008 | 0.476 ± 0.034 | 0.590 ± 0.007 | 22 min |

Per-seed tables with 95% bootstrap intervals are in
[`docs/experiments/2026-11-01_kaggle_t4/`](docs/experiments/2026-11-01_kaggle_t4/) (JerseyNet and
baselines, commit `8a40029`) and
[`docs/experiments/2026-11-01_kaggle_t4_yolo/`](docs/experiments/2026-11-01_kaggle_t4_yolo/)
(YOLOv8n, [`notebooks/kaggle_yolo_baseline.ipynb`](notebooks/kaggle_yolo_baseline.ipynb)).
JerseyNet training curves: [`docs/paper/figures/val_ap_curves.png`](docs/paper/figures/val_ap_curves.png).

What the numbers say, read from the bottom row up (AP50:95):

* **Resolution** (640 → 256 px, both pretrained): −0.035.
* **Pretraining** (COCO → none, both 256 px): −0.032.
* **Design and training recipe** (YOLOv8n → JerseyNet, both from scratch at 256 px): −0.111.

So on this dataset COCO pretraining is worth little; most of the gap to YOLOv8n comes from the
detector design and its training recipe, which is where JerseyNet can be improved. JerseyNet
still turns the v1 network, which was no better than a constant box, into a working detector
(first-box IoU 0.05 → 0.48).

The YOLOv8n 640 px seed-0 run reproduces the earlier first-box result exactly (0.581, 68.5%),
and the mean-box baseline matches its earlier value (0.016), which cross-checks the evaluator.

## Dataset

Roboflow export `jersey-2` (one class, `number`, YOLO format). The images are ice-hockey scenes,
mostly NHL broadcast frames and press photos, from close-ups to wide rink views; all are
128×128 pixels. Roboflow lists the export as public domain, but the underlying photographs may
carry their own rights, so the dataset is used for research only and is not redistributed here.

| Split | Images | Labelled images | Boxes | Max boxes per image |
|---|---|---|---|---|
| Train | 3,165 | 2,784 | 4,941 | 9 |
| Validation | 301 | 263 | 485 | 6 |
| Test | 147 | 130 | 234 | 9 |

The median box side is about 8% of the image (about 10 pixels), and one in ten is under 3%.
Duplicate check (difference hash, distance ≤ 4): 9 validation images and 1 test image have a
near-identical training image; no Roboflow source names are shared across splits.

The dataset It is not included in this repository. Download it from
Roboflow in **YOLOv8** format; the expected layout is:

```
jersey-2/
  train/images/*.jpg   train/labels/*.txt
  valid/images/        valid/labels/
  test/images/         test/labels/
```

Images without a label file, or with an empty one, are kept as negatives. Never commit API keys
or download links: the code reads `ROBOFLOW_API_KEY` from the environment (or a Kaggle secret)
and does not store it.

`jersey-detect stats --data jersey-2` prints counts, box sizes and a cross-split duplicate check.

## Evaluation protocol

* **Primary metric:** COCO-style AP50:95, with AP50, AP75 and AR@100. Every annotated box counts,
  and a box predicted on an image with no label is a false positive. The implementation is in
  [`coco_eval.py`](src/jersey_detect/coco_eval.py) and is tested against `pycocotools`.
* **v1 comparison:** the highest-scoring box against the first annotated box, as mean IoU and the
  share of images with IoU ≥ 0.5.
* **Model selection** uses validation AP50:95 only. The test split is scored once per trained
  model and never influences training or checkpoint choice.
* **Uncertainty:** the test split is small, so test metrics are reported with a 95% bootstrap
  interval over images, and JerseyNet is trained with three seeds (mean and standard deviation).
* **Leakage check:** before training, the pipeline reports images that share a Roboflow source
  name or are near-identical (64-bit difference hash, Hamming distance ≤ 4) across splits. The
  official splits are not changed, so results stay comparable; the counts are reported.
* **Pretrained detectors:** save their predictions as YOLO text files with confidences
  (Ultralytics `save_txt=True, save_conf=True`) and score them with the same code:
  `jersey-detect score --data jersey-2 --split test --predictions runs/detect/predict/labels`.

## Model

| Part | Design |
|---|---|
| Backbone | Residual stages with batch norm at strides 4, 8, 16, 32 (64, 128, 192, 256 channels) |
| Neck | Top-down feature pyramid fused back to stride 4 (jersey numbers are small) |
| Heads | Centre heatmap (penalty-reduced focal loss), width/height and offset (L1) |
| Decoding | 3×3 max-pool peak picking, top 100; no anchors and no NMS |
| Input | 256×256 letterbox (the 128×128 source images are upsampled 2× so the smallest numbers still cover two cells of the stride-4 output); training adds random scale (0.6–1.3), placement and colour jitter. No horizontal flip, which would mirror the digits |
| Training | AdamW, lr 1e-3 with warm-up and cosine decay, weight decay 1e-4, batch 64, mixed precision on GPU, 140 epochs |

About 4.7 million parameters, all trained from scratch. Settings are in
[`configs/jersey_net.json`](configs/jersey_net.json) and [`configs/kaggle.json`](configs/kaggle.json).

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
```

## Running

Full experiment (dataset report, leakage check, baselines, JerseyNet over three seeds):

```bash
jersey-detect run --data /data/jersey-2 --out results --config configs/kaggle.json
```

Single training run, evaluation and prediction:

```bash
jersey-detect train --data /data/jersey-2 --out runs/seed0 --config configs/jersey_net.json --seed 0
jersey-detect evaluate --checkpoint runs/seed0/best.pt --data /data/jersey-2 --split test
jersey-detect predict --checkpoint runs/seed0/best.pt --images frame.jpg --out annotated/
```

Training writes `last.pt` after every epoch and resumes from it when the same command is run
again with the same settings.

### Kaggle

[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) runs the full experiment.
Import it with *File → Import Notebook → Link* and this repository's URL to the notebook, attach
the dataset as an input (or add a `ROBOFLOW_API_KEY` secret), choose *GPU T4 x2*, turn Internet
on and use *Save & Run All*. Results are written to `/kaggle/working/results`.

## Tests

```bash
pytest -q
ruff check src tests
```

The suite covers data loading, the AP implementation (including agreement with `pycocotools`),
target encoding and decoding, resuming, the time budget, the full pipeline on a synthetic dataset,
and an end-to-end run of the Kaggle notebook in CI.

## Project layout

```
src/jersey_detect/
  data.py         YOLO split loader, dataset statistics
  transforms.py   letterbox, augmentation, CenterNet targets (NumPy only)
  detector.py     JerseyNet, focal loss, decoding
  engine.py       training loop (AMP, resume, time budget) and inference
  coco_eval.py    COCO-style AP in NumPy
  metrics.py      first-box protocol, bootstrap intervals
  leakage.py      cross-split duplicate check
  baselines.py    mean-box baseline and the v1 CNN regressor
  pipeline.py     the full experiment
  prepare.py      dataset discovery and Roboflow download
  cli.py          `jersey-detect` command
configs/          training and pipeline settings
notebooks/        Kaggle notebook
legacy/           2025 coursework notebook and slides
docs/paper/       paper draft
tests/            pytest suite
```

## Limitations

* One dataset with a small test split (147 images). Intervals are wide and are reported.
* Frames from the same match can appear in different splits of a Roboflow export. The duplicate
  check measures this; it does not remove it.
* Localisation only. The digits are not read.
* GPU training is seeded but not bit-for-bit deterministic (cuDNN autotuning is on for speed).
* JerseyNet and YOLOv8n differ in input size and pretraining, so that comparison is between
  systems, not architectures.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Licence

[MIT](LICENSE).

## Author

**Muzammal Hussain**, Software Engineer, Data Scientist and AI Researcher
