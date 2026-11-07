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

Test split (147 images, 234 boxes), all models scored with the same evaluator. Each detector row
is the mean ± standard deviation over three seeds (0, 1, 2), trained on a Kaggle T4 GPU.

| Model | Pretrained | Input | AP50:95 | AP50 | AP75 | First-box mean IoU | Train time per seed |
|---|---|---|---|---|---|---|---|
| Mean-box baseline | – | – | 0.000 | 0.000 | 0.000 | 0.016 | – |
| v1 CNN regressor (seed 0) | no | 128 | 0.000 | 0.000 | 0.000 | 0.051 | < 1 min |
| JerseyNet, baseline recipe | no | 256 | 0.241 ± 0.006 | 0.495 ± 0.004 | 0.227 ± 0.008 | 0.476 ± 0.023 | 22 min |
| **JerseyNet** + mosaic + GIoU | no | 384 | **0.308 ± 0.005** | **0.597 ± 0.006** | 0.294 ± 0.028 | 0.520 ± 0.004 | 65 min |
| YOLOv8n | no | 256 | 0.352 ± 0.013 | 0.629 ± 0.019 | 0.372 ± 0.005 | 0.544 ± 0.017 | 34 min |
| YOLOv8n | no | 384 | 0.366 ± 0.005 | 0.649 ± 0.003 | 0.371 ± 0.003 | 0.554 ± 0.016 | 33 min |
| YOLOv8n | COCO | 256 | 0.384 ± 0.003 | 0.675 ± 0.004 | 0.394 ± 0.017 | 0.576 ± 0.016 | 12 min |
| YOLOv8n | COCO | 640 | 0.419 ± 0.012 | 0.701 ± 0.008 | 0.476 ± 0.034 | 0.590 ± 0.007 | 22 min |

Per-seed tables with 95% bootstrap intervals are in
[`docs/experiments/2026-11-01_kaggle_t4/`](docs/experiments/2026-11-01_kaggle_t4/) (JerseyNet and
baselines, commit `8a40029`) and
[`docs/experiments/2026-11-01_kaggle_t4_yolo/`](docs/experiments/2026-11-01_kaggle_t4_yolo/)
(YOLOv8n, [`notebooks/kaggle_yolo_baseline.ipynb`](notebooks/kaggle_yolo_baseline.ipynb)) and
[`docs/experiments/2026-11-01_kaggle_t4_ablation/`](docs/experiments/2026-11-01_kaggle_t4_ablation/)
(ablation, [`notebooks/kaggle_ablation.ipynb`](notebooks/kaggle_ablation.ipynb)) and
[`docs/experiments/2026-11-02_kaggle_t4_yolo384/`](docs/experiments/2026-11-02_kaggle_t4_yolo384/)
(YOLOv8n from scratch at 384 px).
JerseyNet training curves: [`paper/figures/val_ap_curves.pdf`](paper/figures/val_ap_curves.pdf).

What the numbers say, read from the bottom row up (AP50:95):

* **Resolution** (640 → 256 px, both pretrained): −0.035.
* **Pretraining** (COCO → none, both 256 px): −0.032.
* **Design and training recipe** (YOLOv8n → JerseyNet, both from scratch at 256 px): −0.111.

So on this dataset COCO pretraining is worth little; most of the gap to YOLOv8n comes from the
detector design and its training recipe, which is where JerseyNet can be improved. JerseyNet
still turns the v1 network, which was no better than a constant box, into a working detector
(first-box IoU 0.05 → 0.48).

### Ablation: closing part of the recipe gap

Each variant changes one thing in the baseline recipe (seed 0). The variant with the best
validation AP50:95 was then trained with seeds 1 and 2; the test split played no part in the choice.

| Variant (seed 0) | Val AP50:95 | Test AP50:95 | Test AP50 |
|---|---|---|---|
| Baseline (256 px) | 0.259 | 0.235 [0.194, 0.290] | 0.474 |
| + GIoU box loss | 0.256 | 0.255 [0.212, 0.312] | 0.502 |
| + mosaic augmentation | 0.288 | 0.273 [0.228, 0.331] | 0.539 |
| 384 px input | 0.304 | 0.277 [0.234, 0.331] | 0.543 |
| + mosaic + GIoU | 0.293 | 0.258 [0.212, 0.317] | 0.510 |
| + mosaic + GIoU, 384 px | **0.337** | **0.303 [0.260, 0.358]** | **0.595** |

Over three seeds the combined recipe reaches 0.308 ± 0.005 AP50:95, up from 0.241 ± 0.006: a
28% relative gain from the training recipe and input size alone, still without pretraining.
Mosaic and a larger input each help on their own; GIoU alone changes little. Single-seed
differences between neighbouring rows are within the bootstrap intervals, so only the combined
result, repeated over three seeds, should be read as firm. The settings are in
[`configs/jersey_net_best.json`](configs/jersey_net_best.json).

At the matched input size of 384 px, both trained from scratch, YOLOv8n reaches 0.366 ± 0.005
against JerseyNet's 0.308 ± 0.005. The ablation therefore halves the design-and-recipe gap,
from 0.111 at 256 px (baseline recipes) to 0.058 at 384 px.

The YOLOv8n 640 px seed-0 run reproduces the earlier first-box result exactly (0.581, 68.5%),
and the mean-box baseline matches its earlier value (0.016), which cross-checks the evaluator.

### Where the remaining gap is

**Augmentation.** YOLOv8n from scratch at 256 px, trained with augmentation close to JerseyNet's
(no mosaic, mixup or flip, scale ±35%, saturation and value ±30%), drops from 0.352 ± 0.013 to
**0.280 ± 0.012**. About two-thirds of the 0.111 gap to JerseyNet is augmentation; the other 0.039
is architecture, losses and the rest of the recipe.

**Less data.** Nested subsets of the training images, same number of training steps
(test AP50:95; † two seeds, otherwise three):

| Model (256 px) | 10% (317 images) | 25% (792) | 100% (3,165) |
|---|---|---|---|
| JerseyNet, from scratch | 0.124 ± 0.011 | 0.183 ± 0.001 † | 0.241 ± 0.006 |
| YOLOv8n, from scratch | 0.265 ± 0.013 | 0.305 ± 0.012 † | 0.352 ± 0.013 |
| YOLOv8n, COCO | 0.302 ± 0.011 † | 0.349 ± 0.006 † | 0.384 ± 0.003 |
| COCO minus scratch | 0.037 | 0.044 | 0.032 |

COCO pretraining stays worth about 0.04 AP even with 317 images. JerseyNet keeps only half its
accuracy at 10% (YOLOv8n from scratch keeps three-quarters), which fits heavy augmentation acting
as extra data.

**Number size** (test AP50:95; small < 8 px, medium 8-16 px, large > 16 px at 128 px):

| Model | Small | Medium | Large |
|---|---|---|---|
| JerseyNet, 256 px | 0.043 ± 0.011 | 0.320 ± 0.018 | 0.463 ± 0.009 |
| JerseyNet + mosaic + GIoU, 384 px | 0.101 ± 0.012 | 0.408 ± 0.011 | 0.529 ± 0.005 |
| YOLOv8n, from scratch, 256 px | 0.087 ± 0.019 | 0.453 ± 0.015 | 0.609 ± 0.018 |
| YOLOv8n, from scratch, 384 px | 0.103 ± 0.007 | 0.491 ± 0.019 | 0.590 ± 0.019 |
| YOLOv8n, COCO, 256 px | 0.101 ± 0.012 | 0.485 ± 0.008 | 0.678 ± 0.008 |
| YOLOv8n, COCO, 640 px | 0.123 ± 0.009 | 0.542 ± 0.026 | 0.670 ± 0.021 |

Small numbers are hard for every model. At 384 px JerseyNet matches YOLOv8n on them; the remaining
gap is on medium and large numbers. COCO pretraining helps mostly on large numbers.

Archives: [`2026-11-04_kaggle_t4_rescore/`](docs/experiments/2026-11-04_kaggle_t4_rescore/)
(AP by size from the saved checkpoints, [`notebooks/kaggle_rescore.ipynb`](notebooks/kaggle_rescore.ipynb))
and [`2026-11-07_kaggle_t4_study/`](docs/experiments/2026-11-07_kaggle_t4_study/)
(data efficiency and augmentation, [`notebooks/kaggle_study.ipynb`](notebooks/kaggle_study.ipynb)).

## Paper

The preprint is in [`paper/`](paper/) (`main.tex`, `refs.bib`, figures built by
`python paper/make_figure.py` from the archived results). Build with `latexmk -pdf main.tex`.

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

The dataset is not included in this repository. Download it from
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
  metrics.py      first-box protocol, AP by number size, bootstrap intervals
  leakage.py      cross-split duplicate check
  baselines.py    mean-box baseline and the v1 CNN regressor
  pipeline.py     the full experiment
  ablation.py     ablation of the JerseyNet recipe
  yolo_baseline.py  YOLOv8n settings scored with the same evaluator
  study.py        data-efficiency study and augmentation-matched YOLOv8n
  rescore.py      AP by number size for saved checkpoints
  prepare.py      dataset discovery and Roboflow download
  cli.py          `jersey-detect` command
configs/          training and pipeline settings
notebooks/        Kaggle notebooks
legacy/           2025 coursework notebook and slides
paper/            preprint (LaTeX) and figure script
docs/experiments/ archived results of every Kaggle run
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
