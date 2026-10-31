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

**Status:** the v2 runs are in progress. The table will be filled from `results/results.md`
once the Kaggle run has finished. No v2 numbers are reported until then.

The v1 numbers below were measured with the old protocol (first annotated box only, labelled
images only) and are kept for reference.

| Model (v1 protocol) | Test mean IoU | Test IoU ≥ 0.5 |
|---|---|---|
| Mean-box baseline | 0.016 | – |
| v1 CNN regressor, from scratch, 15 epochs | 0.044 | – |
| YOLOv8n, COCO-pretrained, fine-tuned 50 epochs | 0.581 | 68.5% |

## Dataset

Roboflow export `jersey-2` (one class, `number`, YOLO format, public domain), with 3,165 train,
301 validation and 148 test images. It is not included in this repository. Download it from
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
| Input | 512×512 letterbox; training adds random scale (0.6–1.3), placement and colour jitter. No horizontal flip, which would mirror the digits |
| Training | AdamW, lr 1e-3 with warm-up and cosine decay, weight decay 1e-4, batch 32, mixed precision on GPU, 140 epochs |

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

* One dataset with a small test split (148 images). Intervals are wide and are reported.
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
