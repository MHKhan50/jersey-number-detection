# Jersey Number Detection and Player Tracking

A small convolutional network, trained from scratch, that finds a player in a
sports image and reports whether jersey number **0** is present. It is a
single-object detector: one bounding box and one presence probability per image.

> **Scope, stated plainly.** The original project was framed as "jersey number
> classification". The training data has one label (`1`, the number `0`), and
> only the first annotated box in each image is used. The model therefore
> answers *"is the number 0 present, and where is the player?"*. It does not
> read arbitrary jersey numbers, and it does not track players across frames.

## Results

Results will be added once the training runs in [Reproducing](#reproducing)
have been completed on the official splits. No figures are reported yet.

| Split | Box MSE | Class accuracy | Mean IoU (positives) |
|---|---|---|---|
| Validation | _to be filled_ | _to be filled_ | _to be filled_ |

## Dataset

The dataset is a Roboflow export in TFRecord format (`persons-jersey.tfrecord`
in `train/` and `valid/`). It is not included in this repository. Create your
own download link on Roboflow Universe and export it. Never commit API keys or
download tokens.

Each example holds a JPEG (`image/encoded`) and bounding boxes
(`image/object/bbox/{ymin,xmin,ymax,xmax}`, normalised to 0-1) with a class
label. Images without a box are skipped.

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
```

## Reproducing

```bash
jersey-detect train \
  --config configs/default.json \
  --train-record data/train/persons-jersey.tfrecord \
  --val-record data/valid/persons-jersey.tfrecord \
  --out-dir runs/baseline
```

This writes `runs/baseline/best.pt`, `history.json` (per-epoch metrics) and
`config.json` (the exact settings used). Set `--seed` to repeat a run.

## Prediction

```bash
jersey-detect predict --checkpoint runs/baseline/best.pt \
  --image match_frame.jpg --out annotated.jpg
```

The command prints the probability and the box in pixel coordinates, and saves
an annotated copy.

## Docker

```bash
docker build -t jersey-detect .
docker run --rm -v "$PWD:/data" jersey-detect predict \
  --checkpoint /data/runs/baseline/best.pt --image /data/frame.jpg --out /data/out.jpg
```

## Tests

```bash
pytest -q
ruff check src tests
```

The data, metric and TFRecord tests run without PyTorch. Model and training
tests are skipped when PyTorch is not installed.

## Project layout

```
src/jersey_detect/
  tfrecord.py   TFRecord and tf.train.Example reader (no TensorFlow needed)
  data.py       resize to 128x128, keep first box, build samples
  metrics.py    IoU, mean IoU, binary accuracy
  model.py      the CNN (same layer sizes as the original Keras model)
  train.py      training loop with Adam, MSE for the box, BCE for presence
  predict.py    single-image inference and drawing
  cli.py        `jersey-detect train` and `jersey-detect predict`
configs/        default training settings
notebooks/      original exploration notebook (kept for reference)
tests/          pytest suite
```

## Differences from the original notebook

- PyTorch replaces TensorFlow/Keras. The layer sizes, optimiser (Adam, lr 1e-3)
  and losses are kept the same.
- The presence output uses a logit with `BCEWithLogitsLoss`, which is
  numerically equivalent to the original sigmoid with binary cross-entropy.
- Image resizing uses Pillow bilinear, not TensorFlow's resize. Results can
  differ slightly.
- TFRecord CRCs are not checked when reading.

## Limitations

- One object and one class per image. Other players and other numbers are ignored.
- Trained on a small custom dataset. Generalisation to other sports, cameras or
  lighting has not been measured.
- The original notebook split the data by its own train/valid folders. Check
  that no player appears in both splits before reporting results.

## Licence

Released under the [MIT Licence](LICENSE).

## Author

**Muzammal Hussain**, Software Engineer and Data Scientist
