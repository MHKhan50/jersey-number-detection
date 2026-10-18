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

Evaluation uses the first annotated box per image. Images without a ground-truth
box are not scored. Mean IoU is taken over scored images, and an image where the
model predicts no box counts as IoU 0.

Data: Roboflow `jersey-2` export (single class `number`, YOLO format), with
3,165 train, 301 validation and 148 test images. Some validation and test images
have no label and are not scored.

| Model | Split | Scored images | Mean IoU (positives) | Share with IoU ≥ 0.5 |
|---|---|---|---|---|
| Mean-box baseline | Validation | _to be confirmed_ | 0.027 | _not measured_ |
| Custom CNN (from scratch) | Validation | _re-run pending_ | _re-run pending_ | _re-run pending_ |
| YOLOv8n, fine-tuned 50 epochs (imgsz 640) | Validation | 263 | 0.593 | 70.3% |
| YOLOv8n, fine-tuned 50 epochs (imgsz 640) | Test | 130 | 0.581 | 68.5% |

The YOLOv8n row is a pretrained detector fine-tuned on this data, so it is not
the same architecture as the custom CNN. The custom CNN row will be re-run on
the same split before any comparison is drawn.

## Dataset

The dataset is a Roboflow export in TFRecord format (`persons-jersey.tfrecord`
in `train/` and `valid/`). It is not included in this repository. Create your
own download link on Roboflow Universe and export it. Never commit API keys or
download tokens.

Each example holds a JPEG (`image/encoded`) and bounding boxes
(`image/object/bbox/{ymin,xmin,ymax,xmax}`, normalised to 0-1) with a class
label. Images without a box are skipped.

The YOLO-format `jersey-2` export is also supported. Pass the split directory
(`jersey-2/train`, `jersey-2/valid` or `jersey-2/test`, each with `images/` and
`labels/`). Its single class is `number`, so a labelled image means a jersey
number is present. It is not restricted to the number 0. The presence metric
therefore means "a number is present" on this export, and the two datasets
should not be reported as the same task.

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

On the YOLO-format `jersey-2` export, pass the split directories instead:

```bash
jersey-detect train \
  --config configs/default.json \
  --train-record /data/jersey-2/train \
  --val-record /data/jersey-2/valid \
  --out-dir runs/jersey2

jersey-detect evaluate --checkpoint runs/jersey2/best.pt --data /data/jersey-2/test
```

`evaluate` reports the same metrics as training (box MSE, class accuracy,
mean IoU over labelled images) on any split, so validation and test numbers
can be compared directly.

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
