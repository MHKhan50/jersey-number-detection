# Jersey-number presence and box detection: a from-scratch CNN versus a fine-tuned YOLOv8n

Muzammal Hussain

> Draft. Sections marked [PENDING] depend on runs that are not finished yet.

## Abstract

We study whether a small convolutional network trained from scratch can locate a
jersey number in a sports image, and how it compares with a fine-tuned YOLOv8n
detector on the same data. Both models are evaluated on a public Roboflow export
(`jersey-2`, one class) with the same split and the same scoring rule. On the
test split (130 labelled images), the fine-tuned YOLOv8n reaches a mean IoU of
0.581, the scratch CNN reaches 0.044, and a fixed mean-box baseline reaches 0.016.
The scratch CNN is only marginally better than the baseline, which we analyse in
the ablation section. [PENDING: ablation results]

## 1. Introduction

Jersey-number recognition is usually framed as reading digits, but many
practical systems first need to know where the number is. This paper treats the
task as single-object localisation: given an image, predict one box around the
annotated number. We ask two questions. First, how much of the signal does a
small network trained from scratch recover? Second, how much does a
pretrained detector gain from fine-tuning on a small dataset?

Our contributions are:

1. A reproducible evaluation protocol for single-box localisation on a YOLO-format
   export, including how unlabelled images are handled.
2. A comparison of a from-scratch CNN, a fine-tuned YOLOv8n, and a mean-box
   baseline on identical splits.
3. An ablation of the scratch CNN's box loss, training length and checkpoint
   selection rule. [PENDING]

## 2. Related work

YOLO frames detection as a single regression problem over a grid (Redmon et al.,
2016). YOLOv8 is a later family of anchor-free detectors maintained by Ultralytics
(Jocher et al., 2023). For box regression, IoU-based losses address a known
weakness of coordinate losses: IoU is scale invariant, and GIoU keeps a gradient
when two boxes do not overlap (Rezatofighi et al., 2019). We use GIoU as one of
the box losses in our ablation.

[PENDING: add related work on jersey-number recognition and sports player
detection, after a literature check.]

## 3. Data

We use the Roboflow export `jersey-2` (workspace `muzammal-masmatixinnovations-co-uk`,
project `jersey-qyuun-s5ab4`, version 1, licence Public Domain). It has one class,
`number`, in YOLO format.

| Split | Images | Labelled images (scored) |
|---|---|---|
| Train | 3,165 | [PENDING] |
| Validation | 301 | 263 |
| Test | 148 | 130 |

Images without a label file, or with an empty label file, are not scored. Each
image is resized to 128 × 128 for the CNN. YOLOv8n uses its own input size of 640.

Every labelled image in this export is positive. Presence accuracy is therefore
always 1.0 and is not reported. The task we measure is localisation.

## 4. Methods

### 4.1 Scratch CNN

A small convolutional network with a box head (four sigmoid outputs) and a
presence head. Training uses Adam with learning rate 1e-3 and batch size 8. The
box loss is MSE by default, and BCE is used for presence. The checkpoint is the
epoch with the lowest validation loss. The default run is 15 epochs.

### 4.2 Fine-tuned YOLOv8n

Ultralytics YOLOv8n initialised from COCO-pretrained weights and fine-tuned for
50 epochs at image size 640, batch 16, seed 0. The checkpoint is chosen by
Ultralytics' validation fitness.

### 4.3 Mean-box baseline

The average of all training boxes is used as the prediction for every image. It
has no image input and sets a floor for localisation quality on this split.

## 5. Evaluation protocol

For each labelled image, the first annotated box is the ground truth, following
the original notebook. For each model, the prediction is the highest-confidence
box (YOLO) or the single predicted box (CNN, baseline). If a model predicts no
box, the image scores IoU 0. Mean IoU is the mean over scored images. We also
report the share of images with IoU ≥ 0.5.

The test split is the reported result. Validation was used to choose checkpoints,
so its numbers are optimistic and are given for reference only.

## 6. Results

Test split (130 scored images):

| Model | Mean IoU | Share with IoU ≥ 0.5 |
|---|---|---|
| Mean-box baseline | 0.016 | [PENDING] |
| Scratch CNN (15 epochs) | 0.044 | [PENDING] |
| YOLOv8n (fine-tuned, 50 epochs) | 0.581 | 68.5% |

Validation split (263 scored images), for reference:

| Model | Mean IoU | Share with IoU ≥ 0.5 |
|---|---|---|
| Mean-box baseline | 0.027 | [PENDING] |
| Scratch CNN (15 epochs) | 0.045 | [PENDING] |
| YOLOv8n (fine-tuned, 50 epochs) | 0.593 | 70.3% |

The fine-tuned detector is far ahead of both other models. The scratch CNN
improves on the mean-box baseline by a small margin. Section 7 examines why.

## 7. Ablation of the scratch CNN [PENDING]

Four variants, all on the same split, reported on the test split:

- A. Default: MSE box loss, 15 epochs, checkpoint by validation loss.
- B. Longer training: 50 epochs, otherwise as A.
- C. GIoU box loss, otherwise as A.
- D. GIoU box loss, checkpoint by validation IoU, 50 epochs.

[PENDING: fill in after the runs.]

## 8. Limitations

- One class, one box per image. Other players and other numbers are ignored.
- The presence metric is trivial on this export, so the paper reports localisation only.
- The test split has 130 scored images. Differences of a few points are within the
  noise of a split this size. [PENDING: seeds and confidence intervals]
- Checkpoints were selected on the validation split, which the test numbers do not
  use, but the validation numbers are slightly optimistic.
- The CNN and YOLO models use different input sizes (128 and 640) and different
  pretraining, so the comparison is between systems, not architectures.

## 9. Reproducibility

Code: `https://github.com/MHKhan50/jersey-number-detection` (branch `box-loss-giou`).

```bash
jersey-detect train --config configs/default.json \
  --train-record /data/jersey-2/train --val-record /data/jersey-2/valid \
  --out-dir runs/cnn
jersey-detect evaluate --checkpoint runs/cnn/best.pt --data /data/jersey-2/test
```

The YOLO run uses Ultralytics with `yolov8n.pt`, 50 epochs, `imgsz=640`, `batch=16`,
`seed=0`. The mean-box baseline and the scoring rule are in `mean_iou_eval.py`.

## 10. AI-use disclosure

[PENDING: check the target venue's policy before submission.]

## References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (version 8). https://github.com/ultralytics/ultralytics
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *CVPR*.
- Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). Generalized intersection over union. *CVPR*.
