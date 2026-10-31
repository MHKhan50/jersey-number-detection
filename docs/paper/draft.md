# Localising jersey numbers without pretraining: a from-scratch anchor-free detector versus a fine-tuned YOLOv8n

Muzammal Hussain

> Draft. Sections marked [PENDING] depend on runs that are not finished yet.

## Abstract

We study how far a detector trained from random initialisation can go in localising jersey
numbers in sports images, and how it compares with a COCO-pretrained YOLOv8n fine-tuned on the
same data. Our first attempt, a small network that regresses one box from globally pooled
features, reached a test mean IoU of 0.044 against 0.016 for a constant mean-box prediction and
0.581 for the fine-tuned YOLOv8n. We replace it with JerseyNet, a CenterNet-style detector with a
residual backbone and a feature pyramid fused to stride 4, trained from scratch. All models are
scored with COCO-style AP over every annotated box, plus the earlier first-box IoU, on the same
public split, with bootstrap intervals and three seeds. [PENDING: JerseyNet results]

## 1. Introduction

Jersey-number recognition is usually framed as reading digits from a player crop. Many practical
systems first need to know where the number is. This paper treats that stage on its own: given an
image, predict a box and a confidence for every visible jersey number.

We ask two questions. How much of the localisation performance of a pretrained detector can a
network trained only on a few thousand task images recover? And how much of the gap comes from
the model design rather than from pretraining?

Contributions:

1. A reproducible protocol for jersey-number localisation on a public YOLO-format export:
   COCO-style AP over all boxes, negatives included, test intervals by bootstrap, model
   selection on validation only, and a cross-split duplicate check.
2. JerseyNet, a small anchor-free detector trained from scratch, compared with a mean-box
   baseline, our earlier single-box regressor, and a fine-tuned YOLOv8n.
3. An analysis of why the single-box regressor fails. [PENDING]

## 2. Related work

**Detection.** YOLO frames detection as a single regression problem over a grid (Redmon et al.,
2016); YOLOv8 is a later anchor-free family maintained by Ultralytics (Jocher et al., 2023).
CenterNet represents each object by its centre point on a heatmap and regresses its size at that
point (Zhou et al., 2019), with the penalty-reduced focal loss of CornerNet (Law and Deng, 2018).
It needs neither anchors nor non-maximum suppression. He et al. (2019) showed that detectors can
be trained from random initialisation when normalisation is used and training is long enough,
which motivates our from-scratch setting. IoU-based box losses such as GIoU (Rezatofighi et al.,
2019) address the scale dependence of coordinate losses. We use COCO-style average precision
(Lin et al., 2014) as the main metric.

**Jersey numbers.** Most prior work treats the problem as digit reading and assumes the player
has already been localised. Vats et al. (2021) trained a ResNet-34 with whole-number and
per-digit losses on 54,251 player images from NHL games, reporting 89.6% test accuracy. The
SoccerNet 2023 challenge introduced a public tracklet-level benchmark (Cioppa et al., 2024).
Balaji et al. (2023) added keyframe selection in front of a ResNet-18 and bidirectional LSTM,
reaching 68.53% on the SoccerNet test set. Koshkina and Elder (2024) combined legibility
filtering, pose-based torso localisation and the PARSeq scene-text recogniser, and aggregated
frame predictions to tracklet labels, reporting 87.4% tracklet accuracy on SoccerNet. Grad (2025)
proposed a single-stage, uncertainty-aware recogniser on player crops. Liu et al. (2022) used a
detection transformer to find players in American football video and a separate CNN to read their
numbers. Our work isolates the localisation of the number region inside the image, which these
pipelines either assume or obtain from pose estimation.

## 3. Data

We use the Roboflow export `jersey-2` (workspace `muzammal-masmatixinnovations-co-uk`, project
`jersey-qyuun-s5ab4`, version 1, licence Public Domain), one class (`number`), YOLO format.

| Split | Images | Labelled images | Boxes | Max boxes per image |
|---|---|---|---|---|
| Train | 3,165 | 2,784 | 4,941 | 9 |
| Validation | 301 | 263 | 485 | 6 |
| Test | 147 | 130 | 234 | 9 |

All images are 128 × 128 pixels. The median box side is about 8% of the image side (about 10
pixels), and the 10th percentile is about 3% (4 pixels).

Images without a label, or with an empty label file, are kept and act as negatives.
Cross-split duplicates: no Roboflow source name is shared between splits; with a 64-bit
difference hash and a Hamming distance of at most 4, 9 validation images and 1 test image have a
near-identical training image. The official splits are used unchanged.

## 4. Methods

### 4.1 Mean-box baseline

The mean of the first annotated box over labelled training images, predicted on every image
with confidence 1. It has no image input.

### 4.2 v1 CNN regressor

Three 3×3 convolutions (16, 32, 64 channels) with max pooling, global average pooling, a 128-unit
layer and four sigmoid outputs. Input 128×128 (stretched), Adam with learning rate 1e-3, batch 8,
MSE on the first box, 15 epochs, weights after the last epoch.

### 4.3 JerseyNet

* Backbone: a stem of two 3×3 convolutions (stride 2), then four residual stages of two basic
  blocks each, at strides 4, 8, 16 and 32 with 64, 128, 192 and 256 channels, all with batch norm.
* Neck: 1×1 lateral projections to 96 channels and a top-down path with nearest-neighbour
  upsampling and 3×3 smoothing, ending at stride 4.
* Heads at stride 4: a centre heatmap, box width and height, and a sub-pixel offset.
* Loss: penalty-reduced focal loss on the heatmap (α = 2, β = 4) with Gaussian targets whose
  radius follows CenterNet, plus L1 on size (weight 0.1) and offset (weight 1) at object centres.
* Decoding: 3×3 max-pool peak selection, the top 100 peaks, no NMS.
* Input and augmentation: 256×256 letterbox (the 128×128 images upsampled 2×, so the smallest
  numbers still span two cells of the stride-4 output), random scale in [0.6, 1.3], random placement,
  brightness, contrast and saturation jitter (±30%). No horizontal flip, which would mirror the
  digits.
* Optimisation: AdamW, learning rate 1e-3, weight decay 1e-4 (not on norms and biases), three
  warm-up epochs then cosine decay to 1% over 140 epochs, batch 64, gradient clipping at 10,
  mixed precision. Validation every five epochs; the checkpoint with the best validation
  AP50:95 is kept. About 4.7 million parameters, all randomly initialised.

### 4.4 Fine-tuned YOLOv8n

Ultralytics YOLOv8n from COCO-pretrained weights, fine-tuned for 50 epochs at image size 640,
batch 16, seed 0. [PENDING: re-score its saved test predictions with our evaluator
(`jersey-detect score`) so that all rows use the same protocol.]

## 5. Evaluation protocol

* **COCO-style AP.** AP50:95 (mean over IoU thresholds 0.50 to 0.95), AP50, AP75 and AR with at
  most 100 detections per image, over every annotated box. Detections on negative images are
  false positives. Our NumPy implementation is tested against `pycocotools`.
* **First-box IoU (v1).** On each labelled image, the highest-scoring box against the first
  annotated box; mean IoU and the share with IoU ≥ 0.5. An image with no prediction scores 0.
* **Selection and reporting.** Checkpoints are chosen on validation AP50:95. The test split is
  scored once per trained model. Test metrics carry 95% percentile bootstrap intervals over
  images (1,000 resamples). JerseyNet is trained with seeds 0, 1 and 2, and we report mean and
  standard deviation.

## 6. Results

v1 protocol, test split (130 scored images), from the earlier runs:

| Model | Mean IoU | Share with IoU ≥ 0.5 |
|---|---|---|
| Mean-box baseline | 0.016 | [PENDING] |
| v1 CNN regressor (15 epochs) | 0.044 | [PENDING] |
| YOLOv8n (fine-tuned, 50 epochs) | 0.581 | 68.5% |

Full protocol, test split: [PENDING: table from `results/results.md`]

## 7. Analysis of the v1 regressor

Ablation of the v1 regressor, test split, run before the box-ordering fix:

| Variant | Box loss | Epochs | Checkpoint | Box MSE | Mean IoU |
|---|---|---|---|---|---|
| A. Default | MSE | 15 | validation loss | 0.031 | 0.044 |
| B. Longer training | MSE | 50 | validation loss | 0.034 | 0.039 |
| C. GIoU box loss | GIoU | 15 | validation loss | 0.170 | 0.000 |
| D. GIoU, IoU selection | GIoU | 50 | validation IoU | 0.280 | 0.000 |

Longer training does not help, and neither loss moves the regressor beyond the baseline range.
The design is the likely limit: global average pooling discards where features occur, the model
can output only one box when images can contain several numbers, and the training images hold up to nine numbers each,
leaves small numbers a few pixels wide. JerseyNet addresses each of these (spatial heatmap, many
outputs, stride-4 features on a 2× upsampled input). The GIoU collapse is reported as a negative
result with an open diagnosis; the v1 code is preserved in the version 1.0.0 archive.
[PENDING: confirm with the v2 results.]

## 8. Limitations

* One dataset with 147 test images; intervals are wide.
* Frames from one match may sit in different splits; we measure but do not remove this.
* Localisation only; the digits are not read.
* JerseyNet and YOLOv8n differ in input size and pretraining, so that comparison is between
  systems, not architectures.
* GPU training is seeded but not bit-for-bit deterministic.

## 9. Reproducibility

Code: `https://github.com/MHKhan50/jersey-number-detection` (version 2.0.0).

```bash
jersey-detect run --data /data/jersey-2 --out results --config configs/kaggle.json
```

The notebook `notebooks/kaggle_train.ipynb` runs the same command on a Kaggle T4 GPU and records
the environment and the exact commit in `results/environment.json`.

## 10. AI-use disclosure

[PENDING: check the target venue's policy before submission.]

## References

- Balaji, B., Bright, J., Prakash, H., Chen, Y., Clausi, D. A., & Zelek, J. (2023). Jersey number recognition using keyframe identification from low-resolution broadcast videos. *MMSports '23*. arXiv:2309.06285.
- Cioppa, A., Giancola, S., Somers, V., et al. (2024). SoccerNet 2023 challenges results. *Sports Engineering*, 27, 24.
- Grad, Ł. (2025). Single-stage uncertainty-aware jersey number recognition in soccer. *CVPR Workshops (CVsports)*.
- He, K., Girshick, R., & Dollár, P. (2019). Rethinking ImageNet pre-training. *ICCV*.
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (version 8). https://github.com/ultralytics/ultralytics
- Koshkina, M., & Elder, J. H. (2024). A general framework for jersey number recognition in sports video. *CVPR Workshops*, 3235–3244.
- Law, H., & Deng, J. (2018). CornerNet: Detecting objects as paired keypoints. *ECCV*.
- Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). Microsoft COCO: Common objects in context. *ECCV*.
- Liu, H., Aderon, C., Wagon, N., Bamba, A. L., Li, X., Liu, H., MacCall, S., & Gan, Y. (2022). Automated player identification and indexing using two-stage deep learning network. arXiv:2204.13809.
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *CVPR*.
- Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). Generalized intersection over union. *CVPR*.
- Vats, K., Fani, M., Clausi, D. A., & Zelek, J. (2021). Multi-task learning for jersey number recognition in ice hockey. *MMSports '21*.
- Zhou, X., Wang, D., & Krähenbühl, P. (2019). Objects as points. arXiv:1904.07850.
