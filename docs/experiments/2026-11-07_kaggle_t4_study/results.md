# Data efficiency and an augmentation-matched YOLOv8n

Test split (147 images, 234 boxes), mean ± standard deviation over the seeds that finished.
AP by size uses the number side at 128 px: small < 8, medium 8-16, large > 16.
Subset runs are nested (10% inside 25%) and train for the same number of optimiser steps as the
full-data runs (YOLOv8n: 1,400 and 560 epochs from scratch, 500 and 200 from COCO; JerseyNet 1,400 and 560
epochs). The augmentation-matched YOLOv8n uses no mosaic, mixup or flip, scale ±35%,
translation 10% and saturation/value jitter ±30%, to approximate JerseyNet's augmentation.

Run: Kaggle, GPU T4 x2, `notebooks/kaggle_study.ipynb`, 11.0 hours; no errors. Four seed-2 runs were
skipped by the time budget (listed below).

| Model | Train fraction | Seeds | AP50:95 | AP50 | AP small | AP medium | AP large |
|---|---|---|---|---|---|---|---|
| YOLOv8n, from scratch, 256 px, JerseyNet augmentation | 1 | [0, 1, 2] | 0.280 ± 0.012 | 0.540 ± 0.023 | 0.071 ± 0.011 | 0.325 ± 0.035 | 0.543 ± 0.019 |
| JerseyNet, from scratch, 256 px | 0.1 | [0, 1, 2] | 0.124 ± 0.011 | 0.300 ± 0.030 | 0.031 ± 0.007 | 0.159 ± 0.021 | 0.258 ± 0.009 |
| YOLOv8n, from scratch, 256 px | 0.1 | [0, 1, 2] | 0.265 ± 0.013 | 0.512 ± 0.017 | 0.065 ± 0.011 | 0.336 ± 0.026 | 0.499 ± 0.022 |
| YOLOv8n, COCO-pretrained, 256 px | 0.1 | [0, 1] | 0.302 ± 0.011 | 0.572 ± 0.010 | 0.081 ± 0.005 | 0.388 ± 0.011 | 0.553 ± 0.016 |
| JerseyNet, from scratch, 256 px | 0.25 | [0, 1] | 0.183 ± 0.001 | 0.403 ± 0.028 | 0.040 ± 0.010 | 0.252 ± 0.015 | 0.380 ± 0.005 |
| YOLOv8n, from scratch, 256 px | 0.25 | [0, 1] | 0.305 ± 0.012 | 0.556 ± 0.018 | 0.073 ± 0.005 | 0.375 ± 0.016 | 0.581 ± 0.013 |
| YOLOv8n, COCO-pretrained, 256 px | 0.25 | [0, 1] | 0.349 ± 0.006 | 0.624 ± 0.001 | 0.100 ± 0.015 | 0.429 ± 0.026 | 0.635 ± 0.005 |

## Runs

| Model | Train fraction | Seed | Train images | Val AP50:95 | Test AP50:95 [95% CI] | Minutes |
|---|---|---|---|---|---|---|
| YOLOv8n, from scratch, 256 px, JerseyNet augmentation | 1 | 0 | 3165 | 0.316 | 0.265 [0.225, 0.322] | 38 |
| JerseyNet, from scratch, 256 px | 0.1 | 0 | 317 | 0.121 | 0.111 [0.084, 0.150] | 40 |
| YOLOv8n, from scratch, 256 px | 0.1 | 0 | 317 | 0.278 | 0.250 [0.205, 0.309] | 70 |
| YOLOv8n, COCO-pretrained, 256 px | 0.1 | 0 | 317 | 0.334 | 0.310 [0.262, 0.365] | 24 |
| JerseyNet, from scratch, 256 px | 0.25 | 0 | 792 | 0.208 | 0.182 [0.147, 0.234] | 32 |
| YOLOv8n, from scratch, 256 px | 0.25 | 0 | 792 | 0.332 | 0.296 [0.250, 0.353] | 47 |
| YOLOv8n, COCO-pretrained, 256 px | 0.25 | 0 | 792 | 0.371 | 0.353 [0.307, 0.413] | 17 |
| YOLOv8n, from scratch, 256 px, JerseyNet augmentation | 1 | 1 | 3165 | 0.319 | 0.286 [0.241, 0.343] | 36 |
| JerseyNet, from scratch, 256 px | 0.1 | 1 | 317 | 0.134 | 0.129 [0.096, 0.173] | 42 |
| YOLOv8n, from scratch, 256 px | 0.1 | 1 | 317 | 0.276 | 0.274 [0.231, 0.330] | 64 |
| YOLOv8n, COCO-pretrained, 256 px | 0.1 | 1 | 317 | 0.342 | 0.294 [0.249, 0.351] | 22 |
| JerseyNet, from scratch, 256 px | 0.25 | 1 | 792 | 0.209 | 0.183 [0.148, 0.229] | 31 |
| YOLOv8n, from scratch, 256 px | 0.25 | 1 | 792 | 0.331 | 0.314 [0.268, 0.370] | 44 |
| YOLOv8n, COCO-pretrained, 256 px | 0.25 | 1 | 792 | 0.366 | 0.344 [0.294, 0.403] | 16 |
| YOLOv8n, from scratch, 256 px, JerseyNet augmentation | 1 | 2 | 3165 | 0.296 | 0.288 [0.242, 0.348] | 35 |
| JerseyNet, from scratch, 256 px | 0.1 | 2 | 317 | 0.140 | 0.133 [0.101, 0.173] | 31 |
| YOLOv8n, from scratch, 256 px | 0.1 | 2 | 317 | 0.283 | 0.270 [0.225, 0.325] | 65 |

## Skipped by the time budget

- YOLOv8n, COCO-pretrained, 256 px | 0.1 | seed 2
- JerseyNet, from scratch, 256 px | 0.25 | seed 2
- YOLOv8n, from scratch, 256 px | 0.25 | seed 2
- YOLOv8n, COCO-pretrained, 256 px | 0.25 | seed 2
