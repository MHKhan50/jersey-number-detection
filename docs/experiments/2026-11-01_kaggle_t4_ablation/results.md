# JerseyNet ablation

Stage 1: every variant with seed 0. Stage 2: the variant with the best validation AP50:95 with seeds 1 and 2. Test intervals are 95% bootstrap intervals over images.

| Variant | Seed | Val AP50:95 | Test AP50:95 | Test AP50 | Test AP75 | Test first-box IoU | Minutes |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 0.259 | 0.235 [0.194, 0.290] | 0.474 | 0.205 | 0.463 | 25 |
| mosaic | 0 | 0.288 | 0.273 [0.228, 0.331] | 0.539 | 0.248 | 0.504 | 32 |
| giou | 0 | 0.256 | 0.255 [0.212, 0.312] | 0.502 | 0.247 | 0.498 | 25 |
| 384px | 0 | 0.304 | 0.277 [0.234, 0.331] | 0.543 | 0.253 | 0.444 | 52 |
| mosaic+giou | 0 | 0.293 | 0.258 [0.212, 0.317] | 0.510 | 0.255 | 0.496 | 37 |
| mosaic+giou+384px | 0 | 0.337 | 0.303 [0.260, 0.358] | 0.595 | 0.272 | 0.522 | 65 |
| mosaic+giou+384px | 1 | 0.335 | 0.308 [0.260, 0.367] | 0.604 | 0.285 | 0.515 | 65 |
| mosaic+giou+384px | 2 | 0.333 | 0.314 [0.270, 0.371] | 0.593 | 0.326 | 0.523 | 65 |

Best variant on validation (seed 0): **mosaic+giou+384px**

## mosaic+giou+384px over seeds [0, 1, 2] (test)

| Metric | Mean | Std |
|---|---|---|
| ap50_95 | 0.308 | 0.005 |
| ap50 | 0.597 | 0.006 |
| ap75 | 0.294 | 0.028 |
| first_box_mean_iou | 0.520 | 0.004 |

Run: Kaggle, GPU T4 x2 (one GPU used), notebook `notebooks/kaggle_ablation.ipynb`, 6 h 14 min in total.
