# Test AP by number size

Number side in pixels of the 128 px images: small < 8, medium 8-16, large > 16.
Test boxes: small 94, medium 84, large 56. Mean ± standard deviation over seeds; single-seed rows have no spread.
Nothing was retrained: the saved checkpoints of the earlier runs were re-scored
(`notebooks/kaggle_rescore.ipynb`). Overall AP50:95 matches the earlier results for every model.

| Model | Seeds | AP50:95 | AP small | AP medium | AP large |
|---|---|---|---|---|---|
| JerseyNet, 256 px | [0, 1, 2] | 0.241 ± 0.006 | 0.043 ± 0.011 | 0.320 ± 0.018 | 0.463 ± 0.009 |
| JerseyNet, 256 px (ablation run) | [0] | 0.234 | 0.034 | 0.307 | 0.467 |
| JerseyNet, 256 px, GIoU (ablation run) | [0] | 0.255 | 0.053 | 0.302 | 0.533 |
| JerseyNet, 256 px, mosaic (ablation run) | [0] | 0.273 | 0.061 | 0.352 | 0.529 |
| JerseyNet, 256 px, mosaic, GIoU (ablation run) | [0] | 0.258 | 0.050 | 0.321 | 0.527 |
| JerseyNet, 384 px (ablation run) | [0] | 0.277 | 0.085 | 0.352 | 0.515 |
| JerseyNet, 384 px, mosaic, GIoU (ablation run) | [0, 1, 2] | 0.308 ± 0.005 | 0.101 ± 0.012 | 0.408 ± 0.011 | 0.529 ± 0.005 |
| yolov8n_coco_pretrained_256_px | [0, 1, 2] | 0.384 ± 0.003 | 0.101 ± 0.012 | 0.485 ± 0.008 | 0.678 ± 0.008 |
| yolov8n_coco_pretrained_640_px | [0, 1, 2] | 0.419 ± 0.012 | 0.123 ± 0.009 | 0.542 ± 0.026 | 0.670 ± 0.021 |
| yolov8n_from_scratch_256_px | [0, 1, 2] | 0.352 ± 0.013 | 0.087 ± 0.019 | 0.453 ± 0.015 | 0.609 ± 0.018 |
| yolov8n_from_scratch_384_px | [0, 1, 2] | 0.366 ± 0.005 | 0.103 ± 0.007 | 0.491 ± 0.019 | 0.590 ± 0.019 |
