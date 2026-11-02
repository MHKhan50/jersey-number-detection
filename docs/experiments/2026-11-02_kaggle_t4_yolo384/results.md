# YOLOv8n from scratch at 384 px

Same evaluator as JerseyNet. Test intervals are 95% bootstrap intervals over images.

| Model | Split | AP50:95 | AP50 | AP75 | AR@100 | First-box mean IoU | First-box IoU ≥ 0.5 |
|---|---|---|---|---|---|---|---|
| YOLOv8n, from scratch, 384 px (seed 0) | val | 0.389 | 0.669 | 0.427 | 0.493 | 0.569 | 68.8% |
| YOLOv8n, from scratch, 384 px (seed 0) | test | 0.371 [0.326, 0.430] | 0.650 [0.580, 0.732] | 0.372 | 0.483 | 0.572 [0.510, 0.635] | 70.0% |
| YOLOv8n, from scratch, 384 px (seed 1) | val | 0.393 | 0.679 | 0.427 | 0.524 | 0.544 | 66.2% |
| YOLOv8n, from scratch, 384 px (seed 1) | test | 0.362 [0.312, 0.422] | 0.646 [0.582, 0.718] | 0.368 | 0.487 | 0.540 [0.474, 0.603] | 66.2% |
| YOLOv8n, from scratch, 384 px (seed 2) | val | 0.394 | 0.681 | 0.425 | 0.508 | 0.554 | 67.7% |
| YOLOv8n, from scratch, 384 px (seed 2) | test | 0.366 [0.320, 0.424] | 0.651 [0.588, 0.727] | 0.374 | 0.494 | 0.550 [0.486, 0.616] | 66.9% |

Over seeds (test): AP50:95 0.366 ± 0.005, AP50 0.649 ± 0.003, first-box mean IoU 0.554 ± 0.016.

Run: Kaggle, GPU T4 x2, notebook `notebooks/kaggle_yolo_baseline.ipynb` with `ONLY = ["yolov8n_from_scratch_384_px"]`,
about 33 minutes per seed, 1.7 hours in total.
