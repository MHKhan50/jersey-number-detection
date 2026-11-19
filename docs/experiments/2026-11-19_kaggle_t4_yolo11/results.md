# YOLO11n baselines

Same evaluator as JerseyNet. Test intervals are 95% bootstrap intervals over images.

| Model | Split | AP50:95 | AP50 | AP75 | AR@100 | First-box mean IoU | First-box IoU ≥ 0.5 |
|---|---|---|---|---|---|---|---|
| YOLO11n, COCO-pretrained, 640 px (seed 0) | val | 0.435 | 0.722 | 0.472 | 0.558 | 0.574 | 69.2% |
| YOLO11n, COCO-pretrained, 640 px (seed 0) | test | 0.400 [0.348, 0.463] | 0.673 [0.606, 0.749] | 0.442 | 0.550 | 0.563 [0.497, 0.630] | 66.9% |
| YOLO11n, from scratch, 384 px (seed 0) | val | 0.400 | 0.684 | 0.434 | 0.517 | 0.554 | 67.3% |
| YOLO11n, from scratch, 384 px (seed 0) | test | 0.357 [0.309, 0.414] | 0.628 [0.554, 0.713] | 0.378 | 0.489 | 0.557 [0.494, 0.623] | 66.9% |
| YOLO11n, COCO-pretrained, 640 px (seed 1) | val | 0.428 | 0.722 | 0.458 | 0.548 | 0.568 | 68.8% |
| YOLO11n, COCO-pretrained, 640 px (seed 1) | test | 0.422 [0.371, 0.488] | 0.701 [0.630, 0.780] | 0.453 | 0.541 | 0.598 [0.530, 0.661] | 72.3% |
| YOLO11n, from scratch, 384 px (seed 1) | val | 0.395 | 0.687 | 0.417 | 0.509 | 0.572 | 69.6% |
| YOLO11n, from scratch, 384 px (seed 1) | test | 0.394 [0.349, 0.450] | 0.667 [0.601, 0.741] | 0.429 | 0.514 | 0.569 [0.506, 0.636] | 69.2% |
| YOLO11n, COCO-pretrained, 640 px (seed 2) | val | 0.436 | 0.736 | 0.471 | 0.551 | 0.599 | 72.2% |
| YOLO11n, COCO-pretrained, 640 px (seed 2) | test | 0.418 [0.372, 0.476] | 0.713 [0.653, 0.785] | 0.464 | 0.558 | 0.597 [0.530, 0.654] | 72.3% |
| YOLO11n, from scratch, 384 px (seed 2) | val | 0.406 | 0.689 | 0.438 | 0.523 | 0.565 | 68.8% |
| YOLO11n, from scratch, 384 px (seed 2) | test | 0.389 [0.340, 0.446] | 0.659 [0.586, 0.736] | 0.418 | 0.514 | 0.571 [0.509, 0.636] | 68.5% |

## Over seeds (test)

| Setting | Seeds | AP50:95 | AP50 | First-box mean IoU |
|---|---|---|---|---|
| YOLO11n, COCO-pretrained, 640 px | [0, 1, 2] | 0.413 ± 0.012 | 0.696 ± 0.021 | 0.586 ± 0.020 |
| YOLO11n, from scratch, 384 px | [0, 1, 2] | 0.380 ± 0.020 | 0.651 ± 0.021 | 0.566 ± 0.008 |

