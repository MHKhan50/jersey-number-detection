# YOLOv8n baselines

Same evaluator as JerseyNet. Test intervals are 95% bootstrap intervals over images.

| Model | Split | AP50:95 | AP50 | AP75 | AR@100 | First-box mean IoU | First-box IoU ≥ 0.5 |
|---|---|---|---|---|---|---|---|
| YOLOv8n, COCO-pretrained, 640 px (seed 0) | val | 0.444 | 0.737 | 0.497 | 0.554 | 0.593 | 70.3% |
| YOLOv8n, COCO-pretrained, 640 px (seed 0) | test | 0.430 [0.377, 0.496] | 0.706 [0.632, 0.782] | 0.513 | 0.543 | 0.581 [0.517, 0.642] | 68.5% |
| YOLOv8n, COCO-pretrained, 256 px (seed 0) | val | 0.400 | 0.686 | 0.426 | 0.510 | 0.576 | 70.7% |
| YOLOv8n, COCO-pretrained, 256 px (seed 0) | test | 0.381 [0.330, 0.442] | 0.679 [0.608, 0.765] | 0.384 | 0.494 | 0.561 [0.490, 0.622] | 69.2% |
| YOLOv8n, from scratch, 256 px (seed 0) | val | 0.374 | 0.656 | 0.400 | 0.479 | 0.551 | 66.9% |
| YOLOv8n, from scratch, 256 px (seed 0) | test | 0.343 [0.296, 0.400] | 0.624 [0.551, 0.701] | 0.375 | 0.457 | 0.525 [0.459, 0.585] | 64.6% |
| YOLOv8n, COCO-pretrained, 640 px (seed 1) | val | 0.438 | 0.727 | 0.479 | 0.542 | 0.594 | 71.5% |
| YOLOv8n, COCO-pretrained, 640 px (seed 1) | test | 0.407 [0.360, 0.466] | 0.704 [0.637, 0.773] | 0.446 | 0.525 | 0.593 [0.531, 0.655] | 71.5% |
| YOLOv8n, COCO-pretrained, 256 px (seed 1) | val | 0.400 | 0.683 | 0.425 | 0.508 | 0.580 | 69.6% |
| YOLOv8n, COCO-pretrained, 256 px (seed 1) | test | 0.384 [0.335, 0.449] | 0.672 [0.597, 0.753] | 0.384 | 0.488 | 0.593 [0.528, 0.654] | 73.1% |
| YOLOv8n, from scratch, 256 px (seed 1) | val | 0.381 | 0.661 | 0.403 | 0.496 | 0.557 | 67.7% |
| YOLOv8n, from scratch, 256 px (seed 1) | test | 0.346 [0.298, 0.406] | 0.613 [0.544, 0.693] | 0.366 | 0.468 | 0.548 [0.484, 0.611] | 66.9% |
| YOLOv8n, COCO-pretrained, 640 px (seed 2) | val | 0.440 | 0.734 | 0.474 | 0.555 | 0.593 | 71.1% |
| YOLOv8n, COCO-pretrained, 640 px (seed 2) | test | 0.418 [0.368, 0.477] | 0.692 [0.623, 0.761] | 0.469 | 0.547 | 0.595 [0.529, 0.659] | 70.0% |
| YOLOv8n, COCO-pretrained, 256 px (seed 2) | val | 0.404 | 0.683 | 0.423 | 0.500 | 0.581 | 70.3% |
| YOLOv8n, COCO-pretrained, 256 px (seed 2) | test | 0.386 [0.337, 0.447] | 0.675 [0.601, 0.760] | 0.414 | 0.485 | 0.574 [0.510, 0.640] | 70.8% |
| YOLOv8n, from scratch, 256 px (seed 2) | val | 0.374 | 0.653 | 0.407 | 0.479 | 0.534 | 65.0% |
| YOLOv8n, from scratch, 256 px (seed 2) | test | 0.367 [0.318, 0.429] | 0.650 [0.580, 0.732] | 0.374 | 0.483 | 0.558 [0.491, 0.626] | 68.5% |

## Over seeds (test)

| Setting | Seeds | AP50:95 | AP50 | First-box mean IoU |
|---|---|---|---|---|
| YOLOv8n, COCO-pretrained, 640 px | [0, 1, 2] | 0.419 ± 0.012 | 0.701 ± 0.008 | 0.590 ± 0.007 |
| YOLOv8n, COCO-pretrained, 256 px | [0, 1, 2] | 0.384 ± 0.003 | 0.675 ± 0.004 | 0.576 ± 0.016 |
| YOLOv8n, from scratch, 256 px | [0, 1, 2] | 0.352 ± 0.013 | 0.629 ± 0.019 | 0.544 ± 0.017 |
