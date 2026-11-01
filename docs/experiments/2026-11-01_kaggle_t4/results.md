# Results

Test intervals are 95% bootstrap intervals over images.

| Model | Split | AP50:95 | AP50 | AP75 | AR@100 | First-box mean IoU | First-box IoU ≥ 0.5 |
|---|---|---|---|---|---|---|---|
| Mean-box baseline | val | 0.000 | 0.000 | 0.000 | 0.001 | 0.027 | 0.4% |
| Mean-box baseline | test | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.016 [0.008, 0.025] | 0.0% |
| v1 CNN regressor (15 epochs) | val | 0.000 | 0.000 | 0.000 | 0.002 | 0.049 | 1.5% |
| v1 CNN regressor (15 epochs) | test | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.002] | 0.000 | 0.001 | 0.051 [0.033, 0.069] | 0.0% |
| JerseyNet, from scratch (seed 0) | val | 0.247 | 0.505 | 0.199 | 0.362 | 0.461 | 58.2% |
| JerseyNet, from scratch (seed 0) | test | 0.234 [0.195, 0.288] | 0.491 [0.417, 0.580] | 0.219 | 0.360 | 0.454 [0.390, 0.522] | 56.2% |
| JerseyNet, from scratch (seed 1) | val | 0.262 | 0.537 | 0.211 | 0.384 | 0.471 | 60.1% |
| JerseyNet, from scratch (seed 1) | test | 0.246 [0.203, 0.302] | 0.496 [0.418, 0.583] | 0.235 | 0.382 | 0.473 [0.407, 0.539] | 59.2% |
| JerseyNet, from scratch (seed 2) | val | 0.262 | 0.537 | 0.221 | 0.382 | 0.473 | 60.5% |
| JerseyNet, from scratch (seed 2) | test | 0.242 [0.201, 0.298] | 0.498 [0.424, 0.586] | 0.227 | 0.371 | 0.500 [0.438, 0.566] | 63.1% |

## JerseyNet over seeds (test)

Seeds: [0, 1, 2]

| Metric | Mean | Std |
|---|---|---|
| ap50_95 | 0.241 | 0.006 |
| ap50 | 0.495 | 0.004 |
| first_box_mean_iou | 0.476 | 0.023 |

