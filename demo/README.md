# Browser demo

Source of the live demo at [huggingface.co/spaces/MHkhan50/jersey-number-detection](https://huggingface.co/spaces/MHkhan50/jersey-number-detection).

JerseyNet (mosaic + GIoU, 384 px, seed 2 of the ablation) runs in the browser with ONNX Runtime Web, so images are never uploaded.

| File | What it is |
|---|---|
| `index.html`, `app.js` | The app |
| `core.js` | Letterbox (Pillow-style bilinear resize) and heatmap decoding, the same as `transforms.letterbox` and `detector.decode` |
| `export_onnx.py` | Checkpoint to ONNX, with a check against a NumPy forward pass |

Checks: the ONNX model matches the NumPy forward pass to within 1e-5. Scored with the repository's evaluator, the exported pipeline gives test AP50:95 0.314 (AP50 0.593), the same as the checkpoint in PyTorch. `core.js` gives the same input tensor as the Python letterbox on native 128 px images, and the same boxes and scores from the same network outputs.

The ONNX file (18 MB) is stored in the Space, not here. To run the demo locally, put `jerseynet.onnx` in this folder and serve it with `python3 -m http.server`.

Research demo only: trained on 128 × 128 px ice-hockey images; it localises numbers but does not read them.
