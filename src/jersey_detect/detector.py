"""JerseyNet: an anchor-free, CenterNet-style detector trained from random initialisation.

Design (Zhou et al., 2019, "Objects as Points"):

* Backbone: residual stages with batch norm at strides 4, 8, 16 and 32. No pretrained weights;
  batch norm is what makes from-scratch detection training workable (He et al., 2019).
* Neck: a top-down feature pyramid that fuses all stages back to stride 4, because jersey
  numbers are small and need the high-resolution map.
* Heads at stride 4: a centre heatmap (focal loss), box width/height and sub-pixel offset (L1).

Decoding keeps local maxima of the heatmap (3x3 max-pool) and returns the top-k boxes, so no
anchor tuning and no NMS are needed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

STRIDE = 4


def conv_bn(c_in: int, c_out: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv_bn(c_in, c_out, 3, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        return self.act(self.conv2(self.conv1(x)) + identity)


def stage(c_in: int, c_out: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(BasicBlock(c_in, c_out, 2), *[BasicBlock(c_out, c_out) for _ in range(blocks - 1)])


class JerseyNet(nn.Module):
    def __init__(self, widths=(64, 128, 192, 256), neck: int = 96, blocks: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3, 32, 3, 2), conv_bn(32, 32, 3, 1))  # stride 2
        chans = (32, *widths)
        self.stages = nn.ModuleList(stage(chans[i], chans[i + 1], blocks) for i in range(4))
        self.lateral = nn.ModuleList(nn.Conv2d(c, neck, 1) for c in widths)
        self.smooth = nn.ModuleList(conv_bn(neck, neck) for _ in widths[:-1])
        self.heatmap = self._head(neck, 1)
        self.wh = self._head(neck, 2)
        self.offset = self._head(neck, 2)
        nn.init.constant_(self.heatmap[-1].bias, -2.19)  # prior probability 0.1, as in CenterNet
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.heatmap[-1]:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _head(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(conv_bn(c_in, 64), nn.Conv2d(64, c_out, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = []
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
            feats.append(x)  # strides 4, 8, 16, 32
        p = self.lateral[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            lat = self.lateral[i](feats[i])
            p = self.smooth[i](lat + F.interpolate(p, size=lat.shape[-2:], mode="nearest"))
        return {"heatmap": self.heatmap(p), "wh": self.wh(p), "offset": self.offset(p)}


# ------------------------------------------------------------------- loss


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (CornerNet / CenterNet), alpha 2, beta 4."""
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred**2 * (1 - target) ** 4 * neg
    num_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def _gather(feat: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) gathered at flat indices (B, K) -> (B, K, C)."""
    b, c = feat.shape[:2]
    flat = feat.view(b, c, -1)
    return flat.gather(2, index.unsqueeze(1).expand(b, c, index.shape[1])).permute(0, 2, 1)


def giou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Generalised IoU (Rezatofighi et al., 2019) of boxes (..., 4) as [x1, y1, x2, y2]."""
    iw = (torch.min(a[..., 2], b[..., 2]) - torch.max(a[..., 0], b[..., 0])).clamp(min=0)
    ih = (torch.min(a[..., 3], b[..., 3]) - torch.max(a[..., 1], b[..., 1])).clamp(min=0)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + eps
    cw = torch.max(a[..., 2], b[..., 2]) - torch.min(a[..., 0], b[..., 0])
    ch = torch.max(a[..., 3], b[..., 3]) - torch.min(a[..., 1], b[..., 1])
    enclose = cw * ch + eps
    return inter / union - (enclose - union) / enclose


def _centre_boxes(wh: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Boxes around each object's centre, relative to its heatmap cell (feature-map units)."""
    wh = wh.clamp(min=0)
    return torch.cat([offset - wh / 2, offset + wh / 2], dim=-1)


def detection_loss(out: dict, batch: dict, wh_weight: float = 0.1,
                   giou_weight: float = 0.0) -> dict[str, torch.Tensor]:
    """Focal heatmap loss, L1 on size and offset, and optionally a GIoU box loss.

    The GIoU term scores the whole predicted box (size and offset together) at each object
    centre. It is off by default, which reproduces the CenterNet recipe.
    """
    hm = focal_loss(out["heatmap"], batch["heatmap"])
    mask = batch["mask"]
    n = mask.sum().clamp(min=1)
    pred_wh = _gather(out["wh"].float(), batch["index"])
    pred_off = _gather(out["offset"].float(), batch["index"])
    wh = (F.l1_loss(pred_wh, batch["wh"], reduction="none") * mask.unsqueeze(-1)).sum() / n
    off = (F.l1_loss(pred_off, batch["offset"], reduction="none") * mask.unsqueeze(-1)).sum() / n
    total = hm + wh_weight * wh + off
    losses = {"heatmap": hm.detach(), "wh": wh.detach(), "offset": off.detach()}
    if giou_weight > 0:
        g = giou(_centre_boxes(pred_wh, pred_off), _centre_boxes(batch["wh"], batch["offset"]))
        giou_loss = ((1 - g) * mask).sum() / n
        total = total + giou_weight * giou_loss
        losses["giou"] = giou_loss.detach()
    losses["loss"] = total
    return losses


# ------------------------------------------------------------------- decoding


@torch.no_grad()
def decode(out: dict, top_k: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Return boxes (B, K, 4) in input pixels [x1, y1, x2, y2] and scores (B, K)."""
    heat = torch.sigmoid(out["heatmap"].float())
    keep = (F.max_pool2d(heat, 3, stride=1, padding=1) == heat).float()
    heat = heat * keep
    b, _, h, w = heat.shape
    k = min(top_k, h * w)
    scores, idx = heat.view(b, -1).topk(k)
    ys = torch.div(idx, w, rounding_mode="floor").float()
    xs = (idx % w).float()
    off = _gather(out["offset"].float(), idx)
    wh = _gather(out["wh"].float(), idx).clamp(min=0)
    cx, cy = xs + off[..., 0], ys + off[..., 1]
    boxes = torch.stack(
        [cx - wh[..., 0] / 2, cy - wh[..., 1] / 2, cx + wh[..., 0] / 2, cy + wh[..., 1] / 2], dim=-1
    )
    return boxes * STRIDE, scores
