"""Figures for the paper, built from the archived run logs and summaries.

    python paper/make_figure.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "experiments"
OUT = Path(__file__).resolve().parent / "figures"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
# House palette: blue, yellow, red, light green
COLOURS = ["#1F4E9C", "#E3A800", "#C8102E", "#5FA65A"]


def val_curves() -> None:
    log = (EXP / "2026-11-01_kaggle_t4" / "run.log").read_text().splitlines()
    runs, current = {}, None
    for line in log:
        m = re.search(r"=== jersey_net_seed(\d)", line)
        if m:
            current = int(m.group(1))
            runs[current] = []
            continue
        if current is not None and '"val_ap50_95"' in line and '"epoch"' in line:
            rec = json.loads(line.split("] ", 1)[1])
            runs[current].append((rec["epoch"], rec["val_ap50_95"]))
    best = {0: 115, 1: 95, 2: 90}
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for seed, pts in sorted(runs.items()):
        ep, ap = zip(*pts, strict=True)
        ax.plot(ep, ap, color=COLOURS[seed], lw=1.2, label=f"seed {seed}")
        b = dict(pts)[best[seed]]
        ax.plot(best[seed], b, "o", color=COLOURS[seed], ms=5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation AP50:95")
    ax.set_xlim(0, 142)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "val_ap_curves.pdf")
    plt.close(fig)


def data_efficiency() -> None:
    """Test AP50:95 against the share of training images (study runs plus the full-data runs)."""
    agg = json.loads((EXP / "2026-11-07_kaggle_t4_study" / "summary.json").read_text())["aggregate"]
    # Full-data runs, mean and standard deviation over three seeds (docs/experiments/2026-11-01_*).
    full = {"JerseyNet, from scratch, 256 px": (0.241, 0.006),
            "YOLOv8n, from scratch, 256 px": (0.352, 0.013),
            "YOLOv8n, COCO-pretrained, 256 px": (0.384, 0.003)}
    labels = {"JerseyNet, from scratch, 256 px": "JerseyNet, scratch",
              "YOLOv8n, from scratch, 256 px": "YOLOv8n, scratch",
              "YOLOv8n, COCO-pretrained, 256 px": "YOLOv8n, COCO"}
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for i, (model, label) in enumerate(labels.items()):
        pts = [(a["fraction"], a["ap50_95"]["mean"], a["ap50_95"]["std"])
               for a in agg.values() if a["model"] == model]
        pts.append((1.0, *full[model]))
        x, y, e = zip(*sorted(pts), strict=True)
        ax.errorbar(x, y, yerr=e, color=COLOURS[i], marker="o", ms=3.5, lw=1.2, capsize=2, label=label)
    matched = next(a for a in agg.values() if "JerseyNet augmentation" in a["model"])
    ax.errorbar([1.0], [matched["ap50_95"]["mean"]], yerr=[matched["ap50_95"]["std"]], color=COLOURS[1],
                marker="o", mfc="white", ms=4.5, lw=0, elinewidth=1.2, capsize=2,
                label="YOLOv8n, scratch,\nJerseyNet augmentation")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 1.0], ["10%\n(317)", "25%\n(792)", "100%\n(3,165)"])
    ax.minorticks_off()
    ax.set_xlabel("training images")
    ax.set_ylabel("test AP50:95")
    ax.set_ylim(0.08, 0.42)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    fig.savefig(OUT / "data_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    val_curves()
    data_efficiency()
