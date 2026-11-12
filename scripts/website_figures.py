"""Website figures built from the archived Kaggle summaries and run log.

    python scripts/website_figures.py --out figures_web
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "experiments"
sys.path.insert(0, str(ROOT / "paper"))
BLUE, YELLOW, RED, GREEN, GREY = "#1F4E9C", "#E3A800", "#C8102E", "#5FA65A", "#9a9a93"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def ap_comparison(out: Path) -> None:
    main = json.loads((EXP / "2026-11-01_kaggle_t4" / "summary.json").read_text())
    yolo = json.loads((EXP / "2026-11-01_kaggle_t4_yolo" / "summary.json").read_text())["over_seeds"]
    jn = main["jersey_net_over_seeds"]["ap50_95"]
    rows = [("Mean-box baseline", main["results"]["Mean-box baseline"]["test"]["ap50_95"], 0.0, GREY),
            ("v1 CNN regressor (2025)", main["results"]["v1 CNN regressor (15 epochs)"]["test"]["ap50_95"], 0.0, GREY),
            ("JerseyNet, from scratch, 256 px", jn["mean"], jn["std"], BLUE)]
    for name in ("YOLOv8n, from scratch, 256 px", "YOLOv8n, COCO-pretrained, 256 px", "YOLOv8n, COCO-pretrained, 640 px"):
        rows.append((name, yolo[name]["ap50_95"]["mean"], yolo[name]["ap50_95"]["std"], GREY))
    rows.reverse()
    fig, ax = plt.subplots(figsize=(10.7, 5.1))
    names, vals, errs, cols = zip(*rows)
    ax.barh(names, vals, xerr=errs, color=cols, height=0.62, capsize=4, error_kw={"ecolor": "#333333"})
    for y, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + e + 0.006, y, "%.3f" % v, va="center", fontsize=12, color="#333333")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Test AP50:95 (mean of 3 seeds; error bars: 1 std)", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(axis="x", color="#e4e4e4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "jersey-ap-comparison.png", dpi=150)
    plt.close(fig)


def val_curves(out: Path) -> None:
    runs, current = {}, None
    for line in (EXP / "2026-11-01_kaggle_t4" / "run.log").read_text().splitlines():
        m = re.search(r"=== jersey_net_seed(\d)", line)
        if m:
            current = int(m.group(1))
            runs[current] = []
            continue
        if current is not None and '"val_ap50_95"' in line and '"epoch"' in line:
            rec = json.loads(line.split("] ", 1)[1])
            runs[current].append((rec["epoch"], rec["val_ap50_95"]))
    best = {0: 115, 1: 95, 2: 90}
    styles = {0: (BLUE, "o"), 1: (YELLOW, "s"), 2: (GREEN, "^")}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for seed, pts in sorted(runs.items()):
        colour, marker = styles[seed]
        ep, ap = zip(*pts)
        ax.plot(ep, ap, color=colour, marker=marker, ms=5, lw=2, label="seed %d" % seed)
        ax.plot(best[seed], dict(pts)[best[seed]], marker, color=colour, ms=11)
    ax.set_title("JerseyNet validation AP50:95 during training (large marker: selected checkpoint)", fontsize=11, loc="left")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation AP50:95", fontsize=12)
    ax.set_xlim(0, 145)
    ax.set_ylim(0, 0.3)
    ax.grid(axis="y", color="#e4e4e4")
    ax.legend(frameon=False, loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "jersey-val-curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_web")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    ap_comparison(out)
    val_curves(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
