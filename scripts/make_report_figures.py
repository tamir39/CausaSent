"""Generate figures for docs/report/report.md.

Creates:
  docs/report/figures/phobert_train_curve.png   (Hình 4.1)
  docs/report/figures/per_class_f1.png          (Hình 6.1)

Numbers are hard-coded from the actual Kaggle eval output reported in
docs/report/report.md sections 4.4 and 6.2 — re-run on Kaggle and update
this script if the metrics change.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "report" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hình 4.1 — PhoBERT training trajectory (8 epochs)

EPOCHS = [1, 2, 3, 4, 5, 6, 7, 8]
TRAIN_LOSS = [3.95, 1.71, 1.27, 0.93, 0.75, 0.63, 0.54, 0.51]
VAL_LOSS   = [1.93, 1.53, 1.29, 1.31, 1.34, 1.44, 1.45, 1.47]
ASP_F1     = [0.000, 0.000, 0.080, 0.144, 0.240, 0.295, 0.349, 0.327]
CAUSE_F1   = [0.000, 0.343, 0.437, 0.459, 0.450, 0.456, 0.453, 0.434]
MEAN_F1    = [0.000, 0.172, 0.259, 0.302, 0.345, 0.376, 0.401, 0.380]
BEST_EPOCH = 7

fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))

ax_loss.plot(EPOCHS, TRAIN_LOSS, "o-", label="train loss", color="#1f77b4")
ax_loss.plot(EPOCHS, VAL_LOSS,   "s--", label="val loss",  color="#ff7f0e")
ax_loss.axvline(BEST_EPOCH, color="grey", ls=":", lw=1, label=f"best (epoch {BEST_EPOCH})")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Loss curves")
ax_loss.grid(alpha=0.3)
ax_loss.legend()

ax_f1.plot(EPOCHS, ASP_F1,   "o-", label="aspect-sent F1", color="#d62728")
ax_f1.plot(EPOCHS, CAUSE_F1, "s-", label="cause F1",       color="#2ca02c")
ax_f1.plot(EPOCHS, MEAN_F1,  "^--", label="mean F1",       color="#9467bd")
ax_f1.axvline(BEST_EPOCH, color="grey", ls=":", lw=1)
ax_f1.set_xlabel("Epoch")
ax_f1.set_ylabel("Entity-F1 (val)")
ax_f1.set_title("Validation entity-F1")
ax_f1.set_ylim(-0.02, 0.55)
ax_f1.grid(alpha=0.3)
ax_f1.legend()

fig.suptitle("Hình 4.1 — PhoBERT-large huấn luyện 8 epoch trên Kaggle T4")
fig.tight_layout()
fig.savefig(OUT_DIR / "phobert_train_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT_DIR / 'phobert_train_curve.png'}")

# ---------------------------------------------------------------------------
# Hình 6.1 — per-class token-level F1 (aspect-sentiment head)

CLASSES = [
    "APP-NEG", "APP-POS", "PRICE-NEG", "PRICE-NEU", "PRICE-POS",
    "QUAL-NEG", "QUAL-NEU", "QUAL-POS",
    "SVC-NEG", "SVC-NEU", "SVC-POS",
    "USE-NEG", "USE-POS",
]
F1      = [0.000, 0.356, 0.118, 0.000, 0.148, 0.211, 0.000, 0.275, 0.000, 0.000, 0.639, 0.000, 0.000]
SUPPORT = [11, 33, 6, 4, 21, 18, 2, 79, 5, 1, 31, 15, 2]

# Color by support tier (more support = darker)
def support_color(s: int) -> str:
    if s >= 30: return "#1a5fb4"
    if s >= 15: return "#3584e4"
    if s >= 5:  return "#99c1f1"
    return "#deddda"

colors = [support_color(s) for s in SUPPORT]

fig, ax = plt.subplots(figsize=(11, 5))
xs = np.arange(len(CLASSES))
bars = ax.bar(xs, F1, color=colors, edgecolor="black", linewidth=0.5)

# annotate support count on top of each bar
for x, f1, sup in zip(xs, F1, SUPPORT):
    ax.text(x, max(f1, 0) + 0.015, f"n={sup}", ha="center", va="bottom",
            fontsize=8, color="#444")

# threshold line at 0.3 (rough "respectable" line for this dataset)
ax.axhline(0.30, color="red", ls=":", lw=1, alpha=0.6, label="F1 = 0,30")

ax.set_xticks(xs)
ax.set_xticklabels(CLASSES, rotation=35, ha="right")
ax.set_ylabel("Token-level F1 (seqeval)")
ax.set_ylim(0, 0.75)
ax.set_title("Hình 6.1 — F1 theo lớp aspect-sentiment trên test set (228 entity)")
ax.grid(axis="y", alpha=0.3)

# legend for support tiers
from matplotlib.patches import Patch
legend = [
    Patch(facecolor="#1a5fb4", edgecolor="black", label="support ≥ 30"),
    Patch(facecolor="#3584e4", edgecolor="black", label="support 15–29"),
    Patch(facecolor="#99c1f1", edgecolor="black", label="support 5–14"),
    Patch(facecolor="#deddda", edgecolor="black", label="support < 5"),
]
ax.legend(handles=legend, loc="upper left", framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT_DIR / "per_class_f1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT_DIR / 'per_class_f1.png'}")
