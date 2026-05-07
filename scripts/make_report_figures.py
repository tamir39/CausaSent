"""Generate figures for docs/report/report.md.

Creates:
  docs/report/figures/architecture.png          (Hình 3.1)
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
# Hình 3.1 — System architecture

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def _box(ax, xy, w, h, text, *, fc, ec="black", fs=9, lw=1.0):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def _arrow(ax, p1, p2, *, label=None, ls="-"):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.1, color="#444", linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + 0.05, my, label, fontsize=8, color="#333", style="italic")

fig, ax = plt.subplots(figsize=(10, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.set_aspect("equal")
ax.axis("off")

C_INPUT  = "#fff3b0"   # input/output (yellow)
C_PROC   = "#d8e2dc"   # preprocessing (grey-green)
C_MODEL  = "#a8dadc"   # neural model (cyan)
C_DECODE = "#ffd6a5"   # decode/post (orange)
C_OUT    = "#caffbf"   # output (green)

# Input
_box(ax, (3.0, 12.6), 4.0, 0.9, "Review tiếng Việt thô", fc=C_INPUT, fs=10)

# VnCoreNLP
_box(ax, (3.0, 11.0), 4.0, 0.9, "VnCoreNLP word-segmenter\n(annotators=['wseg'])", fc=C_PROC, fs=9)

# PhoBERT encoder
_box(ax, (2.5, 8.6), 5.0, 1.6,
     "PhoBERT-large encoder\n(370M params, 24 layers, hidden=1024)",
     fc=C_MODEL, fs=10, lw=1.4)

# Two heads
_box(ax, (0.4, 6.6), 4.2, 1.2,
     "Aspect-Sentiment head\nLinear(1024 → 43 nhãn BIO)", fc=C_MODEL, fs=9)
_box(ax, (5.4, 6.6), 4.2, 1.2,
     "Cause head\nLinear(1024 → 3 nhãn BIO)", fc=C_MODEL, fs=9)

# Decode
_box(ax, (1.5, 4.4), 7.0, 1.4,
     "Decode (src/inference/decode.py)\n"
     "BIO → spans · ghép aspect ↔ cause theo IoU max\n"
     "drop empty cause · dedup · confidence = mean softmax",
     fc=C_DECODE, fs=9)

# Tuples (aspect, sentiment, cause)
_box(ax, (1.5, 2.8), 7.0, 0.8,
     "list[(aspect, sentiment, cause_span, confidence)]",
     fc=C_OUT, fs=9)

# mT5
_box(ax, (1.5, 1.0), 7.0, 1.4,
     "mT5-base action generator (580M)\n"
     'Input: "<aspect> <sentiment> | <cause_text> | <review>"\n'
     "Output: action ngắn (≤10 từ, mệnh lệnh, beam=4)",
     fc=C_MODEL, fs=9, lw=1.4)

# Final output
_box(ax, (0.7, -0.4), 8.6, 0.9,
     "list[(aspect, sentiment, cause_span, action, confidence)]",
     fc=C_OUT, fs=10, lw=1.2)

# Arrows
_arrow(ax, (5.0, 12.6), (5.0, 11.9))
_arrow(ax, (5.0, 11.0), (5.0, 10.2), label="list[str] words")
_arrow(ax, (4.0, 8.6),  (2.5, 7.8))
_arrow(ax, (6.0, 8.6),  (7.5, 7.8))
_arrow(ax, (2.5, 6.6),  (3.5, 5.8), label="BIO seq")
_arrow(ax, (7.5, 6.6),  (6.5, 5.8), label="BIO seq")
_arrow(ax, (5.0, 4.4),  (5.0, 3.6))
_arrow(ax, (5.0, 2.8),  (5.0, 2.4))
_arrow(ax, (5.0, 1.0),  (5.0, 0.5))

ax.set_title("Hình 3.1 — Kiến trúc CausaSent end-to-end", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "architecture.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT_DIR / 'architecture.png'}")

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
