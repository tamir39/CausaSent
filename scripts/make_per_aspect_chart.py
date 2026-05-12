"""Generate per-aspect F1 bar chart for slide 7."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ASPECTS = [
    ("delivery",          0.4291),
    ("packaging",         0.3698),
    ("price",             0.3203),
    ("appearance",        0.3074),
    ("product_quality",   0.2974),
    ("customer_service",  0.2363),
    ("usability",         0.2295),
]

labels = [a for a, _ in ASPECTS]
values = [v for _, v in ASPECTS]

fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ypos = list(range(len(labels)))[::-1]
bars = ax.barh(ypos, values, color="#6366f1", height=0.62, edgecolor="none")

for bar, v in zip(bars, values):
    ax.text(
        v + 0.008, bar.get_y() + bar.get_height() / 2,
        f"{v:.2f}",
        va="center", ha="left",
        fontsize=14, fontweight="bold", color="#09090b",
    )

ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=14, color="#09090b")
ax.set_xlim(0, 0.5)
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.tick_params(axis="x", labelsize=11, colors="#52525b")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#e4e4e7")
ax.spines["bottom"].set_color("#e4e4e7")
ax.grid(axis="x", linestyle="--", linewidth=0.6, color="#e4e4e7", alpha=0.9)
ax.set_axisbelow(True)

ax.set_title(
    "ATE entity-level F1 theo aspect (test set)",
    fontsize=17, fontweight="bold", color="#09090b",
    loc="left", pad=14,
)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "report" / "per_aspect_f1.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
