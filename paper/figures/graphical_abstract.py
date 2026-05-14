"""Graphical Abstract for IEEE J-BHI submission.

IEEE J-BHI requires a Graphical Abstract (GA) image:
  - 660 x 295 pixels
  - JPG format
  - < 45 KB

The GA should be "a figure or image from the accepted article" that
"summarizes the main contributions." We adapt Figure 2 (cost-per-correct
versus top-1 accuracy frontier) since it carries the deployment-decisive
headline finding and its aspect ratio (~2:1) is closest to GA dimensions
(660:295 ≈ 2.24:1).

Design changes vs the in-paper figure:
  - Right-margin callout boxes are dropped (would crowd the small canvas).
  - Inline labels next to the three Pareto-optimal points carry the same
    information in less space.
  - Tighter axis margins, smaller fonts, smaller markers — all sized for
    660 x 295 px.
  - Output is JPG at the largest quality that keeps file size <45KB.

Source data and Pareto-mask logic are imported from
figure2_cost_frontier.py to keep the two figures in sync.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from figure2_cost_frontier import (
    MODE_LABELS,
    MODE_MARKERS,
    PROVIDER_COLORS,
    PROVIDER_LABELS,
    load,
    pareto_mask,
)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"

OUT_PATH = Path(__file__).resolve().parent / "graphical_abstract.jpg"

# Target dimensions. matplotlib renders at figsize_inches * dpi pixels;
# 6.6 inches × 100 dpi = 660 px, 2.95 × 100 = 295 px.
GA_WIDTH_PX = 660
GA_HEIGHT_PX = 295
GA_DPI = 100
MAX_FILESIZE_BYTES = 45_000

# Inline label specs for the 3 Pareto-optimal callouts. Coordinates in
# data space (accuracy_pct, cost_per_1k) for the anchor + offset (text
# placement relative to point).
INLINE_LABELS = {
    ("gpt-4o-mini", "constrained"): {
        "label": "GPT-4o-mini\n$0.30/1k correct",
        "xytext_offset": (-12, -22),  # below-left
        "ha": "right",
    },
    ("claude-haiku-4-5", "constrained"): {
        "label": "Claude Haiku 4.5\n$4.40/1k correct",
        "xytext_offset": (-12, +8),  # above-left
        "ha": "right",
    },
    ("claude-opus-4-7", "constrained"): {
        "label": "Claude Opus 4.7\n$13.30/1k (dominated)",
        "xytext_offset": (-12, +8),  # above-left
        "ha": "right",
    },
}


def render() -> None:
    df = load()
    pareto = pareto_mask(df)

    fig, ax = plt.subplots(figsize=(GA_WIDTH_PX / GA_DPI, GA_HEIGHT_PX / GA_DPI))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.27, top=0.95)

    # Plot all points; Pareto-optimal get a heavier border.
    for _, row in df.iterrows():
        is_pareto = pareto.loc[row.name]
        ax.scatter(
            row["accuracy_pct"],
            row["cost_per_1k"],
            c=PROVIDER_COLORS[row["provider"]],
            marker=MODE_MARKERS[row["mode"]],
            s=55,
            edgecolors="black",
            linewidths=1.4 if is_pareto else 0.4,
            zorder=4 if is_pareto else 3,
        )

    ax.set_yscale("log")
    ax.set_xlim(73, 100)
    ax.set_ylim(0.05, 30)
    ax.set_xlabel("Top-1 accuracy (%)", fontsize=8)
    ax.set_ylabel("Cost per 1k correct (USD)", fontsize=8)
    ax.grid(True, which="major", linestyle="-", linewidth=0.3, color="#ccc", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2, color="#ddd", alpha=0.5)
    ax.set_axisbelow(True)

    yticks = [0.1, 1.0, 10.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${y:g}" for y in yticks], fontsize=7)
    ax.tick_params(axis="x", labelsize=7)

    # Inline labels for the 3 Pareto-optimal points.
    for (model_id, mode), spec in INLINE_LABELS.items():
        row = df[(df["model_id"] == model_id) & (df["mode"] == mode)]
        if row.empty:
            continue
        x = float(row["accuracy_pct"].iloc[0])
        y = float(row["cost_per_1k"].iloc[0])
        ax.annotate(
            spec["label"],
            xy=(x, y),
            xytext=spec["xytext_offset"],
            textcoords="offset points",
            fontsize=6.5,
            ha=spec["ha"],
            va="center",
            color="#222",
        )

    # Compact bottom legend: providers + Pareto-optimal marker, in one row.
    handles = []
    for provider, color in PROVIDER_COLORS.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=color,
                                   markeredgecolor="black", markersize=6,
                                   label=PROVIDER_LABELS[provider]))
    handles.append(plt.Line2D([0], [0], marker="s", color="w",
                               markerfacecolor="white",
                               markeredgecolor="black", markersize=6,
                               markeredgewidth=1.4,
                               label="Pareto-optimal"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=4,
               fontsize=6.5, frameon=False,
               handletextpad=0.4, columnspacing=1.2)

    # Save as JPG, tuning quality to stay under MAX_FILESIZE_BYTES.
    # Save once at high quality; if too big, drop quality progressively.
    for quality in (95, 90, 85, 80, 75, 70):
        buf = BytesIO()
        fig.savefig(buf, format="jpg", dpi=GA_DPI, pil_kwargs={"quality": quality})
        size = buf.tell()
        if size <= MAX_FILESIZE_BYTES:
            OUT_PATH.write_bytes(buf.getvalue())
            print(f"[graphical_abstract] {OUT_PATH.name}: "
                  f"{GA_WIDTH_PX}x{GA_HEIGHT_PX} JPG, "
                  f"quality={quality}, {size:,} bytes (cap {MAX_FILESIZE_BYTES:,})")
            return
    raise RuntimeError(
        f"Could not get under {MAX_FILESIZE_BYTES:,} bytes even at quality=70"
    )


if __name__ == "__main__":
    render()
