"""Figure 2: cost-per-1000-correct vs top-1 accuracy.

Renders one point per (LLM, prompting mode) configuration in the
deployment-relevant accuracy range (top-1 >= 75%). Color encodes
provider, marker shape encodes prompting mode. Pareto-optimal
configurations (no other point is both cheaper and more accurate)
are drawn with a heavy black border, so the deployment frontier is
visible without a connecting line.

Cost is reported as USD per 1,000 correct predictions to give
deployment-relevant magnitude (configurations span ~$0.10 to $13
per 1,000 correct predictions).

Source: results/summary/n125_run_v2/cost_per_correct.csv (per-(model,
prompting-mode) totals from the n=125 headline run; non-LLM
baselines have zero API cost and are excluded; LLM configurations
below 75% top-1 accuracy are excluded as not deployment-relevant).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Embed fonts as TrueType (type 42) so XeLaTeX preserves glyph metrics
# when the PDF is scaled into the manuscript at width=95%. Default
# (Type 3, glyph-as-postscript-program) renders as visibly stretched
# when XeLaTeX rasterizes for inclusion.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "results" / "summary" / "n125_run_v2" / "cost_per_correct.csv"
OUT_PATH = Path(__file__).resolve().parent / "figure2_cost_frontier.pdf"

ACCURACY_FLOOR = 0.75  # filter to deployment-relevant subset

PROVIDER_COLORS = {
    "anthropic": "#D97757",  # warm orange
    "openai": "#10A37F",     # green
    "google": "#4285F4",     # blue
}

MODE_MARKERS = {
    "zeroshot": "o",
    "constrained": "s",
    "rag": "^",
}

MODE_LABELS = {"zeroshot": "Zero-shot", "constrained": "Constrained", "rag": "RAG"}


def classify_provider(model_name: str) -> str:
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("gpt"):
        return "openai"
    if model_name.startswith("gemini"):
        return "google"
    return "other"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df[["model_id", "mode"]] = df["model"].str.split(":", expand=True)
    df["provider"] = df["model_id"].apply(classify_provider)
    df["accuracy_pct"] = (df["n_exact"] / df["n"]) * 100
    df["cost_per_1k"] = df["cost_per_correct_usd"] * 1000
    df = df.dropna(subset=["cost_per_1k"])
    df = df[df["provider"] != "other"]
    df = df[df["accuracy_pct"] >= ACCURACY_FLOOR * 100]
    return df.reset_index(drop=True)


def pareto_mask(df: pd.DataFrame) -> pd.Series:
    """A point is Pareto-optimal when no other point has both higher
    accuracy AND lower cost."""
    is_pareto = []
    for _, row in df.iterrows():
        dominated = (
            (df["accuracy_pct"] >= row["accuracy_pct"])
            & (df["cost_per_1k"] <= row["cost_per_1k"])
            & ~(
                (df["accuracy_pct"] == row["accuracy_pct"])
                & (df["cost_per_1k"] == row["cost_per_1k"])
            )
        )
        is_pareto.append(not dominated.any())
    return pd.Series(is_pareto, index=df.index)


def render() -> None:
    df = load()
    pareto = pareto_mask(df)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    # Plot all points; Pareto-optimal get a heavier border.
    for _, row in df.iterrows():
        is_pareto = pareto.loc[row.name]
        ax.scatter(
            row["accuracy_pct"],
            row["cost_per_1k"],
            c=PROVIDER_COLORS[row["provider"]],
            marker=MODE_MARKERS[row["mode"]],
            s=130,
            edgecolors="black",
            linewidths=1.8 if is_pareto else 0.5,
            zorder=4 if is_pareto else 3,
        )

    ax.set_yscale("log")
    ax.set_xlim(73, 100)
    ax.set_ylim(0.05, 60)
    ax.set_xlabel("Top-1 accuracy (%)", fontsize=11)
    ax.set_ylabel("Cost per 1,000 correct predictions (USD)", fontsize=11)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, color="#ccc", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="#ddd", alpha=0.5)
    ax.set_axisbelow(True)

    yticks = [0.1, 1.0, 10.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${y:g}" for y in yticks], fontsize=10)
    ax.tick_params(axis="x", labelsize=10)

    # Three deployment-frontier annotations with rounded boxes and curved arrows.
    annotations = {
        ("gpt-4o-mini", "constrained"): {
            "label": "GPT-4o-mini (constrained)\n$0.30 per 1,000 correct",
            "xytext": (-160, 40),
        },
        ("claude-haiku-4-5", "constrained"): {
            "label": "Claude Haiku 4.5 (constrained)\n$4.40 per 1,000 correct",
            "xytext": (-200, 18),
        },
        ("claude-opus-4-7", "constrained"): {
            "label": "Claude Opus 4.7 (constrained)\n$13.30 per 1,000 correct\ndominated by Haiku 4.5",
            "xytext": (-220, 55),
        },
    }
    for (model_id, mode), spec in annotations.items():
        row = df[(df["model_id"] == model_id) & (df["mode"] == mode)]
        if row.empty:
            continue
        x = float(row["accuracy_pct"].iloc[0])
        y = float(row["cost_per_1k"].iloc[0])
        ax.annotate(
            spec["label"],
            xy=(x, y),
            xytext=spec["xytext"],
            textcoords="offset points",
            fontsize=8.5,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#888", linewidth=0.6),
            arrowprops=dict(
                arrowstyle="-",
                color="#444",
                linewidth=0.8,
                connectionstyle="arc3,rad=0.18",
            ),
        )

    # Single combined legend in lower-left.
    handles = []
    for provider, color in PROVIDER_COLORS.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=color,
                                   markeredgecolor="black", markersize=10,
                                   label=provider.capitalize()))
    handles.append(plt.Line2D([0], [0], marker="None", linestyle="None", label=" "))
    for mode, marker in MODE_MARKERS.items():
        handles.append(plt.Line2D([0], [0], marker=marker, color="w",
                                   markerfacecolor="#888",
                                   markeredgecolor="black", markersize=10,
                                   label=MODE_LABELS[mode]))
    handles.append(plt.Line2D([0], [0], marker="None", linestyle="None", label=" "))
    handles.append(plt.Line2D([0], [0], marker="s", color="w",
                               markerfacecolor="white",
                               markeredgecolor="black", markersize=10,
                               markeredgewidth=1.8,
                               label="Pareto-optimal"))
    ax.legend(handles=handles, loc="lower left", fontsize=9,
              frameon=True, framealpha=0.95, edgecolor="#888",
              ncol=1, handletextpad=0.6, borderpad=0.6)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    render()
