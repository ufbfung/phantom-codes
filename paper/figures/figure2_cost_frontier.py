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

# Provider display labels (Python's str.capitalize() lowercases the
# remainder, which renders "OpenAI" as "Openai" — define explicitly).
PROVIDER_LABELS = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google"}


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

    # Wider aspect ratio to leave room on the right for callouts and at
    # the bottom for the legend, so neither overlaps the data area.
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.subplots_adjust(left=0.10, right=0.66, bottom=0.27, top=0.94)

    # Plot all points; Pareto-optimal get a heavier border.
    for _, row in df.iterrows():
        is_pareto = pareto.loc[row.name]
        ax.scatter(
            row["accuracy_pct"],
            row["cost_per_1k"],
            c=PROVIDER_COLORS[row["provider"]],
            marker=MODE_MARKERS[row["mode"]],
            s=110,
            edgecolors="black",
            linewidths=1.8 if is_pareto else 0.5,
            zorder=4 if is_pareto else 3,
        )

    ax.set_yscale("log")
    ax.set_xlim(73, 100)
    ax.set_ylim(0.05, 30)
    ax.set_xlabel("Top-1 accuracy (%)", fontsize=10)
    ax.set_ylabel("Cost per 1,000 correct predictions (USD)", fontsize=10)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, color="#ccc", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="#ddd", alpha=0.5)
    ax.set_axisbelow(True)

    yticks = [0.1, 1.0, 10.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${y:g}" for y in yticks], fontsize=9)
    ax.tick_params(axis="x", labelsize=9)

    # Callouts placed in the right-side margin (axes-fraction coords past
    # the right axis edge). Lines connect each label to its data point.
    annotations = {
        ("claude-opus-4-7", "constrained"): {
            "label": "Claude Opus 4.7\n(constrained)\n$13.30 / 1,000 correct\ndominated by Haiku 4.5",
            "xytext": (1.05, 0.92),
        },
        ("claude-haiku-4-5", "constrained"): {
            "label": "Claude Haiku 4.5\n(constrained)\n$4.40 / 1,000 correct",
            "xytext": (1.05, 0.55),
        },
        ("gpt-4o-mini", "constrained"): {
            "label": "GPT-4o-mini\n(constrained)\n$0.30 / 1,000 correct",
            "xytext": (1.05, 0.18),
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
            xycoords="data",
            xytext=spec["xytext"],
            textcoords="axes fraction",
            fontsize=8,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#888", linewidth=0.5),
            arrowprops=dict(
                arrowstyle="-",
                color="#666",
                linewidth=0.6,
                connectionstyle="arc3,rad=0.0",
                shrinkA=0, shrinkB=4,
            ),
            annotation_clip=False,
        )

    # Legend below plot, horizontal layout (3 columns).
    handles = []
    for provider, color in PROVIDER_COLORS.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=color,
                                   markeredgecolor="black", markersize=8,
                                   label=PROVIDER_LABELS[provider]))
    for mode, marker in MODE_MARKERS.items():
        handles.append(plt.Line2D([0], [0], marker=marker, color="w",
                                   markerfacecolor="#888",
                                   markeredgecolor="black", markersize=8,
                                   label=MODE_LABELS[mode]))
    handles.append(plt.Line2D([0], [0], marker="s", color="w",
                               markerfacecolor="white",
                               markeredgecolor="black", markersize=8,
                               markeredgewidth=1.8,
                               label="Pareto-optimal"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.38, 0.0), ncol=4,
               fontsize=8, frameon=False,
               handletextpad=0.5, columnspacing=1.4)

    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    render()
