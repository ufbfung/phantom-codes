"""Figure 2: cost-per-correct vs top-1 accuracy scatter.

Renders one point per (LLM, prompting mode) configuration. Color
encodes provider, marker shape encodes prompting mode. Pareto
frontier (lower-cost-or-higher-accuracy) is highlighted, with
GPT-4o-mini constrained and Claude Haiku 4.5 constrained labelled
as the deployment frontier and Claude Opus 4.7 constrained labelled
as a dominated configuration.

Source: results/summary/n125_run_v2/cost_per_correct.csv (per-(model,
prompting-mode) totals from the n=125 headline run; non-LLM
baselines have zero API cost and are excluded).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "results" / "summary" / "n125_run_v2" / "cost_per_correct.csv"
OUT_PATH = Path(__file__).resolve().parent / "figure2_cost_frontier.pdf"

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
    df["accuracy"] = df["n_exact"] / df["n"]
    df = df.dropna(subset=["cost_per_correct_usd"])
    df = df[df["provider"] != "other"]
    return df


def pareto_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask of Pareto-optimal points.

    A point is Pareto-optimal when no other point has both higher
    accuracy and lower cost-per-correct.
    """
    is_pareto = []
    for _, row in df.iterrows():
        dominated = (
            (df["accuracy"] >= row["accuracy"])
            & (df["cost_per_correct_usd"] <= row["cost_per_correct_usd"])
            & ~(
                (df["accuracy"] == row["accuracy"])
                & (df["cost_per_correct_usd"] == row["cost_per_correct_usd"])
            )
        )
        is_pareto.append(not dominated.any())
    return pd.Series(is_pareto, index=df.index)


def render() -> None:
    df = load()
    df = df.sort_values("accuracy")

    pareto = pareto_mask(df)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Plot the Pareto frontier as a connecting line first (background).
    pareto_sorted = df[pareto].sort_values("accuracy")
    ax.plot(
        pareto_sorted["accuracy"] * 100,
        pareto_sorted["cost_per_correct_usd"],
        color="#888",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
        label="Pareto frontier",
    )

    # Plot all configurations.
    for (provider, mode), group in df.groupby(["provider", "mode"]):
        ax.scatter(
            group["accuracy"] * 100,
            group["cost_per_correct_usd"],
            c=PROVIDER_COLORS[provider],
            marker=MODE_MARKERS[mode],
            s=60,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_ylabel("Cost per correct prediction (USD, log scale)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate the deployment-frontier configurations.
    annotations = {
        ("gpt-4o-mini", "constrained"): ("GPT-4o-mini\n(constrained)", (8, -18)),
        ("claude-haiku-4-5", "constrained"): ("Claude Haiku 4.5\n(constrained)", (10, 0)),
        ("claude-opus-4-7", "constrained"): ("Claude Opus 4.7\n(constrained, dominated)", (-180, 8)),
    }
    for (model_id, mode), (label, offset) in annotations.items():
        row = df[(df["model_id"] == model_id) & (df["mode"] == mode)]
        if row.empty:
            continue
        x = float(row["accuracy"].iloc[0]) * 100
        y = float(row["cost_per_correct_usd"].iloc[0])
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="left",
            arrowprops=dict(arrowstyle="-", color="black", linewidth=0.5),
        )

    # Two-part legend: providers (color) and modes (marker shape).
    provider_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markeredgecolor="black", markersize=8, label=provider.capitalize())
        for provider, color in PROVIDER_COLORS.items()
    ]
    mode_handles = [
        plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="#666",
                   markeredgecolor="black", markersize=8,
                   label={"zeroshot": "Zero-shot", "constrained": "Constrained",
                          "rag": "RAG"}[mode])
        for mode, marker in MODE_MARKERS.items()
    ]
    pareto_handle = plt.Line2D([0], [0], color="#888", linestyle="--",
                                linewidth=1.0, label="Pareto frontier")
    legend1 = ax.legend(handles=provider_handles, loc="upper left",
                        title="Provider", fontsize=8, title_fontsize=8,
                        frameon=True, framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=mode_handles + [pareto_handle], loc="lower left",
              title="Prompting mode", fontsize=8, title_fontsize=8,
              frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    render()
