"""Figure 1: hallucination rate heatmap, model x prompting mode x degradation mode.

Rows are 24 (LLM, prompting mode) configurations grouped by provider
and prompting mode. Columns are the four degradation modes (D1-D4).
Cell color encodes hallucination percentage; cells annotate the
percentage value.

Source: results/summary/n125_run_v2/hallucination.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "results" / "summary" / "n125_run_v2" / "hallucination.csv"
OUT_PATH = Path(__file__).resolve().parent / "figure1_hallucination_heatmap.pdf"

LLM_ORDER = [
    # Anthropic
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    # OpenAI
    "gpt-4o-mini",
    "gpt-5.5",
    # Google
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
]
MODE_ORDER = ["zeroshot", "constrained", "rag"]
DEGRADE_ORDER = ["D1_full", "D2_no_code", "D3_text_only", "D4_abbreviated"]
DEGRADE_LABELS = ["D1\nfull", "D2\nno code", "D3\ntext only", "D4\nabbreviated"]
MODE_LABELS = {"zeroshot": "zero-shot", "constrained": "constrained", "rag": "RAG"}

PRETTY_MODEL = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5.5": "GPT-5.5",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash Preview",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df[["model_id", "prompting_mode"]] = df["model"].str.split(":", expand=True)
    df = df[df["model_id"].isin(LLM_ORDER)]
    df["rate_pct"] = df["rate"] * 100
    return df


def render() -> None:
    df = load()

    row_keys = [(m, pm) for m in LLM_ORDER for pm in MODE_ORDER]
    matrix = np.zeros((len(row_keys), len(DEGRADE_ORDER)))
    for r, (model_id, pm) in enumerate(row_keys):
        for c, deg in enumerate(DEGRADE_ORDER):
            cell = df[
                (df["model_id"] == model_id)
                & (df["prompting_mode"] == pm)
                & (df["mode"] == deg)
            ]
            matrix[r, c] = float(cell["rate_pct"].iloc[0]) if not cell.empty else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=15, aspect="auto")

    # Cell annotations.
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if np.isnan(val):
                continue
            color = "white" if val > 8 else "black"
            ax.text(
                c, r,
                "0" if val == 0 else f"{val:.1f}",
                ha="center", va="center",
                fontsize=7, color=color,
            )

    # Y-axis: row labels with model + prompting mode.
    ylabels = [f"{PRETTY_MODEL[m]}  ({MODE_LABELS[pm]})" for m, pm in row_keys]
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xticks(np.arange(len(DEGRADE_ORDER)))
    ax.set_xticklabels(DEGRADE_LABELS, fontsize=8)

    # Group separators between providers (every 9 rows = 3 models × 3 modes).
    for sep_row in [9, 15]:  # after Anthropic (3*3=9), after OpenAI (+2*3=6)
        ax.axhline(sep_row - 0.5, color="black", linewidth=1.0)
    # Mode separators within each provider group.
    for r in range(len(row_keys) - 1):
        if (r + 1) % 3 != 0 or r + 1 in {9, 15}:
            continue
        ax.axhline(r + 0.5, color="#bbb", linewidth=0.4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="max")
    cbar.set_label("Hallucination rate (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Degradation mode", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    render()
