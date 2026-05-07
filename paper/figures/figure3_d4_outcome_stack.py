"""Figure 3: D4 abbreviation-stress outcome distribution per (model, mode).

Horizontal stacked bars showing the six-way outcome distribution for
each non-trivial configuration evaluated on the D4_abbreviated
degradation mode. Bars are ordered by exact_match percentage
descending, so the deployment-best configurations appear at the top.

The six stack segments use a color-blind-safe palette:
- exact_match, category_match, chapter_match: greens (decreasing
  intensity), since hierarchical matches are progressively weaker
  forms of correctness.
- out_of_domain: neutral grey for "real ICD-10 code, but wrong".
- no_prediction: blue-grey for abstention, distinguishing it from
  errors per se.
- hallucination: red, since fabrication is the deployment-critical
  failure mode the paper is built around.

Source: results/summary/n125_run_v2/headline.csv (filtered to
mode == 'D4_abbreviated', excluding pubmedbert:classifier and
baseline:exact, which carry zero deployment-relevant signal under
this stress).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "results" / "summary" / "n125_run_v2" / "headline.csv"
OUT_PATH = Path(__file__).resolve().parent / "figure3_d4_outcome_stack.pdf"

EXCLUDE_MODELS = {"pubmedbert:classifier", "baseline:exact"}

BUCKET_ORDER = [
    "exact_match",
    "category_match",
    "chapter_match",
    "out_of_domain",
    "no_prediction",
    "hallucination",
]
BUCKET_LABELS = {
    "exact_match": "Exact match",
    "category_match": "Category match",
    "chapter_match": "Chapter match",
    "out_of_domain": "Out of domain",
    "no_prediction": "No prediction",
    "hallucination": "Hallucination",
}
BUCKET_COLORS = {
    "exact_match": "#1B7837",      # dark green
    "category_match": "#7FBC41",   # mid green
    "chapter_match": "#D9F0A3",    # light green
    "out_of_domain": "#BDBDBD",    # neutral grey
    "no_prediction": "#6E94C4",    # blue-grey (abstention, distinct from error)
    "hallucination": "#C0392B",    # red (deployment-critical failure)
}

# Hatching gives a second visual cue beyond color so readers with
# color-vision deficits can still tell category_match and chapter_match
# (the two lighter greens) apart from exact_match.
BUCKET_HATCH = {
    "exact_match": "",
    "category_match": "//",
    "chapter_match": "xx",
    "out_of_domain": "",
    "no_prediction": "",
    "hallucination": "",
}

PRETTY = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5.5": "GPT-5.5",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash Preview",
    "baseline:fuzzy": "baseline: fuzzy",
    "baseline:tfidf": "baseline: TF-IDF",
    "sentence-transformer:retrieval": "sentence-transformer (retrieval)",
}

MODE_LABELS = {"zeroshot": "zero-shot", "constrained": "constrained", "rag": "RAG"}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[df["mode"] == "D4_abbreviated"].copy()
    df = df[~df["model"].isin(EXCLUDE_MODELS)]
    # Split LLM model:mode; for non-LLM baselines leave as-is.
    is_llm = df["model"].str.contains(":") & ~df["model"].str.startswith("baseline")
    is_llm &= df["model"] != "sentence-transformer:retrieval"
    parts = df["model"].str.split(":", n=1, expand=True)
    df["model_id"] = parts[0]
    df["prompting_mode"] = parts[1].where(is_llm, "")
    # Build a display label.
    def label(row: pd.Series) -> str:
        if row["prompting_mode"]:
            return f"{PRETTY.get(row['model_id'], row['model_id'])} ({MODE_LABELS.get(row['prompting_mode'], row['prompting_mode'])})"
        return PRETTY.get(row["model"], row["model"])
    df["label"] = df.apply(label, axis=1)
    df = df.sort_values("exact_match_pct", ascending=True)  # ascending = bottom-up plot
    return df


def render() -> None:
    df = load()
    n_rows = len(df)

    fig, ax = plt.subplots(figsize=(7.0, max(5.5, 0.28 * n_rows)))

    y_positions = np.arange(n_rows)
    cumulative = np.zeros(n_rows)

    for bucket in BUCKET_ORDER:
        widths = (df[f"{bucket}_pct"].to_numpy()) * 100
        ax.barh(
            y_positions,
            widths,
            left=cumulative,
            color=BUCKET_COLORS[bucket],
            edgecolor="white",
            linewidth=0.4,
            hatch=BUCKET_HATCH[bucket],
            label=BUCKET_LABELS[bucket],
        )
        cumulative += widths

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["label"].tolist(), fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_xlabel("Outcome share at D4 abbreviation stress", fontsize=9)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#aaa", alpha=0.6)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    render()
