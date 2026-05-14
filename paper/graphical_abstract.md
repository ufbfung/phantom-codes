# Graphical Abstract

IEEE J-BHI submission requires both a Graphical Abstract (GA) image
and a short caption that publishes alongside it in the Table of Contents.
Both are uploaded as separate files in the submission portal.

## Image

**File:** [`figures/graphical_abstract.jpg`](figures/graphical_abstract.jpg)
**Dimensions:** 660 × 295 px (per IEEE J-BHI spec)
**Format:** JPEG, quality 95
**Size:** ~37 KB (cap is 45 KB)
**Generator:** [`figures/graphical_abstract.py`](figures/graphical_abstract.py)

The GA reuses Figure 2 from the manuscript (cost-per-correct vs.
top-1 accuracy, with Pareto-optimal configurations highlighted by
heavy black borders). The right-margin callouts from the in-paper
figure are dropped; inline labels next to the three Pareto-optimal
points carry the same information at the smaller GA dimensions.

## Caption (≤ 50 words, 1–2 sentences per spec)

> Across 24 frontier-LLM configurations for medical concept
> normalization, cost-per-correct varied 270×: the deployment
> frontier consists of GPT-4o-mini constrained ($0.30 per 1,000
> correct) and Claude Haiku 4.5 constrained ($4.40), while the
> largest model tested is dominated. Constrained prompting
> eliminated fabrication for every Anthropic and OpenAI model.

Word count: 46 words. Two sentences. Carries both headline findings
(the cost-vs-accuracy frontier + the constrained-prompting
fabrication elimination) and matches what the GA image shows.
