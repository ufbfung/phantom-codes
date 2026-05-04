# Paper writing + NEJM AI submission scope

> **Status (2026-05-03):** Plan drafted and approved. Execution
> deferred until Brian's headline evaluation run completes. After
> headline run finishes, revisit this file to discuss next steps
> before kicking off Phase 1.
>
> **This is a temporary planning document.** Once the paper-writing
> scope kicks off, phase progress moves into BACKLOG.md (or its own
> dedicated tracker) and this file can be deleted.

---

## Context

Phantom Codes' v1 paper scaffolding is in good shape — IMRaD layout,
structured abstract, biblatex setup, Vancouver-style citations, 35
annotated bibliography entries, and ~9,200 words across six section
files. But the manuscript can't be submitted as-is. Several gaps
require dedicated focused work that explicitly does NOT belong in the
Synthea-eval scope (which already absorbed too much scope creep):

1. **Numbers fill-in** — every `[TBD]` marker in §3 Results, §4
   Discussion, §5 Conclusions, the abstract Results+Conclusions
   blocks, and the cost-economics tables waits on the headline run
   completing.
2. **Figures and tables** — `paper/figures/` and `paper/tables/` are
   empty. There's no figure-generation code anywhere in the repo.
   NEJM AI Original Articles typically run with 4-6 figures.
3. **Word-count overage** — current 9,235 words exceeds NEJM AI's
   5,000-8,000 main-text budget by 1,200-4,200 words.
   `02_cost_economics.md` alone is 3,268 words and is the natural
   candidate for moving the bulk of its content to supplementary.
4. **No supplementary materials structure** — no `supplementary.tex`,
   no place for full prompt templates, MI-CLAIM checklist, extended
   results, or the reproducibility appendix.
5. **No reporting-checklist compliance** — NEJM AI strongly
   recommends MI-CLAIM (Minimum Information about Clinical AI
   Modeling) for AI-method papers.
6. **No Statistical Analysis subsection** in Methods — Wilson 95%
   CIs are computed in `report.py` but the methodology section
   never declares this.
7. **No LLM-use disclosure** — current submission norms (incl. NEJM
   AI) require a statement of any AI assistance in manuscript
   preparation. Phantom Codes has been developed with Claude Code;
   this needs explicit disclosure.
8. **No cover letter, plain-language summary, suggested-reviewer
   list, ORCID setup, or arXiv-prep artifacts**.
9. **No CHANGELOG.md** to formally lock the methodology with a
   timestamp once the headline run completes.

The intended outcome of this scope: a complete NEJM AI submission
package — main manuscript PDF (≤8,000 words, lineno + double-spacing
on, all figures/tables in place, no `[TBD]` markers, MI-CLAIM
checklist as supplementary, statistical-analysis declaration in
Methods, LLM-use disclosure, cover letter, plain-language summary,
ORCID-linked author block — plus a same-week arXiv preprint posting.

## Decisions confirmed (2026-05-03)

Settled before this scope kicks off, so phases run without re-litigation:

- **Word-count strategy**: move ~70% of `02_cost_economics.md` to
  supplementary (S5). Keep one paragraph framing + headline
  cost-per-correct table in main. Drives Phase 3.
- **MI-CLAIM scope**: full 16-item checklist as supplementary
  appendix (S4). Drives Phase 5, item 2.
- **arXiv timing**: same week as NEJM AI submission. Drives Phase 8,
  item 3.
- **Statistical analysis**: 95% Wilson CIs only across the matrix
  (no NHST), with McNemar as a single secondary paired test for the
  within-model zero-shot vs constrained contrast. Drives Phase 5,
  item 1.

## Out of scope for this initiative

- Synthea pipeline expansion (done; benchmark generated)
- Headline evaluation run itself (Brian executing; gates Phase 1)
- HF Datasets + Zenodo cohort release (post-paper, BACKLOG stretch)
- v2 (LOINC) and v3 (RxNorm) extensions
- Cross-encoder reranker / fine-tuned small LLM (P2.5 deferred)
- v2 sensitivity analyses requiring extra figure code beyond the
  v1 headline figures
- Fallback-venue switch logistics (reactive after a hypothetical
  reject; keep the venue-swap path in `paper/README.md`)

## Current state (snapshot, 2026-05-03)

| Asset | Status |
|---|---|
| `paper/sections/00_introduction.md` | Drafted (1,529 words) |
| `paper/sections/01_methodology.md` | Drafted (2,380 words) — needs Statistical Analysis subsection added |
| `paper/sections/02_cost_economics.md` | Drafted (3,268 words) — most content moves to supplementary |
| `paper/sections/03_results.md` | Scaffolded, all numbers `[TBD]` |
| `paper/sections/04_discussion.md` | Scaffolded, ~5 `[TBD]` paragraphs |
| `paper/sections/05_conclusion.md` | Scaffolded, 161 words, both paragraphs `[TBD]` |
| Abstract Results+Conclusions blocks | `[TBD]` in `paper/main.tex` |
| `paper/figures/` | Empty |
| `paper/tables/` | Empty |
| `paper/supplementary.tex` | Does not exist |
| `paper/cover_letter.md` | Does not exist |
| `paper/references.bib` | 35 entries; 4 currently orphaned (Gururangan, Hendrycks, Singhal, Williams) |
| `paper/main.tex` | Complete with back-matter blocks; lineno+double-spacing commented for review submission |
| `CHANGELOG.md` | Does not exist |
| Word count main text | 9,235 (over budget) |
| Build pipeline | `make pdf` / `make snapshot` working; pandoc → xelatex → biber |

## Phases (execution order)

Phases 1, 2, and 3 can run somewhat in parallel after the headline
run completes; phases 4-8 are roughly sequential.

### Phase 1 — Headline-numbers integration (headline run completed 2026-05-04)

Drop the report tables into the paper. ~3-4 hours.

**Cohort actually evaluated**: 125 unique Synthea Conditions × 4
degradation modes × 29 model-entry configurations. Smaller than the
originally-targeted 500-record cap because of a `--max-records`
semantic ambiguity (it counts cohort rows, not unique resources;
see BACKLOG.md §P4 for the fix-it followup). Per-cell N is therefore
**125, not 500** — Wilson 95% CI half-widths are roughly **±5.4% on
a 10% rate, ±9% on a 50% rate** (vs ±2.7% / ±4.4% at n=500). This
is publishable but the §Methods cohort-size paragraph and §Results
table headers should be honest about it. Replication at the
originally-intended n=500 is straightforward (`--max-records 2000`)
and goes in §Limitations as a v2 path.

**Source CSV**: `results/raw/headline_n125_20260504T040931Z.csv`
(renamed from `headline_20260504T040931Z.csv` to make the cohort
size explicit in the filename and prevent accidental glob-pickup of
future larger runs).

1. Run `phantom-codes report --csv results/raw/headline_n125_20260504T040931Z.csv
   --pricing configs/pricing.yaml --out results/summary/n125_run/`.
2. Replace every `[TBD]` in `paper/sections/03_results.md` with the
   corresponding value from the report tables. Affected locations:
   - Cohort top-50 coverage percentage (line ~25)
   - Per-mode classifier accuracy table (lines 47-50)
   - Hallucination-rate table by model × mode (line 83) — include
     Wilson 95% CIs in each cell, not just point estimates
   - Top-1 accuracy table (line 92) — same CI treatment
   - Cost-per-correct table (lines 96-99)
   - 5-way outcome distribution under D4 (line 118)
3. Replace `[TBD]` rows in `paper/sections/02_cost_economics.md`
   per-model cost table (lines 332-340) with values from
   `headline.json` cost totals.
4. Fill `paper/sections/01_methodology.md` line 284 (validation
   top-1 accuracy at convergence) AND add a sentence to the cohort
   paragraph noting "evaluated on 125 unique Synthea-generated FHIR
   Conditions across all four degradation modes."
5. **Note in §Methods + §Limitations**: Gemini 3.1 Pro Preview was
   excluded from headline_set entirely (Tier 1 quota constraints); a
   separate ~21-record archived partial-coverage run (see BACKLOG
   "Stretch / future") is referenced for any S6 supplementary
   exploration. Gemini 2.5 Pro had ~88-90% per-mode coverage in the
   headline run (Tier 1 RPD throttling at high call volume); this
   should be footnoted in the per-cell tables for that model.
6. Verify all numbers reconcile across §1, §2, §3 (same model, same
   mode, same number).

**Files modified:** `paper/sections/01_methodology.md`,
`paper/sections/02_cost_economics.md`,
`paper/sections/03_results.md`.

### Phase 2 — Figure generation infrastructure (NEW code)

Currently zero figures and no figure-generation code. Add a small
matplotlib + seaborn module + CLI command. ~5-7 hours.

1. New file `src/phantom_codes/eval/figures.py` with these functions:
   - `plot_outcome_distribution(df, mode='D4_abbreviated', out_path)` —
     5-way stacked bar, models on y-axis, outcome buckets as colored
     segments. The headline figure.
   - `plot_hallucination_by_mode(df, out_path)` — line plot,
     models as series, modes (D1→D4) as x-axis, hallucination rate
     as y-axis. Shows the "abbreviation stress reveals what other
     modes don't" pattern.
   - `plot_top_k_lift(df, out_path)` — grouped bar, top-1 vs top-5
     exact-match rate per model.
   - `plot_cost_vs_accuracy(df, pricing, out_path)` — 2D scatter,
     cost-per-correct on x, accuracy on y, one point per
     (model, mode) configuration. Annotated with model names.
2. New CLI command `phantom-codes figures`:
   ```
   uv run phantom-codes figures \
     --csv results/raw/headline_<run>.csv \
     --pricing configs/pricing.yaml \
     --out paper/figures/
   ```
   Writes 4 PDFs at 300 DPI (print-ready).
3. Wire `\includegraphics` calls into `paper/sections/03_results.md`
   via Pandoc-compatible `![caption](../figures/foo.pdf)` markdown
   syntax.
4. Captions follow NEJM AI convention: "Figure N. <one-sentence
   description>. <2-3 sentence expansion of what reader should
   notice.>"
5. New tests `tests/test_figures.py` — assert each function
   produces a non-empty PDF given a fixture CSV. ~4 unit tests.

**Files created:** `src/phantom_codes/eval/figures.py`,
`tests/test_figures.py`,
`paper/figures/{outcome,hallucination,topk,cost}.pdf`.
**Files modified:** `src/phantom_codes/cli.py` (add `figures`
command), `paper/sections/03_results.md` (insert figure refs).

### Phase 3 — Word-count compliance + supplementary materials structure

Cut from 9,235 to ≤7,500 words main text (leaves headroom for
reviewer-requested additions). ~3-5 hours.

1. Create `paper/supplementary.tex` — separate document, same
   biblatex backbone, same article class, header reading
   "Supplementary Materials for Phantom Codes …".
2. Create `paper/supp_sections/` with these section markdown files:
   - `S1_prompt_templates.md` — verbatim system + user prompts for
     zero-shot, constrained, RAG; per-provider tool-use schemas.
   - `S2_extended_results.md` — per-disease-group breakdowns,
     per-vocabulary-position breakdowns, full hallucinated-code
     listing with frequency.
   - `S3_reproducibility.md` — exact CLI sequence to reproduce;
     `uv export` snapshot; Synthea pinned SHA; configs SHA.
   - `S4_miclaim.md` — MI-CLAIM checklist (created in Phase 5).
   - `S5_cost_economics_extended.md` — receives ~70% of current
     `02_cost_economics.md` content.
3. Move from `paper/sections/02_cost_economics.md` to
   `S5_cost_economics_extended.md`:
   - Full break-even analysis with sensitivity surfaces
   - Pricing methodology and snapshot date discussion
   - Anthropic prompt-caching threshold deep-dive
   - All but one headline cost-per-correct table

   Keep in main text:
   - One paragraph framing cost-per-correct as a deployment metric
   - The headline cost-per-correct table (one table)
   - One paragraph break-even summary referencing the supplementary
     analysis
4. Trim `paper/sections/01_methodology.md` for redundancy with §0
   Introduction (target ~10-15% reduction).
5. Update `paper/Makefile` to add `make supp` target producing
   `paper/build/supplementary.pdf`.
6. Re-measure: `wc -w paper/sections/*.md`. Target ≤ 7,500 words.

**Files created:** `paper/supplementary.tex`,
`paper/supp_sections/{S1..S5}_*.md`.
**Files modified:** `paper/sections/01_methodology.md`,
`paper/sections/02_cost_economics.md`, `paper/Makefile`.

### Phase 4 — §4 Discussion + §5 Conclusions + Abstract

Write from the §3 numbers Phase 1 produced. ~4-6 hours.

1. `04_discussion.md`:
   - Principal findings paragraph (line 13) — restate headline
     numbers in prose, three findings in priority order.
   - Comparison with prior work (line 58) — quantify our
     hallucination band against Soroush 2024 (46/34/50% on
     ICD-9/10/CPT) and Hybrid-Code v2 neural-baseline 6-18%.
   - Why D4 reveals what other modes don't (line 65) — fill
     interpretive paragraph.
   - Implications for clinical deployment (line 91) — fill
     paragraph referencing §Cost economics break-even (now in
     supplementary).
2. `05_conclusion.md` — rewrite both paragraphs to ~150 words total.
3. `paper/main.tex`:
   - Abstract Results block (line 144) — fill with headline numbers.
   - Abstract Conclusions block (line 150) — three sentences.
   - Verify combined abstract ≤ 300 words (NEJM AI convention).

**Files modified:** `paper/sections/04_discussion.md`,
`paper/sections/05_conclusion.md`, `paper/main.tex`.

### Phase 5 — Reporting checklist + statistical analysis + disclosures

Compliance and methodology-statement work. ~4-5 hours.

1. Add **Statistical analysis** subsection to `01_methodology.md`:
   - 95% Wilson confidence intervals for all proportions (already
     computed in `eval/report.py`)
   - McNemar's test for paired within-model
     (zero-shot vs constrained) comparisons — describe; add to
     `report.py` if not already there
   - Decision on multiple-comparison correction: declare we report
     95% CIs and decline NHST across the 36-cell matrix (cleaner
     and increasingly preferred in ML reporting); cite Bonferroni
     in passing as the alternative we considered
2. Populate `paper/supp_sections/S4_miclaim.md` — MI-CLAIM checklist
   (16 items; populate each from current scaffolding). Items
   include: study purpose, data sources, cohort selection, ground
   truth, models compared, evaluation procedure, performance
   metrics, statistical analysis, error analysis, validation
   strategy, deployment considerations.
3. Add **LLM-use disclosure** to `paper/main.tex` back-matter
   (between `\section*{Acknowledgments}` and bibliography):
   ```
   \section*{Use of large language models in manuscript preparation}
   This manuscript was prepared with assistance from a coding/writing
   assistant (Claude Code by Anthropic). All experimental design,
   scientific claims, statistical analyses, and final wording are
   the corresponding author's responsibility.
   ```
4. Note **lack of pre-registration** under §Limitations (we have
   internal methodology-discipline guardrail in CLAUDE.md but no
   public OSF / AsPredicted preregistration).

**Files modified:** `paper/sections/01_methodology.md`,
`paper/sections/04_discussion.md` (Limitations subsection),
`paper/main.tex`, `paper/supp_sections/S4_miclaim.md`,
`src/phantom_codes/eval/report.py` (if McNemar isn't there).

### Phase 6 — Manuscript polish + final compile

Submission-ready formatting. ~2-3 hours.

1. Uncomment in `paper/main.tex`:
   - `\usepackage{lineno}` + `\linenumbers`
   - `\usepackage{setspace}` + `\doublespacing`
2. Spell-check + grammar pass on rendered PDF (markdown view
   sometimes hides things visible only in LaTeX).
3. Verify Vancouver-style citation rendering side-by-side with a
   recent NEJM AI published article (sample any 2025-2026 Original
   Article from NEJM AI).
4. Audit orphan citations: 4 originally-unused entries (Gururangan,
   Hendrycks, Singhal, Williams) — either cite or strip. Plus 4 newly-
   added provider-doc entries (`OpenAIModels2026`, `OpenAIPricing2026`,
   `GeminiModels2026`, `GeminiPricing2026`) — these need to be wired
   into the prose to avoid orphan status. Recommended placement:
   - §1 Methods, where the LLM lineup is introduced: cite
     `[@OpenAIModels2026; @GeminiModels2026]` next to the first
     mention of model_id strings to ground reproducibility.
   - §2 Cost economics, where per-token rates first appear: cite
     `[@OpenAIPricing2026; @GeminiPricing2026]` with the pricing
     snapshot table; once Anthropic pricing is verified
     (BACKLOG cost-infra item), add `[@AnthropicPricing2026]` here too.
5. Expand any `and others` author lists in `references.bib` to
   full lists (NEJM AI may want this).
6. Final compile: `make clean && make pdf`. Verify zero warnings,
   zero `[TBD]` markers, zero orphan citations.
7. `make snapshot` → promote `build/paper.pdf` to `paper/paper.pdf`
   for committing.

**Files modified:** `paper/main.tex`, `paper/references.bib`,
`paper/paper.pdf` (snapshot).

### Phase 7 — Submission package preparation

Logistics + artifacts. ~3-4 hours code + writing + external setup.

1. **Cover letter** (`paper/cover_letter.md`, 1 page):
   - Novelty: 5-way taxonomy + D4 stress test + within-model ablation
   - Reproducibility story: compliance-by-design Synthea evaluation
   - Explicit non-overlap statement vs. Soroush 2024
   - Why this paper fits NEJM AI (clinical-deployment audience,
     AI-methods rigor)
2. **Plain-language summary** (NEJM AI Original Articles require
   this; ≤150 words, lay-friendly): one paragraph for a
   generally-educated medical audience without ML background.
   Stored in `paper/plain_language_summary.md` and as a separate
   section in `main.tex` if NEJM AI's submission portal needs it
   inline.
3. **Suggested reviewers list** (`paper/suggested_reviewers.md`,
   3-5 names): Soroush, Almeida, Hatem, Motzfeldt, Kim. Each entry:
   name, affiliation, email, ORCID, one-sentence justification.
4. **Decline-reviewers list** (`paper/decline_reviewers.md`): any
   COIs (none expected for v1).
5. **ORCID registration** (external, ~10 min via orcid.org).
   Update author block in `main.tex`.
6. **Affiliation finalization** — currently "Independent
   Researcher"; confirm before submission.

**Files created:** `paper/cover_letter.md`,
`paper/plain_language_summary.md`,
`paper/suggested_reviewers.md`, `paper/decline_reviewers.md`.
**External actions:** ORCID registration.

### Phase 8 — Submission + arXiv preprint + CHANGELOG methodology lock

Submit + freeze. ~2-3 hours active + asynchronous follow-up.

1. **Create `CHANGELOG.md`** at repo root. Document the locked
   methodology with timestamp:
   - Locked degradation modes (D1/D2/D3/D4 definitions)
   - Locked abbreviation table (`abbreviations.yaml` SHA at lock)
   - Locked candidate-list construction (CMS + observed)
   - Locked 5-way outcome taxonomy
   - Locked scope filter
   - Locked headline LLM set (`configs/models.yaml#headline_set`
     SHA at lock)
   - Lock date = NEJM AI submission date
2. **NEJM AI submission**:
   - Set up corresponding-author account at NEJM AI submission
     portal
   - Confirm ORCID linked
   - Upload: main manuscript PDF, supplementary PDF, cover letter,
     plain-language summary, suggested + decline reviewer lists
   - Submit
3. **arXiv preprint** (within same week):
   - Verify NEJM AI's preprint policy at submission time (currently
     preprint-friendly per their published author guidance)
   - Line up endorser for `cs.CL` (former Mayo / Verily / ONC
     colleague with prior arXiv submissions)
   - Submit same content; categories `cs.CL` (primary) + `cs.LG`,
     `stat.AP`, `q-bio.QM` (cross)
   - License: CC BY 4.0
4. Track NEJM AI status; respond to revision requests.

**Files created:** `CHANGELOG.md`.
**External actions:** NEJM AI portal account + submission, arXiv
endorsement + submission.

## Files to create / modify (summary)

### Created
- `src/phantom_codes/eval/figures.py` (Phase 2)
- `tests/test_figures.py` (Phase 2)
- `paper/figures/{outcome,hallucination,topk,cost}.pdf` (Phase 2)
- `paper/supplementary.tex` (Phase 3)
- `paper/supp_sections/S1_prompt_templates.md` (Phase 3)
- `paper/supp_sections/S2_extended_results.md` (Phase 3)
- `paper/supp_sections/S3_reproducibility.md` (Phase 3)
- `paper/supp_sections/S4_miclaim.md` (Phase 5)
- `paper/supp_sections/S5_cost_economics_extended.md` (Phase 3)
- `paper/cover_letter.md` (Phase 7)
- `paper/plain_language_summary.md` (Phase 7)
- `paper/suggested_reviewers.md` (Phase 7)
- `paper/decline_reviewers.md` (Phase 7)
- `CHANGELOG.md` (Phase 8)

### Modified
- `src/phantom_codes/cli.py` (Phase 2 — add `figures` command)
- `src/phantom_codes/eval/report.py` (Phase 5 — McNemar if missing)
- `paper/sections/01_methodology.md` (Phases 1, 3, 5)
- `paper/sections/02_cost_economics.md` (Phases 1, 3)
- `paper/sections/03_results.md` (Phases 1, 2)
- `paper/sections/04_discussion.md` (Phases 4, 5)
- `paper/sections/05_conclusion.md` (Phase 4)
- `paper/main.tex` (Phases 4, 5, 6, 7)
- `paper/references.bib` (Phase 6)
- `paper/Makefile` (Phase 3)
- `paper/paper.pdf` (Phase 6 — snapshot)
- `BACKLOG.md` (per-phase progress)

## Verification

After each phase:

- **Phase 1**: `grep -rn '\[TBD' paper/sections/` shows zero
  matches in §3, §2 cost table; numbers reconcile across sections.
- **Phase 2**: `uv run pytest tests/test_figures.py -q` green;
  `uv run phantom-codes figures` produces 4 valid PDFs in
  `paper/figures/`; `make pdf` includes them in §3.
- **Phase 3**: `wc -w paper/sections/*.md` ≤ 7,500;
  `make supp` produces `paper/build/supplementary.pdf` cleanly.
- **Phase 4**: `grep -rn '\[TBD' paper/sections/ paper/main.tex`
  shows zero matches; abstract `wc -w` ≤ 300.
- **Phase 5**: MI-CLAIM checklist all 16 items populated; Methods
  contains a Statistical Analysis subsection; LLM-use disclosure
  block compiles in back-matter.
- **Phase 6**: `make clean && make pdf` exits zero with no warnings;
  rendered PDF has line numbers + double spacing on; no `[TBD]`
  remains anywhere.
- **Phase 7**: cover letter ≤ 1 page; plain-language summary
  ≤ 150 words; ORCID confirmed in author block.
- **Phase 8**: `CHANGELOG.md` committed with methodology-lock date;
  NEJM AI submission ID received; arXiv preprint URL received.

End-to-end final check:
```
make clean && make pdf && make supp
wc -w paper/sections/*.md       # ≤ 7500 main text
grep -rn '\[TBD' paper/          # zero matches
uv run ruff check src/ tests/    # clean
uv run pytest tests/ -q          # all green
```

## Estimated effort

| Phase | Implementation effort |
|---|---|
| Phase 1 — Headline numbers | 3-4 hours |
| Phase 2 — Figure infrastructure | 5-7 hours |
| Phase 3 — Word-count + supplementary | 3-5 hours |
| Phase 4 — Discussion + Conclusions | 4-6 hours |
| Phase 5 — Reporting checklist + stats + disclosures | 4-5 hours |
| Phase 6 — Polish + final compile | 2-3 hours |
| Phase 7 — Submission package | 3-4 hours + ORCID setup |
| Phase 8 — Submission + arXiv + CHANGELOG | 2-3 hours active |
| **Total** | **~26-37 hours active work** |

External-dependency wait: ORCID setup (~10 min), arXiv endorsement
(could take days depending on endorser response).
