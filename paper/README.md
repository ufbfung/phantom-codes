# Phantom Codes — paper

This directory contains the *Phantom Codes: Hallucination in
LLM-Based Clinical Concept Normalization* manuscript and its
companion documents. Three rendered PDFs are committed at the repo
root for casual reading without a local LaTeX install:

- [`paper.pdf`](paper.pdf) — main manuscript (target venue:
  **JAMIA Research and Applications**)
- [`supplementary.pdf`](supplementary.pdf) — supplementary
  materials (extended results, prompt templates, MI-CLAIM
  checklist, extended cost economics)
- [`arxiv.pdf`](arxiv.pdf) — companion technical report on the
  PubMedBERT trained-classifier baseline (planned arXiv preprint)

For project context see the top-level [README](../README.md); for
reproducing the headline numbers see
[`BENCHMARK.md`](../BENCHMARK.md). Read on if you want to rebuild
the PDFs from source.

## Status

Headline n=125 Synthea evaluation completed 2026-05-04; all six
main-text sections are drafted with realized numbers (no `[TBD]`
markers). Main text fits within JAMIA's 4,000-word limit (~3,830
words); abstract is 251 prose words (≤250 cap, within rounding).

| Section | State |
|---|---|
| Introduction | ✅ drafted (529 words) |
| Materials and Methods | ✅ drafted (805 words) |
| Cost & deployment economics | ✅ drafted (329 words) |
| Results | ✅ drafted (1,328 words; 4 tables) |
| Discussion | ✅ drafted (680 words) |
| Conclusion | ✅ drafted (156 words) |

Submission-readiness tasks are tracked in the project
[BACKLOG](../BACKLOG.md) under §P5.

## Layout

```
paper/
  paper.pdf            # JAMIA main manuscript snapshot
  supplementary.pdf    # JAMIA supplement snapshot
  arxiv.pdf            # arXiv tech report snapshot
  main.tex             # JAMIA main LaTeX master
  supplementary.tex    # JAMIA supplement master
  arxiv.tex            # arXiv tech report master
  references.bib       # shared bibliography (single source of truth)
  Makefile             # build automation (pdf / supp / arxiv / snapshot-*)
  README.md            # this file
  sections/            # JAMIA main prose (markdown canonical)
    00_introduction.md … 05_conclusion.md
  supp_sections/       # JAMIA supplement prose
    S1_prompt_templates.md … S5_cost_economics_extended.md
                       # (S3 lifted into arxiv_sections/)
  arxiv_sections/      # arXiv tech report prose
    00_introduction.md … 06_appendix.md
  build/               # generated; gitignored
  figures/             # TBD
  tables/              # TBD
```

## Editing model

- **Prose edits** happen in `sections/*.md`, `supp_sections/*.md`,
  or `arxiv_sections/*.md`. Markdown is canonical; LaTeX is
  generated.
- **Citations** use Pandoc `[@Key]` syntax (e.g., `[@Soroush2024]`;
  multiple: `[@Kim2025; @Hatem2025]`). Pandoc converts to
  `\autocite{}`; biblatex resolves at build time against the
  shared `references.bib`.
- **Bibliography**: append a BibTeX entry to `references.bib` with
  a `% Why we cite: …` comment block above it. Cite from prose with
  `[@NewKey]`; uncited entries do not appear in the rendered
  bibliography (biblatex default).
- **Document structure** (sections, packages, title, abstract)
  lives in `main.tex` / `supplementary.tex` / `arxiv.tex`. Edit
  these only when adding/removing/reordering sections.
- **Adding a section**: create the `*.md` file and add a matching
  `\input{build/<dir>/NN_name}` line in the corresponding `.tex`
  master.

## Building the PDFs locally

### One-time setup

```bash
make deps           # prints install hints for pandoc + LaTeX + biber
```

On macOS (typical):

```bash
brew install pandoc
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended \
                   collection-latexrecommended \
                   framed parskip xurl csquotes microtype \
                   biblatex biber logreq biblatex-vancouver
```

### Build commands

```bash
cd paper

make pdf            # JAMIA main → build/paper.pdf
make supp           # JAMIA supplement → build/supplementary.pdf
make arxiv          # arXiv tech report → build/arxiv.pdf

make snapshot       # promote build/paper.pdf → paper/paper.pdf
make snapshot-supp  # promote build/supplementary.pdf → paper/supplementary.pdf
make snapshot-arxiv # promote build/arxiv.pdf → paper/arxiv.pdf
make snapshot-all   # all three at once

make clean          # wipe build/
make watch          # rebuild on any source change (requires `brew install entr`)
```

The committed `*.pdf` files in this directory are
manually-promoted snapshots rather than every build, so commit
history stays clean. Run `make snapshot-all` after a meaningful
content change worth a fresh public PDF.

### Build pipeline

`make pdf` (and the `supp` / `arxiv` siblings) runs four phases:

1. **Pandoc** converts each `*.md` to `build/<dir>/*.tex`,
   preserving headings, tables (booktabs), code blocks (verbatim),
   and links (hyperref). The `--biblatex` flag emits
   `\autocite{Key}` for `[@Key]` callouts.
2. **xelatex pass 1** — collects citations into the auxiliary
   files.
3. **biber** — resolves citations against `references.bib`.
4. **xelatex passes 2 and 3** — insert the bibliography and
   resolve cross-references.

## Switching citation style

Single-line change in the relevant `.tex` master. Find the biblatex
`\usepackage` line and edit `style=`:

```latex
% vancouver [1] superscripted — biomedical (JAMIA, NEJM AI, BMJ) ← current
\usepackage[backend=biber, style=vancouver, sortlocale=en_US]{biblatex}

% numeric-comp [1, 2] — IEEE, Nature, NeurIPS
\usepackage[backend=biber, style=numeric-comp, sortlocale=en_US]{biblatex}

% authoryear (Devlin, 2019) — easier-to-skim drafting style
\usepackage[backend=biber, style=authoryear, sortlocale=en_US]{biblatex}
```

Then `make clean && make <pdf|supp|arxiv>`. No changes to the
`.bib` or to the markdown citations — biblatex re-renders
everything in the new style.

## Reusing this scaffolding

The files in this directory (`main.tex`, `supplementary.tex`,
`arxiv.tex`, `Makefile`, `references.bib` as a template,
`.gitignore`) are MIT-licensed along with the rest of the
repository. Drop them into a fresh `paper/` directory, swap in
your own prose and citations, and you should be running with
`make pdf` in a few minutes (assuming pandoc + a TeX distribution
are installed).
