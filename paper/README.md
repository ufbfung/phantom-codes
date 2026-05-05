# Phantom Codes — paper

Two papers from this project live here, each in its own subdirectory:

| Snapshot PDF | Sources | What it is |
|---|---|---|
| [`phantom_codes.pdf`](phantom_codes.pdf) | [`phantom_codes/`](phantom_codes/) | Main manuscript (target venue: **JAMIA Research and Applications**) — *Phantom Codes: Hallucination in LLM-Based Clinical Concept Normalization* |
| [`phantom_codes_supplementary.pdf`](phantom_codes_supplementary.pdf) | [`phantom_codes/supp_sections/`](phantom_codes/supp_sections/) | Supplement to the main manuscript (extended results, prompt templates, MI-CLAIM checklist, extended cost economics) |
| [`pubmedbert_finetuning.pdf`](pubmedbert_finetuning.pdf) | [`pubmedbert/`](pubmedbert/) | Companion technical report (planned arXiv preprint) — *Local Fine-Tuning of PubMedBERT for ACCESS-Scope ICD-10-CM Classification under PhysioNet Compliance Constraints* |

Both papers share [`references.bib`](references.bib) via biblatex.

For project context see the top-level [README](../README.md); for
reproducing the headline numbers see
[`BENCHMARK.md`](../BENCHMARK.md). Read on if you want to rebuild
the PDFs from source.

## Status

Headline n=125 Synthea evaluation completed 2026-05-04; all six
main-text sections of the JAMIA paper are drafted with realized
numbers (no `[TBD]` markers). Main text fits within JAMIA's
4,000-word limit (~3,830 words); abstract is 251 prose words
(≤250 cap, within rounding). The PubMedBERT tech report has 7
sections (~3,220 words) covering data + compliance, architecture,
hardware, optimization, results, discussion, and a reproduction
appendix.

| Document | Sections | State |
|---|---|---|
| Phantom Codes (JAMIA) | §0 Introduction → §5 Conclusion | ✅ drafted |
| Phantom Codes supplement | S1, S2, S4, S5 | ✅ scaffolded with realized numbers in S5 |
| PubMedBERT tech report | §0 Introduction → §6 Appendix | ✅ drafted |

Submission-readiness tasks are tracked in the project
[BACKLOG](../BACKLOG.md) under §P5.

## Layout

```
paper/
  README.md, Makefile, references.bib   (shared infrastructure)
  .gitignore                             (build/ subdirs are gitignored)

  phantom_codes.pdf                      (main paper snapshot)
  phantom_codes_supplementary.pdf        (supplement snapshot)
  pubmedbert_finetuning.pdf              (tech report snapshot)

  phantom_codes/
    main.tex                             (main paper LaTeX master)
    supplementary.tex                    (supplement LaTeX master)
    sections/                            (00_introduction.md … 05_conclusion.md)
    supp_sections/                       (S1, S2, S4, S5)
    build/                               (gitignored)

  pubmedbert/
    main.tex                             (tech report LaTeX master)
    sections/                            (00_introduction.md … 06_appendix.md)
    build/                               (gitignored)

  figures/                               (TBD)
  tables/                                (TBD)
```

## Editing model

- **Prose edits** happen in markdown under
  `phantom_codes/sections/`, `phantom_codes/supp_sections/`, or
  `pubmedbert/sections/`. Markdown is canonical; LaTeX is generated.
- **Citations** use Pandoc `[@Key]` syntax (e.g., `[@Soroush2024]`;
  multiple: `[@Kim2025; @Hatem2025]`). Pandoc converts to
  `\autocite{}`; biblatex resolves at build time against the
  shared `references.bib` (each LaTeX master points at
  `\addbibresource{../references.bib}`).
- **Bibliography**: append a BibTeX entry to `references.bib`
  with a `% Why we cite: …` comment block above it. Cite from
  prose with `[@NewKey]`; uncited entries do not appear in the
  rendered bibliography (biblatex default).
- **Document structure** (sections, packages, title, abstract)
  lives in each LaTeX master (`phantom_codes/main.tex`,
  `phantom_codes/supplementary.tex`, `pubmedbert/main.tex`).
- **Adding a section**: create the `*.md` file under the
  appropriate `sections/` directory and add a matching
  `\input{build/sections/NN_name}` line in the LaTeX master.

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

make phantom_codes               # → phantom_codes/build/main.pdf
make phantom_codes_supp          # → phantom_codes/build/supplementary.pdf
make pubmedbert                  # → pubmedbert/build/main.pdf
make all                         # all three at once

make snapshot-phantom_codes      # promote → paper/phantom_codes.pdf
make snapshot-phantom_codes-supp # promote → paper/phantom_codes_supplementary.pdf
make snapshot-pubmedbert         # promote → paper/pubmedbert_finetuning.pdf
make snapshot-all                # all three at once

make clean                       # wipe both build/ subdirectories
make watch                       # rebuild on any source change (requires `brew install entr`)
```

The committed `*.pdf` files in this directory are
manually-promoted snapshots rather than every build, so commit
history stays clean. Run `make snapshot-all` after a meaningful
content change worth a fresh public PDF.

## Switching citation style

Single-line change in the relevant LaTeX master. Find the biblatex
`\usepackage` line and edit `style=`:

```latex
% vancouver [1] superscripted — biomedical (JAMIA, NEJM AI, BMJ) ← current
\usepackage[backend=biber, style=vancouver, sortlocale=en_US]{biblatex}

% numeric-comp [1, 2] — IEEE, Nature, NeurIPS
\usepackage[backend=biber, style=numeric-comp, sortlocale=en_US]{biblatex}

% authoryear (Devlin, 2019) — easier-to-skim drafting style
\usepackage[backend=biber, style=authoryear, sortlocale=en_US]{biblatex}
```

Then `make clean && make <target>`. No changes to the `.bib` or
to the markdown citations — biblatex re-renders everything in the
new style.

## Reusing this scaffolding

The files in this directory (LaTeX masters, `Makefile`,
`references.bib` as a template, `.gitignore`) are MIT-licensed
along with the rest of the repository. The two-paper subdirectory
layout is also a reusable pattern when a project produces a main
manuscript plus one or more companion technical reports — drop in
your own prose and citations, and you should be running with
`make all` in a few minutes (assuming pandoc + a TeX distribution
are installed).
