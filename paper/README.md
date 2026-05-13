# Phantom Codes — paper

Two papers from this project live here, each in its own subdirectory:

| Snapshot PDF | Sources | What it is |
|---|---|---|
| [`phantom_codes.pdf`](phantom_codes.pdf) | [`phantom_codes/`](phantom_codes/) | Main manuscript (target venue: **JAMIA Research and Applications**) — *Phantom Codes: Hallucination, Accuracy, and Cost in LLM-Based Medical Concept Normalization* |
| [`phantom_codes_supplementary.pdf`](phantom_codes_supplementary.pdf) | [`phantom_codes/supp_sections/`](phantom_codes/supp_sections/) | Supplement to the main manuscript (S1 prompt templates, S2 extended results, S5 extended cost economics) |
| [`pubmedbert_finetuning.pdf`](pubmedbert_finetuning.pdf) | [`pubmedbert/`](pubmedbert/) | Companion technical report (planned arXiv preprint) — *Local Fine-Tuning of PubMedBERT for ACCESS-Scope ICD-10-CM Classification under PhysioNet Compliance Constraints* |

Both papers share [`references.bib`](references.bib) via biblatex.

For project context see the top-level [README](../README.md); for
reproducing the headline numbers see
[`BENCHMARK.md`](../BENCHMARK.md). Read on if you want to rebuild
the PDFs from source.

## Status

Headline n=125 Synthea evaluation completed 2026-05-04; the JAMIA
manuscript is submission-ready. Main text fits JAMIA's 4,000-word
limit (~3,200 words); structured abstract is ~225 prose words
(≤250 cap). 4 tables and 3 figures (at JAMIA's 4-table / 6-figure
caps). The PubMedBERT tech report is a separate arXiv-bound
companion (~3,200 words) covering data + compliance, architecture,
hardware, optimization, results, discussion, and a reproduction
appendix.

| Document | Structure | State |
|---|---|---|
| Phantom Codes (JAMIA) | Title page → Abstract → §1 Background and Significance → §5 Conclusion → back-matter | ✅ submission-ready |
| Phantom Codes supplement | S1 prompts, S2 extended results, S5 extended cost economics | ✅ submission-ready |
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

  cover_letter.md                        (JAMIA submission cover letter)
  scripts/flatten_paper.py                (modular .tex → flat .tex)

  phantom_codes.tex                      (GENERATED — main, single self-contained .tex)
  phantom_codes_supplementary.tex        (GENERATED — supplement, single self-contained .tex)

  phantom_codes/
    main.tex                             (main paper LaTeX master)
    supplementary.tex                    (supplement LaTeX master)
    sections/                            (00_title_page, 01_abstract,
                                          02_background_and_significance,
                                          03_materials_and_methods,
                                          04_results, 05_discussion,
                                          06_conclusion)
    supp_sections/                       (S1_prompt_templates,
                                          S2_extended_results,
                                          S5_cost_economics_extended)
    build/                               (gitignored, xelatex/biber output)

  pubmedbert/
    main.tex                             (tech report LaTeX master)
    sections/                            (00_introduction.tex … 06_appendix.tex)
    build/                               (gitignored)

  figures/                               (figure1 heatmap, figure2 cost
                                          frontier, figure3 D4 outcome
                                          stack — .py source + .pdf each)
```

## Editing model

Pure LaTeX. xelatex + biber compile section sources directly; there
is no markdown / pandoc step.

> **Modular sources are canonical; the top-level
> `phantom_codes.tex` and `phantom_codes_supplementary.tex` are
> generated build artifacts** (auto-flattened from the modular
> sources by `make flatten`, chained into every `make snapshot-*`).
> They're committed for JAMIA's reference. The first 12 lines of
> each carry a `% GENERATED FILE — DO NOT EDIT` banner. Edit the
> small section files; the flat .tex regenerates on snapshot.

- **Prose edits** happen in `.tex` files under
  `phantom_codes/sections/`, `phantom_codes/supp_sections/`, or
  `pubmedbert/sections/`.
- **Citations** use `\autocite{Key}` (e.g., `\autocite{Soroush2024}`;
  multiple: `\autocite{Kim2025,Hatem2025}`). biblatex resolves at
  build time against the shared `references.bib` (each LaTeX master
  points at `\addbibresource{../references.bib}`).
- **Bibliography**: append a BibTeX entry to `references.bib`
  with a `% Why we cite: …` comment block above it. Cite from
  prose with `\autocite{NewKey}`; uncited entries do not appear in
  the rendered bibliography (biblatex default).
- **Document structure** (sections, packages, title, abstract)
  lives in each LaTeX master (`phantom_codes/main.tex`,
  `phantom_codes/supplementary.tex`, `pubmedbert/main.tex`).
- **Adding a section**: create `sections/NN_name.tex` (or
  `supp_sections/SN_name.tex`) and add a matching
  `\input{sections/NN_name}` line in the relevant LaTeX master.

## Building the PDFs locally

### One-time setup

```bash
make deps           # prints install hints for LaTeX + biber
```

On macOS (typical):

```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended \
                   collection-latexrecommended \
                   framed parskip xurl csquotes microtype \
                   biblatex biber logreq biblatex-vancouver \
                   stix2-otf
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
make test                        # build all three + sanity-check page counts

make flatten-all                 # generate phantom_codes.tex + phantom_codes_supplementary.tex
make verify-flat                 # confirm flat .tex renders identically to modular build
```

The committed `*.pdf` files in this directory are
manually-promoted snapshots rather than every build, so commit
history stays clean. Run `make snapshot-all` after a meaningful
content change worth a fresh public PDF.

## Switching citation style

Single-line change in the relevant LaTeX master. Find the biblatex
`\usepackage` line and edit `style=`:

```latex
% vancouver [1] superscripted — biomedical (JAMIA, BMJ) ← current
\usepackage[backend=biber, style=vancouver, sortlocale=en_US]{biblatex}

% numeric-comp [1, 2] — IEEE, Nature, NeurIPS
\usepackage[backend=biber, style=numeric-comp, sortlocale=en_US]{biblatex}

% authoryear (Devlin, 2019) — easier-to-skim drafting style
\usepackage[backend=biber, style=authoryear, sortlocale=en_US]{biblatex}
```

Then `make clean && make <target>`. No changes to the `.bib` or
to the `\autocite{...}` calls in section sources — biblatex
re-renders everything in the new style.

## Reusing this scaffolding

The files in this directory (LaTeX masters, `Makefile`,
`references.bib` as a template, `.gitignore`) are MIT-licensed
along with the rest of the repository. The two-paper subdirectory
layout is also a reusable pattern when a project produces a main
manuscript plus one or more companion technical reports — drop in
your own prose and citations, and you should be running with
`make all` in a few minutes (assuming a TeX distribution with
biblatex/biber and the STIX Two Text font is installed).
