# Phantom Codes — paper

This directory contains the in-progress manuscript for *Phantom Codes:
Hallucination in LLM-Based Clinical Concept Normalization*. The paper
is a working draft; the latest rendered PDF snapshot is committed at
[`paper.pdf`](paper.pdf) for casual reading without needing a local
LaTeX install.

The paper accompanies the open benchmark and reproducible code in this
repository's top-level [README](../README.md). If you're a researcher
wanting to evaluate your own model against the benchmark or reproduce
the paper's headline numbers, see [`BENCHMARK.md`](../BENCHMARK.md)
for the full step-by-step reproduction guide (no MIMIC credentialing
required). If you're here to read the current draft of the paper
itself, open `paper.pdf`. If you want to rebuild the PDF from source
— read on.

## Status

The manuscript is structured to NEJM AI's Original Article guidance as
a working target — IMRaD layout, structured abstract, Vancouver/AMA
citations. We may submit elsewhere as the work develops; switching to
a different venue is largely a citation-style swap and a section-order
re-arrange.

Sections currently in the draft, in order:

| Section | State |
|---|---|
| Introduction | ✅ drafted |
| Methods | ✅ drafted (data, cohort, architecture, hardware) |
| Cost & deployment economics | ✅ drafted (framework + extrapolations) |
| Results | 🟡 scaffolded; numbers fill in as headline runs complete |
| Discussion | 🟡 scaffolded; written from Results numbers |
| Conclusions | 🟡 scaffolded |

Submission-readiness tasks are tracked in the project [BACKLOG](../BACKLOG.md)
under §P5.

## Layout

```
paper/
  paper.pdf         # current rendered draft (commit when material updates land)
  main.tex          # LaTeX skeleton — documentclass, title, structured abstract,
                    #   biblatex setup, \input{} of generated sections, back-matter
  references.bib    # bibliography (BibTeX); single source of truth
  Makefile          # build automation
  README.md         # this file
  .gitignore        # ignore build/
  sections/         # canonical prose (edit here)
    00_introduction.md
    01_methodology.md
    02_cost_economics.md
    03_results.md
    04_discussion.md
    05_conclusion.md
  build/            # generated; gitignored. Final PDF lives here at build/paper.pdf
                    #   before being snapshotted to paper/paper.pdf.
```

## Editing model

- **Prose edits** happen in `sections/*.md`. Markdown is the canonical
  source of truth for the body of the paper; the LaTeX is generated.
- **Citations** in prose use the Pandoc `[@Key]` syntax — for example,
  `[@Soroush2024]`. Multiple citations: `[@Kim2025; @Hatem2025]`.
  Pandoc converts these to `\autocite{}` macros, biblatex resolves them
  at build time against `references.bib`.
- **Bibliography** is `references.bib` (BibTeX format). Add a new entry
  by appending an `@article{Key, ...}` block. The "Why we cite" reading
  notes are preserved as `% comment` lines above each entry — comments
  don't appear in the rendered PDF but stay alongside the entry they
  describe, helping future contributors understand the rationale.
- **Document structure** (sections, packages, title, abstract, citation
  style) lives in `main.tex`. You only touch this when adding/removing/
  reordering sections or changing how the bibliography is formatted.
- **Adding a section**: create `sections/NN_name.md` and add a matching
  `\input{build/sections/NN_name}` line in `main.tex`. The Makefile
  picks up the new markdown automatically.

## Building the PDF locally

### One-time setup

```bash
make deps           # prints install hints for pandoc + LaTeX + biber
```

On macOS (typical):

```bash
brew install pandoc
brew install --cask basictex            # ~100MB; sufficient
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended \
                   collection-latexrecommended \
                   framed parskip xurl csquotes microtype \
                   biblatex biber logreq biblatex-vancouver
```

If `tlmgr` mirror access is flaky, switch to a specific mirror:

```bash
sudo tlmgr option repository https://mirrors.mit.edu/CTAN/systems/texlive/tlnet
```

### Build commands

```bash
cd paper

make pdf            # rebuild → build/paper.pdf
make snapshot       # promote build/paper.pdf → paper/paper.pdf (for committing)
make clean          # wipe build/
make watch          # rebuild on any change to sections/, references.bib,
                    #   or main.tex (requires `brew install entr`)
```

The `paper/paper.pdf` checked into the repo is intentionally a
manually-promoted snapshot rather than every build, so commit history
stays clean. Run `make snapshot` after a meaningful content change
worth a new public PDF.

## What gets generated

`make pdf` runs in four phases:

1. **Pandoc** converts each `sections/*.md` to `build/sections/*.tex`,
   preserving headings (`#` → `\section`), tables (booktabs), code
   blocks (verbatim), and links (hyperref). The `--biblatex` flag makes
   pandoc emit `\autocite{Key}` for `[@Key]` callouts.
2. **xelatex pass 1** — collects citations into `build/main.bcf`.
3. **biber** — resolves citations against `references.bib`, writes the
   formatted bibliography to `build/main.bbl`.
4. **xelatex passes 2 and 3** — first inserts the bibliography, second
   resolves cross-references and the table of contents.

Output: `build/paper.pdf`.

## Switching citation style

Single-line change in `paper/main.tex`. Find the biblatex `\usepackage`
line in the preamble and edit `style=`:

```latex
% vancouver [1] superscripted — biomedical (NEJM AI, JAMIA, BMJ) ← current
\usepackage[backend=biber, style=vancouver, sortlocale=en_US]{biblatex}

% numeric-comp [1, 2] — IEEE, Nature, NeurIPS
\usepackage[backend=biber, style=numeric-comp, sortlocale=en_US]{biblatex}

% authoryear (Devlin, 2019) — easier-to-skim drafting style
\usepackage[backend=biber, style=authoryear, sortlocale=en_US]{biblatex}
```

Then `make clean && make pdf`. No changes to the `.bib` or to the
markdown citations — biblatex re-renders everything in the new style.

## Adding a new reference

1. Open `references.bib`.
2. Add a `% Why we cite: ...` comment block describing the rationale.
3. Add the BibTeX entry below it. Citation key convention:
   `LastNameYear` (e.g., `Smith2025`); for multi-author papers use the
   first author's surname; for organizational authors use a short
   acronym (`CMS2026`).
4. Cite from prose with `[@NewKey]`. Run `make pdf` — the entry appears
   in the bibliography automatically (only if cited; uncited entries
   are excluded by biblatex's default behavior).

## When to drop pandoc

Once the paper structure is locked and we're closer to submission, the
right move is:

- Convert each section to `.tex` once via pandoc and commit the `.tex`.
- Drop the Makefile's pandoc step.
- Hand-edit the `.tex` for venue-specific styling and figure placement.
- Switch `main.tex` to a venue-specific style file (`acl_natbib.sty`,
  `neurips.sty`, etc.) if the venue provides one.
- The bibliography flow (biblatex + `references.bib`) stays the same.

For draft preview, the markdown-driven flow is much faster to iterate on.

## Known limitations of the current setup

- **No figures yet.** `paper/figures/` is empty. Add PDFs/PNGs there
  and reference via `![caption](../figures/foo.pdf)` in markdown
  (pandoc will translate to `\includegraphics`).
- **Some bibliographic entries use `and others`** in place of full
  author lists where the source markdown only had `et al.` — biblatex
  renders these correctly, but if a venue requires full author lists
  for the published bibliography, expand the affected `author = {...}`
  fields in `references.bib`.

## Reusing this scaffolding

If you're building your own research paper and want to start from a
similar markdown-canonical / Pandoc + biblatex / xelatex setup, the
files in this directory (`main.tex`, `Makefile`, `references.bib` as a
template, `.gitignore`) are MIT-licensed along with the rest of the
repository. Drop them into a fresh `paper/` directory, swap in your
own prose and citations, and you should be running with `make pdf` in
a few minutes (assuming you have pandoc + a TeX distribution
installed).
