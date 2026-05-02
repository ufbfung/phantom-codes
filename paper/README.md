# Paper build

The Phantom Codes working draft. Markdown is canonical for prose;
LaTeX is generated; bibliography lives in `references.bib`.

**Target venue: NEJM AI** (first submission). The paper structure
(IMRaD), citation style (Vancouver / AMA-family numeric superscripts),
structured abstract, and back-matter blocks (funding, data
availability, code availability, author contributions, competing
interests) all match NEJM AI's manuscript guidance. Switching to a
different venue is mostly a citation-style swap and a section-order
re-arrange — the prose stays.

## Layout

```
paper/
  main.tex          # LaTeX skeleton — documentclass, title,
                    #   structured abstract, biblatex setup,
                    #   \input{} of generated sections, back-matter
  references.bib    # bibliography (BibTeX); single source of truth
  Makefile          # build automation
  README.md         # this file
  .gitignore        # ignore build/
  sections/         # canonical prose (edit here)
    00_introduction.md       # framing + 3 contributions + this-work overview
    01_methodology.md        # data + cohort + architecture + hardware
    02_cost_economics.md     # cost-per-correct framework + extrapolations
    03_results.md            # SCAFFOLD — fill from headline run
    04_discussion.md         # SCAFFOLD — fill from §03 numbers
    05_conclusion.md         # SCAFFOLD — short, ~150 words
  build/            # generated (gitignored)
    sections/*.tex  # pandoc-converted .tex files
    paper.pdf       # final output
```

## Editing model

- **Prose edits** happen in `sections/*.md`. Markdown stays the source
  of truth for the body of the paper.
- **Citations** in prose use the `[@Key]` syntax — for example,
  `[@Soroush2024]`. Multiple citations: `[@Kim2025; @Hatem2025]`.
  Pandoc converts these to `\autocite{}` macros, biblatex resolves
  them at build time against `references.bib`.
- **Bibliography** is `references.bib` (BibTeX format). Add a new
  entry by appending an `@article{Key, ...}` block. The "Why we cite"
  reading notes are preserved as `% comment` lines above each entry —
  comments don't appear in the rendered PDF but stay alongside the
  entry they describe.
- **Document structure** (sections, packages, title, abstract, citation
  style) lives in `main.tex`. You only touch this when adding/removing/
  reordering sections, swapping document class, or changing how the
  bibliography is formatted.
- **Adding a section**: create `sections/NN_name.md` and add a
  matching `\input{build/sections/NN_name}` line in `main.tex`. The
  Makefile picks up the new markdown automatically.

## Building

### One-time setup

```bash
make deps           # prints install hints for pandoc + LaTeX + biber
```

On macOS (typical):

```bash
brew install pandoc
brew install --cask basictex            # ~100MB; sufficient for our packages
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended \
                   collection-latexrecommended \
                   framed parskip xurl csquotes microtype \
                   biblatex biber logreq
```

### Build commands

```bash
cd paper

make pdf            # build/paper.pdf
make clean          # wipe build/
make watch          # rebuild on any change to sections/, references.bib,
                    #   or main.tex (requires `brew install entr`)
```

## What gets generated

`make pdf` runs in four phases:

1. **Pandoc** converts each `sections/*.md` to `build/sections/*.tex`,
   preserving headings (`#` → `\section`), tables (booktabs), code
   blocks (verbatim), and links (hyperref). The `--biblatex` flag
   makes pandoc emit `\autocite{Key}` for `[@Key]` callouts.
2. **xelatex pass 1** — collects citations into `build/main.bcf`.
3. **biber** — resolves citations against `references.bib`, writes
   the formatted bibliography to `build/main.bbl`.
4. **xelatex passes 2 and 3** — first inserts the bibliography,
   second resolves cross-references and the table of contents.

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

## Pre-submission checklist (NEJM AI)

Before submission, walk through this list:

- [ ] **Results section populated** with headline numbers from the
      Synthea evaluation matrix and the trained-classifier test pass.
- [ ] **Discussion section populated** with interpretation grounded
      in §Results numbers.
- [ ] **Conclusions rewritten** from scaffold to two-paragraph final
      form (~150 words).
- [ ] **Abstract Results + Conclusions filled in** (still TBD as of
      this draft); abstract under 300 words total across all four
      structured sections.
- [ ] **Back-matter populated**: funding statement final; data
      availability accurate; code availability points to the correct
      repository URL; author contributions accurate; competing-
      interests declaration accurate; acknowledgments listed if any.
- [ ] **Line numbers enabled** for review: uncomment the
      `\usepackage{lineno}` and `\linenumbers` lines in `main.tex`.
- [ ] **Double-spacing enabled** for review: uncomment
      `\usepackage{setspace}` and `\doublespacing`.
- [ ] **Word count under target**: NEJM AI Original Articles run
      ~5,000-8,000 words main text (excluding abstract, refs,
      methods that go in supplementary). Check current word count
      with `pandoc sections/*.md --lua-filter wordcount.lua` or just
      `wc -w paper/sections/*.md`.
- [ ] **Figures and tables**: NEJM AI typically allows 5-6 each.
      Inventory before submission and consider moving overflow to
      supplementary materials.
- [ ] **All `[TBD]` markers resolved or moved to supplementary**.
      Search with `grep -rn '\[TBD' paper/sections/`.
- [ ] **Spell-check + grammar pass** on the rendered PDF (not just
      the markdown — pandoc conversion can occasionally surface
      issues that don't show in markdown).

## Adding a new reference

1. Open `references.bib`.
2. Add a `% Why we cite: ...` comment block describing the rationale.
3. Add the BibTeX entry below it. Citation key convention:
   `LastNameYear` (e.g., `Smith2025`); for multi-author papers use the
   first author's surname; for organizational authors use a short
   acronym (`CMS2026`).
4. Cite from prose with `[@NewKey]`. Run `make pdf` — the entry
   appears in the bibliography automatically (only if cited; uncited
   entries are excluded by biblatex's default behavior).

## When to drop pandoc

Once the paper structure is locked and we're closer to submission, the
right move is:

- Convert each section to `.tex` once via pandoc and **commit the
  `.tex`** to version control.
- Drop the Makefile's pandoc step.
- Hand-edit the `.tex` for venue-specific styling and figure placement.
- Switch `main.tex` to a venue-specific style file (`acl_natbib.sty`,
  `neurips.sty`, etc.).
- The bibliography flow (biblatex + `references.bib`) stays the same.

For draft preview, the markdown-driven flow is much faster to iterate on.

## Known limitations of the current setup

- **No figures yet.** `paper/figures/` is empty. Add PDFs/PNGs there
  and reference via `![caption](../figures/foo.pdf)` in markdown
  (pandoc will translate to `\includegraphics`).
- **Some bibliographic entries use `and others`** in place of full
  author lists where the source markdown only had `et al.` — biblatex
  renders these correctly, but if you need the full author list for a
  venue submission, fill in the `author = {...}` field of the affected
  entries in `references.bib`.
