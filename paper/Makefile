# Phantom Codes — paper build.
#
# Targets:
#   make pdf      — convert all sections/*.md → build/sections/*.tex via
#                   pandoc, then compile main.tex → build/paper.pdf
#   make snapshot — promote build/paper.pdf → paper/paper.pdf so the
#                   committed draft PDF is up to date. Run after
#                   meaningful content changes; commit the resulting
#                   paper/paper.pdf so casual readers can preview the
#                   draft on GitHub without a local LaTeX install.
#   make clean    — wipe build/
#   make watch    — rebuild on any source change (requires `entr`:
#                   `brew install entr`)
#   make deps     — print install hints for pandoc + LaTeX
#
# Quick reference for new contributors:
#   - Edit prose in sections/*.md (markdown is canonical)
#   - Edit document structure / packages / title in main.tex
#   - Run `make pdf` to preview
#   - Generated build/ is gitignored

# ─── Inputs ────────────────────────────────────────────────────────────
# All markdown sections, in lexicographic order. Add a new section by
# dropping a sections/NN_name.md file and appending an \input{} line
# in main.tex; the Makefile picks it up automatically.
MD_SECTIONS := $(sort $(wildcard sections/*.md))
TEX_SECTIONS := $(patsubst sections/%.md,build/sections/%.tex,$(MD_SECTIONS))
PDF := build/paper.pdf

# ─── Tools ─────────────────────────────────────────────────────────────
PANDOC := pandoc
PDFLATEX := xelatex
BIBER := biber
# --biblatex makes pandoc emit \autocite{key} for [@key] callouts in the
# markdown, deferring resolution to biblatex/biber at xelatex time.
PANDOC_OPTS := \
	--from markdown+pipe_tables+backtick_code_blocks+yaml_metadata_block \
	--to latex \
	--top-level-division=section \
	--wrap=preserve \
	--biblatex

PDFLATEX_OPTS := \
	-interaction=nonstopmode \
	-halt-on-error \
	-output-directory=build

BIBER_OPTS := \
	--input-directory=build \
	--output-directory=build

# ─── Targets ───────────────────────────────────────────────────────────
.PHONY: pdf snapshot clean watch deps

pdf: $(PDF)

$(PDF): main.tex references.bib $(TEX_SECTIONS) | build
	@echo "[paper] compiling main.tex (pass 1 — collect citations)"
	$(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[paper] running biber (resolve bibliography)"
	$(BIBER) $(BIBER_OPTS) main >/dev/null
	@echo "[paper] compiling main.tex (pass 2 — insert bibliography)"
	$(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[paper] compiling main.tex (pass 3 — finalize cross-refs)"
	$(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@mv build/main.pdf $(PDF)
	@echo "[paper] ✓ $(PDF)"

build/sections/%.tex: sections/%.md | build/sections
	@echo "[paper] pandoc: $< → $@"
	$(PANDOC) $(PANDOC_OPTS) $< -o $@

build:
	@mkdir -p build

build/sections:
	@mkdir -p build/sections

clean:
	@rm -rf build/
	@echo "[paper] cleaned build/"

# Promote the latest build to the committed snapshot. Idempotent — safe to
# re-run. We keep this manual rather than auto-copying on every `make pdf`
# so the committed PDF only updates at meaningful checkpoints (per the
# user's discretion), not on every transient rebuild.
snapshot: $(PDF)
	@cp $(PDF) paper.pdf
	@echo "[paper] snapshot → paper/paper.pdf (commit when ready)"

# Live-rebuild on any change. Requires `entr` (brew install entr).
watch:
	@command -v entr >/dev/null || { \
		echo "entr not found. Install with: brew install entr"; exit 1; }
	@echo "[paper] watching for changes — Ctrl+C to stop"
	@find sections main.tex Makefile | entr -c make pdf

deps:
	@echo "Build requires pandoc + a LaTeX distribution + biblatex/biber."
	@echo
	@echo "macOS:"
	@echo "  brew install pandoc"
	@echo "  brew install --cask basictex      # ~100MB minimal TeX"
	@echo "  # OR: brew install --cask mactex  # ~5GB full TeX"
	@echo
	@echo "After installing basictex, install our extra packages:"
	@echo "  sudo tlmgr update --self"
	@echo "  sudo tlmgr install collection-fontsrecommended \\"
	@echo "                     collection-latexrecommended \\"
	@echo "                     framed parskip xurl csquotes microtype \\"
	@echo "                     biblatex biber logreq"
	@echo
	@echo "Verify install:"
	@echo "  pandoc --version"
	@echo "  xelatex --version"
	@echo "  biber --version"
