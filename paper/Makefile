# Phantom Codes — paper build (pure LaTeX, no pandoc).
#
# Repository hosts two papers from a single project:
#
#   1. Phantom Codes (JAMIA submission)
#      - sources: paper/phantom_codes/{main.tex,supplementary.tex,sections/,supp_sections/}
#      - snapshots: paper/phantom_codes.pdf, paper/phantom_codes_supplementary.pdf
#
#   2. PubMedBERT fine-tuning tech report (planned arXiv preprint)
#      - sources: paper/pubmedbert/{main.tex,sections/}
#      - snapshot: paper/pubmedbert_finetuning.pdf
#
# Both papers share paper/references.bib via biblatex (their .tex masters
# point at ../references.bib). Each paper has its own self-contained build/
# subdirectory under its sources directory for xelatex/biber intermediate
# output (.aux, .bbl, .toc) and the final .pdf.
#
# Engine: xelatex (fontspec + STIX Two Text for full Unicode coverage).
# Citation backend: biber + biblatex (Vancouver style, see main.tex preamble).
#
# Targets:
#   make phantom_codes              — build phantom_codes/build/main.pdf
#   make phantom_codes_supp         — build phantom_codes/build/supplementary.pdf
#   make pubmedbert                 — build pubmedbert/build/main.pdf
#   make all                        — build all three
#
#   make snapshot-phantom_codes     — promote → paper/phantom_codes.pdf
#   make snapshot-phantom_codes-supp — promote → paper/phantom_codes_supplementary.pdf
#   make snapshot-pubmedbert        — promote → paper/pubmedbert_finetuning.pdf
#   make snapshot-all               — all three snapshots in one go
#
#   make clean                      — wipe both build/ subdirectories
#   make watch                      — rebuild on any source change (requires `entr`)
#   make deps                       — print install hints
#   make test                       — run regression tests (build + page-count check)

# ─── Source/target inputs ──────────────────────────────────────────────
PC_DIR := phantom_codes
PC_TEX_SECTIONS := $(sort $(wildcard $(PC_DIR)/sections/*.tex))
PC_PDF := $(PC_DIR)/build/main.pdf

PC_TEX_SUPP_SECTIONS := $(sort $(wildcard $(PC_DIR)/supp_sections/*.tex))
PC_SUPP_PDF := $(PC_DIR)/build/supplementary.pdf

PB_DIR := pubmedbert
PB_TEX_SECTIONS := $(sort $(wildcard $(PB_DIR)/sections/*.tex))
PB_PDF := $(PB_DIR)/build/main.pdf

# ─── Tools ─────────────────────────────────────────────────────────────
PDFLATEX := xelatex
BIBER := biber
PDFLATEX_OPTS := -interaction=nonstopmode -halt-on-error -output-directory=build

# ─── Targets ───────────────────────────────────────────────────────────
.PHONY: all phantom_codes phantom_codes_supp pubmedbert \
	snapshot-phantom_codes snapshot-phantom_codes-supp snapshot-pubmedbert snapshot-all \
	clean watch deps test

all: phantom_codes phantom_codes_supp pubmedbert
phantom_codes: $(PC_PDF)
phantom_codes_supp: $(PC_SUPP_PDF)
pubmedbert: $(PB_PDF)

# ─── Phantom Codes main paper ──────────────────────────────────────────
$(PC_PDF): $(PC_DIR)/main.tex references.bib $(PC_TEX_SECTIONS) | $(PC_DIR)/build
	@echo "[phantom_codes] compiling main.tex (pass 1)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[phantom_codes] biber"
	cd $(PC_DIR) && $(BIBER) --input-directory=build --output-directory=build main >/dev/null
	@echo "[phantom_codes] compiling main.tex (pass 2)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[phantom_codes] compiling main.tex (pass 3)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[phantom_codes] ✓ $(PC_PDF)"

# ─── Phantom Codes supplement ──────────────────────────────────────────
$(PC_SUPP_PDF): $(PC_DIR)/supplementary.tex references.bib $(PC_TEX_SUPP_SECTIONS) | $(PC_DIR)/build
	@echo "[phantom_codes_supp] compiling supplementary.tex (pass 1)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) supplementary.tex >/dev/null
	@echo "[phantom_codes_supp] biber"
	cd $(PC_DIR) && $(BIBER) --input-directory=build --output-directory=build supplementary >/dev/null
	@echo "[phantom_codes_supp] compiling supplementary.tex (pass 2)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) supplementary.tex >/dev/null
	@echo "[phantom_codes_supp] compiling supplementary.tex (pass 3)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) supplementary.tex >/dev/null
	@echo "[phantom_codes_supp] ✓ $(PC_SUPP_PDF)"

# ─── PubMedBERT tech report ────────────────────────────────────────────
$(PB_PDF): $(PB_DIR)/main.tex references.bib $(PB_TEX_SECTIONS) | $(PB_DIR)/build
	@echo "[pubmedbert] compiling main.tex (pass 1)"
	cd $(PB_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[pubmedbert] biber"
	cd $(PB_DIR) && $(BIBER) --input-directory=build --output-directory=build main >/dev/null
	@echo "[pubmedbert] compiling main.tex (pass 2)"
	cd $(PB_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[pubmedbert] compiling main.tex (pass 3)"
	cd $(PB_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[pubmedbert] ✓ $(PB_PDF)"

# ─── Build directories (order-only prerequisites) ──────────────────────
$(PC_DIR)/build:
	@mkdir -p $(PC_DIR)/build

$(PB_DIR)/build:
	@mkdir -p $(PB_DIR)/build

# ─── Snapshot promote (build PDF → committed snapshot at paper/) ───────
# Each snapshot is manual rather than auto so the committed PDFs only
# update at meaningful checkpoints, not on every transient rebuild.

snapshot-phantom_codes: $(PC_PDF)
	@cp $(PC_PDF) phantom_codes.pdf
	@echo "[phantom_codes] snapshot → paper/phantom_codes.pdf (commit when ready)"

snapshot-phantom_codes-supp: $(PC_SUPP_PDF)
	@cp $(PC_SUPP_PDF) phantom_codes_supplementary.pdf
	@echo "[phantom_codes_supp] snapshot → paper/phantom_codes_supplementary.pdf (commit when ready)"

snapshot-pubmedbert: $(PB_PDF)
	@cp $(PB_PDF) pubmedbert_finetuning.pdf
	@echo "[pubmedbert] snapshot → paper/pubmedbert_finetuning.pdf (commit when ready)"

snapshot-all: snapshot-phantom_codes snapshot-phantom_codes-supp snapshot-pubmedbert
	@echo "[paper] all three snapshots refreshed (commit when ready)"

# ─── Cleanup ───────────────────────────────────────────────────────────
clean:
	@rm -rf $(PC_DIR)/build $(PB_DIR)/build
	@echo "[paper] cleaned phantom_codes/build/ and pubmedbert/build/"

# ─── Live rebuild ──────────────────────────────────────────────────────
watch:
	@command -v entr >/dev/null || { \
		echo "entr not found. Install with: brew install entr"; exit 1; }
	@echo "[paper] watching for changes — Ctrl+C to stop"
	@find phantom_codes pubmedbert references.bib Makefile | entr -c make all

# ─── Regression test ───────────────────────────────────────────────────
# Builds all three documents and verifies expected page counts. Used by
# CI / pre-commit to catch source-level errors that would otherwise only
# surface at submission time.
test: all
	@echo "[test] checking page counts"
	@pages=$$(pdfinfo $(PC_PDF) | awk '/^Pages:/ {print $$2}'); \
		[ "$$pages" -ge 15 ] || { echo "FAIL: $(PC_PDF) has $$pages pages (expected >=15)"; exit 1; }; \
		echo "[test] ✓ phantom_codes: $$pages pages"
	@pages=$$(pdfinfo $(PC_SUPP_PDF) | awk '/^Pages:/ {print $$2}'); \
		[ "$$pages" -ge 30 ] || { echo "FAIL: $(PC_SUPP_PDF) has $$pages pages (expected >=30)"; exit 1; }; \
		echo "[test] ✓ phantom_codes_supp: $$pages pages"
	@pages=$$(pdfinfo $(PB_PDF) | awk '/^Pages:/ {print $$2}'); \
		[ "$$pages" -ge 8 ] || { echo "FAIL: $(PB_PDF) has $$pages pages (expected >=8)"; exit 1; }; \
		echo "[test] ✓ pubmedbert: $$pages pages"
	@echo "[test] all builds clean, page counts plausible"

deps:
	@echo "Build requires a LaTeX distribution + biblatex/biber."
	@echo
	@echo "macOS:"
	@echo "  brew install --cask basictex      # ~100MB minimal TeX"
	@echo "  # OR: brew install --cask mactex  # ~5GB full TeX"
	@echo
	@echo "After installing basictex, install our extra packages:"
	@echo "  sudo tlmgr update --self"
	@echo "  sudo tlmgr install collection-fontsrecommended \\"
	@echo "                     collection-latexrecommended \\"
	@echo "                     framed parskip xurl csquotes microtype \\"
	@echo "                     biblatex biber logreq stix2-otf"
	@echo
	@echo "Verify install:"
	@echo "  xelatex --version"
	@echo "  biber --version"
