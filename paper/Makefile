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
# Four xelatex passes (1 + biber + 3) so the TOC + back-matter
# bibliography pagination converges; without the 4th pass, biblatex's
# "Label(s) may have changed. Rerun..." warning is left unresolved
# and the TOC page numbers stay stale by 1 page in some sections.
$(PC_PDF): $(PC_DIR)/main.tex references.bib $(PC_TEX_SECTIONS) | $(PC_DIR)/build
	@echo "[phantom_codes] compiling main.tex (pass 1)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[phantom_codes] biber"
	cd $(PC_DIR) && $(BIBER) --input-directory=build --output-directory=build main >/dev/null
	@echo "[phantom_codes] compiling main.tex (pass 2)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[phantom_codes] compiling main.tex (pass 3)"
	cd $(PC_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[phantom_codes] compiling main.tex (pass 4)"
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
	@echo "[phantom_codes_supp] compiling supplementary.tex (pass 4)"
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
	@echo "[pubmedbert] compiling main.tex (pass 4)"
	cd $(PB_DIR) && $(PDFLATEX) $(PDFLATEX_OPTS) main.tex >/dev/null
	@echo "[pubmedbert] ✓ $(PB_PDF)"

# ─── Build directories (order-only prerequisites) ──────────────────────
$(PC_DIR)/build:
	@mkdir -p $(PC_DIR)/build

$(PB_DIR)/build:
	@mkdir -p $(PB_DIR)/build

# ─── Flatten (modular sources → single self-contained .tex) ────────────
# JAMIA wants the LaTeX source as supplemental, and reviewing 12 .tex
# files across 2 directories is awkward. The flatten step generates
# `paper/phantom_codes.tex` and `paper/phantom_codes_supplementary.tex`
# as build artifacts: single self-contained files that compile to
# byte-identical PDFs as the modular build. Edit the modular sections;
# the flat .tex is regenerated automatically by every snapshot.

PYTHON := python3
FLATTEN := $(PYTHON) scripts/flatten_paper.py

PC_FLAT_TEX := phantom_codes.tex
PC_SUPP_FLAT_TEX := phantom_codes_supplementary.tex

.PHONY: flatten-phantom_codes flatten-phantom_codes-supp flatten-all verify-flat

flatten-phantom_codes: $(PC_DIR)/main.tex $(PC_TEX_SECTIONS)
	@$(FLATTEN) $(PC_DIR)/main.tex $(PC_FLAT_TEX)

flatten-phantom_codes-supp: $(PC_DIR)/supplementary.tex $(PC_TEX_SUPP_SECTIONS)
	@$(FLATTEN) $(PC_DIR)/supplementary.tex $(PC_SUPP_FLAT_TEX)

flatten-all: flatten-phantom_codes flatten-phantom_codes-supp

# `verify-flat` compiles the flat .tex into a separate build dir
# and runs a 3-axis equivalence check against the modular-build PDF:
#
#   1. Page count          (catches structural drift via pdfinfo)
#   2. Plain text content  (catches prose / citation / section drift via pdftotext)
#   3. Layout-aware text   (catches table column shifts and figure
#                           position drift via pdftotext -layout)
#
# All three must match (modulo \today date drift) for verify to pass.
# verify-flat is chained into `snapshot-all` and `test`; you don't
# normally need to call it directly.
verify-flat: SHELL := /bin/bash
verify-flat: flatten-all $(PC_PDF) $(PC_SUPP_PDF) | build
	@echo "[verify-flat] compiling flat phantom_codes.tex (4 passes for TOC convergence)"
	@$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=build $(PC_FLAT_TEX) >/dev/null
	@$(BIBER) --input-directory=build --output-directory=build phantom_codes >/dev/null
	@$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=build $(PC_FLAT_TEX) >/dev/null
	@$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=build $(PC_FLAT_TEX) >/dev/null
	@$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=build $(PC_FLAT_TEX) >/dev/null
	@echo "[verify-flat] compiling flat phantom_codes_supplementary.tex (4 passes for TOC convergence)"
	@$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=build $(PC_SUPP_FLAT_TEX) >/dev/null
	@$(BIBER) --input-directory=build --output-directory=build phantom_codes_supplementary >/dev/null
	@$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=build $(PC_SUPP_FLAT_TEX) >/dev/null
	@$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=build $(PC_SUPP_FLAT_TEX) >/dev/null
	@$(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=build $(PC_SUPP_FLAT_TEX) >/dev/null
	@echo "[verify-flat] axis 1: page count"
	@m_main=$$(pdfinfo $(PC_PDF) | awk '/^Pages:/ {print $$2}'); \
		f_main=$$(pdfinfo build/phantom_codes.pdf | awk '/^Pages:/ {print $$2}'); \
		[ "$$m_main" = "$$f_main" ] || { echo "FAIL: phantom_codes page count $$m_main (modular) != $$f_main (flat)"; exit 1; }; \
		echo "[verify-flat]   ✓ phantom_codes: $$m_main pages match"
	@m_supp=$$(pdfinfo $(PC_SUPP_PDF) | awk '/^Pages:/ {print $$2}'); \
		f_supp=$$(pdfinfo build/phantom_codes_supplementary.pdf | awk '/^Pages:/ {print $$2}'); \
		[ "$$m_supp" = "$$f_supp" ] || { echo "FAIL: phantom_codes_supp page count $$m_supp (modular) != $$f_supp (flat)"; exit 1; }; \
		echo "[verify-flat]   ✓ phantom_codes_supp: $$m_supp pages match"
	@echo "[verify-flat] axis 2: plain text content"
	@pdftotext $(PC_PDF) /tmp/verify_modular_main.txt
	@pdftotext build/phantom_codes.pdf /tmp/verify_flat_main.txt
	@pdftotext $(PC_SUPP_PDF) /tmp/verify_modular_supp.txt
	@pdftotext build/phantom_codes_supplementary.pdf /tmp/verify_flat_supp.txt
	@: > /tmp/verify-flat.diff
	@diff /tmp/verify_modular_main.txt /tmp/verify_flat_main.txt | grep -vE '^[0-9]+c[0-9]+$$|^---$$|^[<>] [A-Z][a-z]+ [0-9]+, 20[0-9][0-9]$$' >> /tmp/verify-flat.diff || true
	@diff /tmp/verify_modular_supp.txt /tmp/verify_flat_supp.txt | grep -vE '^[0-9]+c[0-9]+$$|^---$$|^[<>] [A-Z][a-z]+ [0-9]+, 20[0-9][0-9]$$' >> /tmp/verify-flat.diff || true
	@if [ -s /tmp/verify-flat.diff ]; then \
		echo "FAIL: plain-text content differs (non-date drift):"; \
		head -40 /tmp/verify-flat.diff; \
		[ $$(wc -l < /tmp/verify-flat.diff) -gt 40 ] && echo "... (full diff in /tmp/verify-flat.diff)"; \
		exit 1; \
	fi
	@echo "[verify-flat]   ✓ plain text identical (date drift only)"
	@echo "[verify-flat] axis 3: layout-aware text"
	@pdftotext -layout $(PC_PDF) /tmp/verify_modular_main_layout.txt
	@pdftotext -layout build/phantom_codes.pdf /tmp/verify_flat_main_layout.txt
	@pdftotext -layout $(PC_SUPP_PDF) /tmp/verify_modular_supp_layout.txt
	@pdftotext -layout build/phantom_codes_supplementary.pdf /tmp/verify_flat_supp_layout.txt
	@: > /tmp/verify-flat-layout.diff
	@diff /tmp/verify_modular_main_layout.txt /tmp/verify_flat_main_layout.txt | grep -vE '^[0-9]+c[0-9]+$$|^---$$|^[<>][[:space:]]*[A-Z][a-z]+ [0-9]+, 20[0-9][0-9][[:space:]]*$$' >> /tmp/verify-flat-layout.diff || true
	@diff /tmp/verify_modular_supp_layout.txt /tmp/verify_flat_supp_layout.txt | grep -vE '^[0-9]+c[0-9]+$$|^---$$|^[<>][[:space:]]*[A-Z][a-z]+ [0-9]+, 20[0-9][0-9][[:space:]]*$$' >> /tmp/verify-flat-layout.diff || true
	@if [ -s /tmp/verify-flat-layout.diff ]; then \
		echo "FAIL: layout-aware text differs (table reflow, figure shift, etc.):"; \
		head -40 /tmp/verify-flat-layout.diff; \
		[ $$(wc -l < /tmp/verify-flat-layout.diff) -gt 40 ] && echo "... (full diff in /tmp/verify-flat-layout.diff)"; \
		exit 1; \
	fi
	@echo "[verify-flat]   ✓ layout identical (date drift only)"
	@echo "[verify-flat] ✓ flat .tex matches modular: page count, plain text, layout"

build:
	@mkdir -p build

# ─── Snapshot promote (build PDF → committed snapshot at paper/) ───────
# Each snapshot is manual rather than auto so the committed PDFs only
# update at meaningful checkpoints, not on every transient rebuild.
# Flatten is chained in so phantom_codes.tex / phantom_codes_supplementary.tex
# always regenerate alongside the PDF — no drift possible.

snapshot-phantom_codes: $(PC_PDF) flatten-phantom_codes
	@cp $(PC_PDF) phantom_codes.pdf
	@echo "[phantom_codes] snapshot → paper/phantom_codes.pdf + phantom_codes.tex (commit when ready)"

snapshot-phantom_codes-supp: $(PC_SUPP_PDF) flatten-phantom_codes-supp
	@cp $(PC_SUPP_PDF) phantom_codes_supplementary.pdf
	@echo "[phantom_codes_supp] snapshot → paper/phantom_codes_supplementary.pdf + phantom_codes_supplementary.tex (commit when ready)"

snapshot-pubmedbert: $(PB_PDF)
	@cp $(PB_PDF) pubmedbert_finetuning.pdf
	@echo "[pubmedbert] snapshot → paper/pubmedbert_finetuning.pdf (commit when ready)"

snapshot-all: snapshot-phantom_codes snapshot-phantom_codes-supp snapshot-pubmedbert verify-flat
	@echo "[paper] all three snapshots refreshed and equivalence-verified (commit when ready)"

# ─── Cleanup ───────────────────────────────────────────────────────────
clean:
	@rm -rf $(PC_DIR)/build $(PB_DIR)/build build
	@echo "[paper] cleaned phantom_codes/build/, pubmedbert/build/, and build/"

# ─── Live rebuild ──────────────────────────────────────────────────────
watch:
	@command -v entr >/dev/null || { \
		echo "entr not found. Install with: brew install entr"; exit 1; }
	@echo "[paper] watching for changes — Ctrl+C to stop"
	@find phantom_codes pubmedbert references.bib Makefile | entr -c make all

# ─── Regression test ───────────────────────────────────────────────────
# Builds all three documents, verifies page counts, and runs the
# 3-axis flat-vs-modular equivalence check. Used as the pre-submission
# gate to catch any drift between flat .tex and modular sources.
test: all verify-flat
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
