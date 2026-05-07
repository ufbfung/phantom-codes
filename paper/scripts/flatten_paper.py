#!/usr/bin/env python3
"""Flatten a modular LaTeX paper into a single self-contained .tex file.

Reads a master .tex file (e.g. paper/phantom_codes/main.tex), inlines
the contents of every `\\input{path}` it encounters, and writes the
result to a destination path that lives one directory level above
the master. Two relative paths are rewritten in place during
inlining so the flat file resolves correctly from its new location:

  \\addbibresource{../references.bib}      → \\addbibresource{references.bib}
  \\includegraphics{...}{../figures/foo}   → \\includegraphics{...}{figures/foo}

Usage:
    python3 flatten_paper.py <master.tex> <output.tex>

The script is intentionally simple: it understands plain
`\\input{path}` (one per line, no nesting beyond one level), skips
commented inputs, and treats anything else as a literal line. It
fails loudly on unexpected input shapes rather than silently
mangling the output.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

INPUT_RE = re.compile(r"^\s*\\input\{([^}]+)\}\s*$")

HEADER = """\
% ====================================================================
% GENERATED FILE — DO NOT EDIT.
%
% This file is auto-flattened by `make flatten` from the modular
% sources at paper/phantom_codes/{{main.tex,sections/}} (or the
% supplementary equivalent). Edit those instead; this file is
% overwritten on every `make snapshot-phantom_codes`.
%
% Source master: {source}
% Generator:     paper/scripts/flatten_paper.py
% ====================================================================

"""


def is_commented(line: str) -> bool:
    """Return True if the \\input is inside a LaTeX line comment."""
    pct = line.find("%")
    inp = line.find("\\input{")
    return pct != -1 and pct < inp


def flatten(master_path: Path) -> str:
    """Read master_path and return its contents with all \\input{} lines
    inlined verbatim. We deliberately do NOT add per-section banner
    comments around inlined content; even pure LaTeX comments can shift
    typesetting in subtle ways (TOC pagination drifted by 1 page when
    we tried it). The top-of-file generated-banner is the only marker
    a reader needs to know not to edit the file."""
    out: list[str] = []
    master_dir = master_path.parent
    for raw in master_path.read_text().splitlines(keepends=True):
        m = INPUT_RE.match(raw)
        if m and not is_commented(raw):
            target = master_dir / (m.group(1) + ".tex")
            if not target.exists():
                raise FileNotFoundError(f"\\input target missing: {target}")
            out.append(target.read_text())
            if not out[-1].endswith("\n"):
                out.append("\n")
        else:
            out.append(raw)
    return "".join(out)


def rewrite_paths(text: str) -> str:
    """Rewrite the two relative paths that change when the flat .tex
    moves up one directory."""
    replacements = {
        r"\addbibresource{../references.bib}": r"\addbibresource{references.bib}",
        "../figures/": "figures/",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Sanity: no remaining ../ paths should leak through.
    if "../" in text:
        leaks = [line.strip() for line in text.splitlines() if "../" in line]
        sys.stderr.write(
            "WARNING: ../ paths remain after rewrite — flatten may be incomplete:\n"
        )
        for leak in leaks[:5]:
            sys.stderr.write(f"  {leak}\n")
    return text


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    master = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()

    flat = flatten(master)
    flat = rewrite_paths(flat)
    banner = HEADER.format(source=master.relative_to(output.parent))
    output.write_text(banner + flat)

    n_lines = flat.count("\n")
    print(f"[flatten_paper] {master.name} → {output.name} ({n_lines} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
