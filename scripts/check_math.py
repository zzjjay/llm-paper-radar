"""Lint LaTeX math in Markdown for GitHub's restricted renderer.

GitHub renders `$...$` / `$$...$$` math with a *restricted* MathJax: on top of
normal LaTeX errors it maintains its own "macro not allowed" denylist, and it
breaks on a few structural things (a `$` nested inside a `$$` block, an
unrecognized `\left` delimiter, unbalanced braces). Those failures only show up
as red error boxes *after* you push. This script catches them beforehand.

Note on why not KaTeX: KaTeX is more permissive than GitHub here — it renders
`\operatorname`, `\lVert`, etc. fine — so it would miss exactly the GitHub-
specific rejections we hit. This checker targets GitHub's actual failure modes
directly, with no node dependency.

What it checks, per file:
  1. total `$` count is even (no unclosed math),
  2. no stray `$` inside a `$$...$$` display block (nested `$` closes it early —
     e.g. `\text{$b$-bit}` -> orphans the following `\left`),
  3. denylisted macros GitHub rejects (`\operatorname`, `\lVert`/`\rVert`,
     `\big`/`\Big` sizing, `\def`/`\newcommand`/`\color`/`\require`/...),
  4. balanced `{ }` and balanced `\left ... \right` within each math span.

Each finding prints `file:line  <problem>  <fix>`. Exit code 1 if any file has
findings, 0 if all clean — so it works as a pre-commit / CI gate.

Usage:
    uv run python scripts/check_math.py interpretations/GSQ-2604.18556/translation_zh.md
    uv run python scripts/check_math.py interpretations/**/*.md
    uv run python scripts/check_math.py            # defaults to interpretations/**/*.md
"""

from __future__ import annotations

import re
import sys
from glob import glob
from pathlib import Path

# GitHub-rejected macros -> suggested GitHub-safe replacement.
DENYLIST: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\operatorname\*?"), r"\operatorname → use \arg\min / \arg\max / bare \min,\max"),
    (re.compile(r"\\[lr]Vert"), r"\lVert/\rVert → use \|"),
    (re.compile(r"\\(?:big|Big|bigg|Bigg)[lrm]?\b"), r"\big/\Big sizing → drop it, or use \left…\right"),
    (re.compile(r"\\newcommand|\\renewcommand|\\def\b"), r"macro definitions are not allowed on GitHub"),
    (re.compile(r"\\color\b|\\textcolor\b"), r"\color/\textcolor is not allowed on GitHub"),
    (re.compile(r"\\require\b|\\htmlClass\b|\\href\b"), r"\require/\htmlClass/\href is not allowed on GitHub"),
]


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _iter_math_spans(text: str):
    """Yield (kind, body, start_index) for each math span.

    Display `$$...$$` first (greedy-safe, non-nested), then inline `$...$` on the
    text with display spans masked out so their inner `$` don't re-trigger.
    """
    spans = []
    masked = list(text)
    for m in re.finditer(r"\$\$(.+?)\$\$", text, flags=re.DOTALL):
        spans.append(("display", m.group(1), m.start()))
        for i in range(m.start(), m.end()):
            masked[i] = " " if text[i] != "\n" else "\n"
    masked_text = "".join(masked)
    # Inline: single $ not adjacent to another $.
    for m in re.finditer(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", masked_text, flags=re.DOTALL):
        spans.append(("inline", m.group(1), m.start()))
    spans.sort(key=lambda s: s[2])
    return spans


def _balanced(s: str, open_ch: str, close_ch: str) -> bool:
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":  # skip escaped char (e.g. \{ \})
            i += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    findings: list[str] = []
    rel = path

    # 1. even $ count
    dollars = text.count("$")
    if dollars % 2 != 0:
        findings.append(f"{rel}  odd '$' count ({dollars}) — an unclosed $ or $$ somewhere")

    # 2. stray $ inside a display block (line-scoped for a precise pointer)
    for m in re.finditer(r"\$\$(.+?)\$\$", text, flags=re.DOTALL):
        if "$" in m.group(1):
            ln = _line_of(text, m.start())
            findings.append(
                f"{rel}:{ln}  stray '$' inside a $$…$$ block (e.g. \\text{{$x$}}) — "
                r"closes math early; write x\text{…} instead"
            )

    # 3 & 4: per-span macro denylist + brace / \left-\right balance
    for kind, body, idx in _iter_math_spans(text):
        ln = _line_of(text, idx)
        for pat, msg in DENYLIST:
            if pat.search(body):
                findings.append(f"{rel}:{ln}  disallowed macro: {msg}")
        if not _balanced(body, "{", "}"):
            findings.append(f"{rel}:{ln}  unbalanced {{ }} in {kind} math")
        n_left = len(re.findall(r"\\left\b", body))
        n_right = len(re.findall(r"\\right\b", body))
        if n_left != n_right:
            findings.append(f"{rel}:{ln}  \\left ({n_left}) / \\right ({n_right}) mismatch in {kind} math")

    return findings


def main(argv: list[str]) -> int:
    args = argv or ["interpretations/**/*.md"]
    files: list[Path] = []
    for a in args:
        files.extend(Path(p) for p in glob(a, recursive=True))
    files = sorted({f for f in files if f.is_file()})
    if not files:
        print("check_math: no files matched", file=sys.stderr)
        return 0

    total = 0
    for f in files:
        findings = check_file(f)
        for line in findings:
            print(line)
        total += len(findings)

    n = len(files)
    if total:
        print(f"\ncheck_math: {total} issue(s) across {n} file(s)")
        return 1
    print(f"check_math: {n} file(s) clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
