"""Scan surfaced papers and generate paper-river for those without one.

A "surfaced paper" is any paper in `data/summarized/*.json` whose
`relevance_breakdown.hard_gate` is false — i.e. every paper that actually
appears in some digest/<date>.md.

Dedup: a paper is considered "done" if `paper-river/*<id>.org` exists in
either dot form (`...2604.18556.org`) or legacy dash form
(`...2604-18556.org`). The `_en.org` siblings are NOT counted as the
primary file — they are translations of the zh original (see
scripts/translate_paper_river.py).

Per-run cap: `PAPER_RIVER_MAX` env var. 0 (default) means unlimited.
On first run with a large window this can easily be 300+ candidates ×
5-10 minutes each = days of work; set the cap on first runs and lift
it when you're ready for a backfill batch.

The actual generation work is delegated to scripts/gen_paper_river.sh,
which invokes the ljg-paper-river Claude Code skill in headless mode.
Per-paper failure is logged and does not abort the batch.

Usage:
    uv run python scripts/auto_paper_river.py            # scan + gen all missing
    PAPER_RIVER_MAX=2 uv run python scripts/auto_paper_river.py   # cap at 2 this run
    uv run python scripts/auto_paper_river.py --dry-run  # list what would run
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
PAPER_RIVER_DIR = ROOT / "paper-river"
SUMMARIZED_DIR = ROOT / "data" / "summarized"
GEN_SCRIPT = ROOT / "scripts" / "gen_paper_river.sh"

# Matches both 2604.18556 (new dot form) and 2604-18556 (legacy dash form).
_ID_RE = re.compile(r"(\d{4})[.-](\d{4,5})")


def existing_ids() -> set[str]:
    """Return arxiv IDs that already have a paper-river/.org file (zh, the
    primary, not _en.org siblings). Both filename conventions counted."""
    out: set[str] = set()
    if not PAPER_RIVER_DIR.exists():
        return out
    for f in PAPER_RIVER_DIR.glob("*.org"):
        if f.stem.endswith("_en"):
            continue  # translation, not the primary
        m = _ID_RE.search(f.stem)
        if m:
            out.add(f"{m.group(1)}.{m.group(2)}")
    return out


def surfaced_ids() -> set[str]:
    """Return arxiv IDs of every paper in any summarized JSON that is not
    hard_gated (i.e. would surface in a digest)."""
    out: set[str] = set()
    if not SUMMARIZED_DIR.exists():
        return out
    for f in sorted(SUMMARIZED_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for p in data:
            bd = p.get("relevance_breakdown") or {}
            if bd.get("hard_gate"):
                continue
            pid = (p.get("id") or "").split(":")[-1].strip()
            if _ID_RE.fullmatch(pid):
                out.add(pid)
    return out


@click.command()
@click.option("--dry-run", is_flag=True, help="List candidates and exit without invoking the skill.")
@click.option("--no-warn-sleep", is_flag=True, help="Skip the 10-second 'about to start' warning sleep.")
def main(dry_run: bool, no_warn_sleep: bool) -> None:
    if not GEN_SCRIPT.exists():
        sys.exit(f"missing helper script: {GEN_SCRIPT}")

    cap = int(os.environ.get("PAPER_RIVER_MAX", "0"))
    done = existing_ids()
    surf = surfaced_ids()
    todo = sorted(surf - done)
    if cap > 0 and len(todo) > cap:
        todo = todo[:cap]

    print(f"auto_paper_river: surfaced={len(surf)}, already done={len(done)}, "
          f"to generate={len(todo)}"
          + (f" (capped from {len(surf - done)} via PAPER_RIVER_MAX={cap})" if cap > 0 else ""))

    if not todo:
        print("auto_paper_river: nothing to do")
        return

    if dry_run:
        for arxiv_id in todo:
            print(f"  would generate: {arxiv_id}")
        return

    # Sanity warning when the batch is large — first runs without a cap
    # can easily be 300+ papers × 5-10 min. Skippable for cron via
    # --no-warn-sleep so the daily pipeline doesn't waste 10s per night.
    if len(todo) > 5 and not no_warn_sleep:
        print(f"\n  WARNING: {len(todo)} paper-rivers × ~5-10 min each ≈ "
              f"{len(todo) * 7 // 60} hours of runtime.")
        print(f"  WARNING: each call uses Claude API + web fetches. Set PAPER_RIVER_MAX=N to cap.")
        print(f"  WARNING: starting in 10s — Ctrl-C to abort.\n")
        time.sleep(10)

    ok = 0
    failed = 0
    for i, arxiv_id in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] generating paper-river for {arxiv_id}")
        try:
            r = subprocess.run(
                [str(GEN_SCRIPT), arxiv_id],
                cwd=str(ROOT),
                check=False,
            )
            if r.returncode == 0:
                ok += 1
            else:
                failed += 1
                print(f"  gen_paper_river exit={r.returncode} (continuing)")
        except Exception as e:
            failed += 1
            print(f"  exception: {type(e).__name__}: {e}")

    print(f"\nauto_paper_river: done ({ok} ok, {failed} failed)")


if __name__ == "__main__":
    main()
