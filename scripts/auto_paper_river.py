"""Scan surfaced papers and generate paper-river for those without one.

A "surfaced paper" is any paper in `data/summarized/*.json` whose
`relevance_breakdown.hard_gate` is false — i.e. every paper that actually
appears in some digest/<date>.md.

By default the scan is scoped to the **current rollup window** (last
`--window-days` days, default 2 to match daily.sh's default rollup), so
the script's notion of "surfaced" matches what just got rendered into
the README's latest table. Pass `--all-history` to revisit every
summarized JSON ever written (useful for one-off backfills).

Dedup: a paper is considered "done" if `paper-river/*<id>.org` exists in
either dot form (`...2604.18556.org`) or legacy dash form
(`...2604-18556.org`). The `_en.org` siblings are NOT counted as the
primary file — they are translations of the zh original (see
scripts/translate_paper_river.py).

Per-run cap: `PAPER_RIVER_MAX` env var. 0 (default) means unlimited.
In window mode the candidate set is usually small (single digits per
day); in --all-history mode it can be 300+ candidates × 5-10 minutes
each = days of work.

The actual generation work is delegated to scripts/gen_paper_river.sh,
which invokes the ljg-paper-river Claude Code skill in headless mode.
Per-paper failure is logged and does not abort the batch.

Usage:
    uv run python scripts/auto_paper_river.py                 # current 2-day window
    uv run python scripts/auto_paper_river.py --window-days 7 # current 7-day window
    uv run python scripts/auto_paper_river.py --all-history   # legacy: every surfaced paper ever
    PAPER_RIVER_MAX=2 uv run python scripts/auto_paper_river.py   # cap at 2 this run
    uv run python scripts/auto_paper_river.py --dry-run       # list what would run
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Make the project's `pipeline` package importable when this script is
# invoked directly (uv run python scripts/auto_paper_river.py only adds
# scripts/ to sys.path).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import click

from pipeline._clock import today_utc

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


def _window_paths(window_days: int, today: datetime | None = None) -> list[Path]:
    """Return data/summarized/<date>.json paths for the last `window_days`
    days (today inclusive), newest first. Missing files are skipped.

    "today" honors RADAR_DAY_OFFSET (set to 1 by daily.sh so the cron
    treats yesterday-UTC as the canonical day)."""
    if today is None:
        today = today_utc()
    out: list[Path] = []
    for d in range(window_days):
        date = today - timedelta(days=d)
        p = SUMMARIZED_DIR / f"{date.strftime('%Y-%m-%d')}.json"
        if p.exists():
            out.append(p)
    return out


def surfaced_ids(window_days: int | None = None) -> set[str]:
    """Return arxiv IDs of non-hard-gated papers from summarized JSON.

    `window_days=None` → scan every file in SUMMARIZED_DIR (legacy
    behavior, used by --all-history backfills). Otherwise scope to the
    last N days, matching the daily.sh rollup window.
    """
    out: set[str] = set()
    if not SUMMARIZED_DIR.exists():
        return out
    if window_days is None:
        targets = sorted(SUMMARIZED_DIR.glob("*.json"))
    else:
        targets = _window_paths(window_days)
    for f in targets:
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
@click.option(
    "--window-days",
    default=2,
    type=int,
    help="How many recent days of summarized JSON to scan. Default 2 to match"
    " daily.sh's rollup. Ignored when --all-history is set.",
)
@click.option(
    "--all-history",
    is_flag=True,
    help="Scan every summarized JSON ever written (legacy behavior). Use for"
    " one-off backfills of historical surfaced papers.",
)
@click.option("--dry-run", is_flag=True, help="List candidates and exit without invoking the skill.")
@click.option("--no-warn-sleep", is_flag=True, help="Skip the 10-second 'about to start' warning sleep.")
def main(window_days: int, all_history: bool, dry_run: bool, no_warn_sleep: bool) -> None:
    if not GEN_SCRIPT.exists():
        sys.exit(f"missing helper script: {GEN_SCRIPT}")

    cap = int(os.environ.get("PAPER_RIVER_MAX", "0"))
    done = existing_ids()
    surf = surfaced_ids(window_days=None if all_history else window_days)
    todo = sorted(surf - done)
    if cap > 0 and len(todo) > cap:
        todo = todo[:cap]

    scope = "all history" if all_history else f"last {window_days}d"
    print(f"auto_paper_river: scope={scope}, surfaced={len(surf)}, already done={len(done)}, "
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
