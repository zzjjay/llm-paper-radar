"""Ad-hoc compact-table rollup for an arbitrary date window.

Reads `data/summarized/YYYY-MM-DD.json` for each day in the window, applies
the current scoring/bucket state, and writes a single merged compact table
(same format as the README LATEST block) to stdout or a file.

Does NOT touch README.md, INDEX.md, or digests/ — purely a query view.
Useful when you want a paper list for a non-7-day window (e.g. 30-day,
or a specific bug-bash range) without polluting the long-term archive.

Examples:
  uv run python -m pipeline.rollup --days 30
  uv run python -m pipeline.rollup --start 2026-05-01 --end 2026-05-21
  uv run python -m pipeline.rollup --days 14 --output /tmp/last-14d.md
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from pipeline.render import _compute_day, _render_aggregated_compact_md
from sources.base import Paper


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def collect_window(
    start: datetime,
    end: datetime,
    in_root: Path,
) -> list[tuple[datetime, int, list[Paper], list[Paper]]]:
    """Walk the window (inclusive) and load whatever per-day summarized
    files exist. Missing days are skipped silently — the rendered table
    just won't have rows from those days."""
    days: list[tuple[datetime, int, list[Paper], list[Paper]]] = []
    cursor = start
    while cursor <= end:
        in_path = in_root / f"{cursor.strftime('%Y-%m-%d')}.json"
        if in_path.exists():
            scanned, watched, surviving = _compute_day(in_path)
            days.append((cursor, scanned, watched, surviving))
        cursor += timedelta(days=1)
    return days


@click.command()
@click.option("--start", default=None, help="Inclusive start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="Inclusive end date (YYYY-MM-DD). Defaults to today UTC.")
@click.option(
    "--days",
    type=int,
    default=None,
    help="Window length ending at --end (or today). Mutually exclusive with --start.",
)
@click.option(
    "--in-root",
    default="data/summarized",
    type=click.Path(path_type=Path),
    help="Directory of per-day summarized JSONs.",
)
@click.option(
    "--digests-dir",
    default="digests",
    type=click.Path(path_type=Path),
    help="Used only to build relative paper links in the table.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Output markdown file. Default: stdout.",
)
def main(
    start: str | None,
    end: str | None,
    days: int | None,
    in_root: Path,
    digests_dir: Path,
    output: Path | None,
) -> None:
    end_dt = _parse_date(end) if end else datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    if start and days is not None:
        raise click.UsageError("Pass either --start or --days, not both.")
    if start:
        start_dt = _parse_date(start)
    elif days is not None:
        if days < 1:
            raise click.UsageError("--days must be >= 1.")
        start_dt = end_dt - timedelta(days=days - 1)
    else:
        start_dt = end_dt - timedelta(days=6)  # default = 7-day window

    if start_dt > end_dt:
        raise click.UsageError(
            f"--start ({start_dt.date()}) is after --end ({end_dt.date()})."
        )

    window = collect_window(start_dt, end_dt, in_root)
    if not window:
        text = (
            f"# LLM Inference Optimization Rollup · "
            f"{start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}\n\n"
            "_No summarized data found in this window._\n"
        )
    else:
        text = _render_aggregated_compact_md(window, digests_dir)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        click.echo(f"rollup: wrote {output}", err=True)
    else:
        click.echo(text)


if __name__ == "__main__":
    main()
