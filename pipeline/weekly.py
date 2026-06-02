"""7-day rollup digest as a single compact table; emits weekly/YYYYMMDD-YYYYMMDD.md and splices the same table into README.md (between the WEEKLY markers)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pipeline._clock import today_utc
from pipeline.config import load_config
from pipeline.render import _compact_row, _splice_weekly_into_readme, sort_papers
from sources.base import Paper


def _passed_gate(p: Paper) -> bool:
    if p.relevance_score is None:
        return False
    bd = p.relevance_breakdown or {}
    return not bd.get("hard_gate", False)


def _render_table(
    sorted_papers: list[Paper],
    digest_date_by_id: dict[str, str],
    start_date: datetime,
    end_date: datetime,
    digest_prefix: str,
) -> str:
    """Build the compact weekly table. `digest_prefix` is the path to the
    digests/ dir relative to wherever the table will live (`../digests/` for
    the weekly/ file, `digests/` for the repo-root README)."""
    body: list[str] = []
    body.append(f"# Weekly Digest · {start_date.date()} → {end_date.date()}\n")
    body.append(f"> Surfaced: {len(sorted_papers)} papers\n")
    body.append("| # | Bucket | Paper | Authors | Date | Why |")
    body.append("|---|--------|-------|---------|------|-----|")
    for i, p in enumerate(sorted_papers, start=1):
        digest_link = f"{digest_prefix}{digest_date_by_id[p.id]}.md"
        body.append(_compact_row(i, p, digest_link))
    return "\n".join(body) + "\n"


def render_weekly(
    end_date: datetime,
    summarized_root: Path,
    out_dir: Path,
    readme_path: Path | None = None,
) -> None:
    # Track each paper's best score AND the digest date it came from, so the
    # Why-column link points to the correct daily digest file.
    by_id: dict[str, tuple[Paper, str]] = {}
    for d in range(7):
        date = end_date - timedelta(days=d)
        date_str = date.strftime("%Y-%m-%d")
        f = summarized_root / f"{date_str}.json"
        if not f.exists():
            continue
        for item in json.loads(f.read_text()):
            p = Paper.model_validate(item)
            prev = by_id.get(p.id)
            if not prev or (p.relevance_score or 0) > (prev[0].relevance_score or 0):
                by_id[p.id] = (p, date_str)

    surviving = [(p, d) for p, d in by_id.values() if _passed_gate(p)]
    digest_date_by_id = {p.id: d for p, d in surviving}
    sorted_papers = sort_papers([p for p, _ in surviving])

    start_date = end_date - timedelta(days=6)
    fname = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.md"

    # weekly/<file>.md → ../digests/<date>.md
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_text(
        _render_table(
            sorted_papers, digest_date_by_id, start_date, end_date, "../digests/"
        )
    )

    # README at repo root → digests/<date>.md
    if readme_path is not None:
        _splice_weekly_into_readme(
            readme_path,
            _render_table(
                sorted_papers, digest_date_by_id, start_date, end_date, "digests/"
            ),
        )


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--end-date", default=None)
    @click.option("--in-root", default="data/summarized", type=click.Path(path_type=Path))
    @click.option("--out-dir", default="weekly", type=click.Path(path_type=Path))
    @click.option("--readme", default="README.md", type=click.Path(path_type=Path))
    def main(end_date, in_root, out_dir, readme):
        load_config()
        end = (
            datetime.fromisoformat(end_date).replace(tzinfo=UTC)
            if end_date
            else today_utc()
        )
        render_weekly(end, in_root, out_dir, readme_path=readme)
        start = end - timedelta(days=6)
        print(f"weekly: digest written for {start.date()} → {end.date()}")

    main()
