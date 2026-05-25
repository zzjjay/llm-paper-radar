"""7-day rollup digest as a single compact table; emits weekly/YYYYMMDD-YYYYMMDD.md (GitHub Actions weekly.yml only)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pipeline.config import load_config
from pipeline.render import _compact_row, sort_papers
from sources.base import Paper


def _passed_gate(p: Paper) -> bool:
    if p.relevance_score is None:
        return False
    bd = p.relevance_breakdown or {}
    return not bd.get("hard_gate", False)


def render_weekly(
    end_date: datetime,
    summarized_root: Path,
    out_dir: Path,
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

    body: list[str] = []
    body.append(f"# Weekly Digest · {start_date.date()} → {end_date.date()}\n")
    body.append(f"> Surfaced: {len(sorted_papers)} papers\n")
    body.append("| # | Bucket | Paper | Authors | Date | Why |")
    body.append("|---|--------|-------|---------|------|-----|")
    for i, p in enumerate(sorted_papers, start=1):
        # weekly/<file>.md → ../digests/<date>.md
        digest_link = f"../digests/{digest_date_by_id[p.id]}.md"
        body.append(_compact_row(i, p, digest_link))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--end-date", default=None)
    @click.option("--in-root", default="data/summarized", type=click.Path(path_type=Path))
    @click.option("--out-dir", default="weekly", type=click.Path(path_type=Path))
    def main(end_date, in_root, out_dir):
        load_config()
        end = (
            datetime.fromisoformat(end_date).replace(tzinfo=UTC)
            if end_date
            else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        )
        render_weekly(end, in_root, out_dir)
        start = end - timedelta(days=6)
        print(f"weekly: digest written for {start.date()} → {end.date()}")

    main()
