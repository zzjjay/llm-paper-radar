from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pipeline.config import Config, load_config
from sources.base import Paper


def merge_papers(papers: list[Paper], source_priority: list[str]) -> list[Paper]:
    """Merge papers by id; for shared fields, take from the highest-priority source.
    `source_priority` is most-trusted-first."""
    by_id: dict[str, list[Paper]] = {}
    for p in papers:
        by_id.setdefault(p.id, []).append(p)

    rank = {name: i for i, name in enumerate(source_priority)}

    def primary_source_rank(p: Paper) -> int:
        return rank.get(p.sources[0].name, 999)

    merged: list[Paper] = []
    for group in by_id.values():
        group_sorted = sorted(group, key=primary_source_rank)
        head = group_sorted[0].model_copy(deep=True)
        all_sources = [g.sources[0] for g in group]
        head.sources = all_sources
        for other in group_sorted[1:]:
            for fld in ("title", "abstract", "pdf_url", "code_url", "code_meta"):
                if not getattr(head, fld) and getattr(other, fld):
                    setattr(head, fld, getattr(other, fld))
            if not head.authors and other.authors:
                head.authors = other.authors
            if not head.categories and other.categories:
                head.categories = other.categories
                head.primary_category = other.primary_category
        merged.append(head)
    return merged


def _load_raw(raw_dir: Path) -> list[Paper]:
    papers: list[Paper] = []
    if not raw_dir.exists():
        return papers
    for f in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for item in data:
            papers.append(Paper.model_validate(item))
    return papers


def dedupe_for_date(
    date: datetime,
    raw_root: Path,
    out_path: Path,
    seen_path: Path,
    config: Config,
) -> int:
    raw_dir = raw_root / date.strftime("%Y-%m-%d")
    papers = _load_raw(raw_dir)
    merged = merge_papers(papers, config.dedupe.source_priority)

    seen: set[str] = set()
    if seen_path.exists():
        seen = set(json.loads(seen_path.read_text()))
    for p in merged:
        if p.id in seen:
            p.seen_before = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([p.model_dump(mode="json") for p in merged], indent=2))

    seen.update(p.id for p in merged)
    seen_path.write_text(json.dumps(sorted(seen), indent=2))

    return len(merged)


if __name__ == "__main__":
    from datetime import UTC, timedelta

    import click

    @click.command()
    @click.option("--date", default=None, help="YYYY-MM-DD; default today UTC")
    @click.option("--backfill-days", default=0, type=int, help="Process today + N days back. Default 0 = today only. Each day is fetched/processed independently.")
    @click.option("--raw-root", default="data/raw", type=click.Path(path_type=Path))
    @click.option("--out-root", default="data/deduped", type=click.Path(path_type=Path))
    @click.option("--seen-path", default="data/seen.json", type=click.Path(path_type=Path))
    def main(date, backfill_days, raw_root, out_root, seen_path):
        cfg = load_config()
        if date:
            base = datetime.fromisoformat(date).replace(tzinfo=UTC)
        else:
            base = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            if backfill_days > 0 and (
                Path("digests") / f"{target.strftime('%Y-%m-%d')}.md"
            ).exists():
                print(f"dedupe: skip {target.date()} (digest exists)")
                continue
            out = out_root / f"{target.strftime('%Y-%m-%d')}.json"
            n = dedupe_for_date(target, raw_root, out, seen_path, cfg)
            print(f"dedupe: {n} unique papers for {target.date()}")

    main()
