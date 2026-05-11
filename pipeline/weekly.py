from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pipeline.config import load_config
from pipeline.render import sort_papers
from sources.base import Paper


def render_weekly(
    end_date: datetime,
    summarized_root: Path,
    out_dir: Path,
    top_n: int,
    threshold: int,
) -> None:
    all_papers: list[Paper] = []
    for d in range(7):
        date = end_date - timedelta(days=d)
        f = summarized_root / f"{date.strftime('%Y-%m-%d')}.json"
        if not f.exists():
            continue
        for item in json.loads(f.read_text()):
            all_papers.append(Paper.model_validate(item))

    by_id: dict[str, Paper] = {}
    for p in all_papers:
        prev = by_id.get(p.id)
        if not prev or (p.relevance_score or 0) > (prev.relevance_score or 0):
            by_id[p.id] = p
    surviving = [p for p in by_id.values() if (p.relevance_score or 0) >= threshold]
    surviving = sort_papers(surviving)[:top_n]

    src_counts = Counter(s.name for p in surviving for s in p.sources)

    iso = end_date.isocalendar()
    fname = f"{iso.year}-W{iso.week:02d}.md"
    body = []
    body.append(f"# 周报 · Week {iso.week} of {iso.year} (ending {end_date.date()})\n")
    body.append(f"## Top {len(surviving)} (本周精华)\n")
    for i, p in enumerate(surviving, start=1):
        body.append(f"### {i}. [{p.title}]({p.url}) ({p.relevance_score}/10)")
        if p.summary_zh:
            body.append(f"\n{p.summary_zh}\n")
        if p.summary_en:
            body.append(f"\n*{p.summary_en}*\n")

    body.append("\n## 来源贡献 / Per-source contribution\n")
    body.append("| Source | Count |")
    body.append("|--------|-------|")
    for src, cnt in src_counts.most_common():
        body.append(f"| {src} | {cnt} |")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_text("\n".join(body))


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--end-date", default=None)
    @click.option("--in-root", default="data/summarized", type=click.Path(path_type=Path))
    @click.option("--out-dir", default="weekly", type=click.Path(path_type=Path))
    def main(end_date, in_root, out_dir):
        cfg = load_config()
        end = (
            datetime.fromisoformat(end_date).replace(tzinfo=UTC)
            if end_date
            else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        )
        render_weekly(end, in_root, out_dir, top_n=20, threshold=cfg.filter.threshold)
        print(f"weekly: digest written for week ending {end.date()}")

    main()
