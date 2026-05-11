from __future__ import annotations

import re
from datetime import datetime

import feedparser
import httpx

from sources._arxiv_lookup import extract_arxiv_ids, fetch_arxiv_by_ids
from sources.base import Paper, Source

GITHUB_RE = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+")


class PapersWithCodeSource(Source):
    name = "papers_with_code"
    RSS_URL = "https://paperswithcode.com/latest/rss.xml"

    async def fetch(self, target_date: datetime) -> list[Paper]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(self.RSS_URL)
            r.raise_for_status()
        feed = feedparser.parse(r.text)

        id_to_code: dict[str, str] = {}
        for item in feed.entries:
            blob = " ".join([item.get("title", ""), item.get("description", "")])
            ids = extract_arxiv_ids(blob)
            code_match = GITHUB_RE.search(blob)
            code_url = code_match.group(0) if code_match else None
            for aid in ids:
                id_to_code.setdefault(aid, code_url)

        papers = await fetch_arxiv_by_ids(list(id_to_code.keys()), source_name="papers_with_code")
        for p in papers:
            if id_to_code.get(p.id):
                p.code_url = id_to_code[p.id]
        return papers


if __name__ == "__main__":
    import asyncio
    import json
    from datetime import UTC, timedelta
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.papers_with_code.enabled:
            print("papers_with_code disabled")
            return
        src = PapersWithCodeSource()
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            try:
                papers = asyncio.run(src.fetch(target))
            except Exception as e:
                print(f"papers_with_code: skip {target.date()}: {e}")
                continue
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "papers_with_code.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"papers_with_code: wrote {len(papers)} for {target.date()}")

    main()
