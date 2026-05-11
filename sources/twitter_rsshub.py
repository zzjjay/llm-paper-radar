from __future__ import annotations

import os
from datetime import datetime

import feedparser
import httpx

from sources._arxiv_lookup import extract_arxiv_ids, fetch_arxiv_by_ids
from sources.base import Paper, Source


class TwitterRSSHubSource(Source):
    name = "twitter_rsshub"

    def __init__(self, accounts: list[str]):
        self.accounts = accounts

    async def fetch(self, target_date: datetime) -> list[Paper]:
        base = os.environ.get("RSSHUB_BASE_URL", "").rstrip("/")
        if not base:
            return []

        id_to_accounts: dict[str, set[str]] = {}
        async with httpx.AsyncClient(timeout=20.0) as client:
            for acc in self.accounts:
                try:
                    r = await client.get(f"{base}/twitter/user/{acc}")
                    if r.status_code != 200:
                        continue
                    feed = feedparser.parse(r.text)
                except httpx.HTTPError:
                    continue
                for item in feed.entries:
                    blob = " ".join([
                        item.get("title", ""),
                        item.get("description", ""),
                        item.get("summary", ""),
                    ])
                    for aid in extract_arxiv_ids(blob):
                        id_to_accounts.setdefault(aid, set()).add(acc)

        if not id_to_accounts:
            return []

        extras_per_id = {aid: {"accounts": sorted(accs)} for aid, accs in id_to_accounts.items()}
        return await fetch_arxiv_by_ids(
            list(id_to_accounts.keys()), source_name="twitter_rsshub", extras_per_id=extras_per_id
        )


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
        if not cfg.sources.twitter_rsshub.enabled:
            print("twitter_rsshub disabled")
            return
        src = TwitterRSSHubSource(accounts=cfg.sources.twitter_rsshub.accounts)
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            try:
                papers = asyncio.run(src.fetch(target))
            except Exception as e:
                print(f"twitter_rsshub: skip {target.date()}: {e}")
                continue
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "twitter_rsshub.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"twitter_rsshub: wrote {len(papers)} for {target.date()}")

    main()
