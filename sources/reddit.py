from __future__ import annotations

import os
from datetime import datetime

import httpx

from sources._arxiv_lookup import extract_arxiv_ids, fetch_arxiv_by_ids
from sources.base import Paper, Source

USER_AGENT = "llm-paper-radar/0.1 (by /u/zhaolin-amd)"


class RedditSource(Source):
    name = "reddit"

    def __init__(self, subreddit: str = "LocalLLaMA", top_window: str = "day"):
        self.subreddit = subreddit
        self.top_window = top_window

    async def _get_token(self, client: httpx.AsyncClient) -> str:
        cid = os.environ["REDDIT_CLIENT_ID"]
        secret = os.environ["REDDIT_CLIENT_SECRET"]
        r = await client.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        return r.json()["access_token"]

    async def fetch(self, target_date: datetime) -> list[Paper]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            token = await self._get_token(client)
            r = await client.get(
                f"https://oauth.reddit.com/r/{self.subreddit}/top.json",
                params={"t": self.top_window, "limit": 50},
                headers={"Authorization": f"bearer {token}", "User-Agent": USER_AGENT},
            )
            r.raise_for_status()
            data = r.json()

        id_to_extras: dict[str, dict] = {}
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            text_blob = " ".join([post.get("title", ""), post.get("selftext", "")])
            ids = extract_arxiv_ids(text_blob)
            for aid in ids:
                id_to_extras.setdefault(
                    aid,
                    {
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "thread_url": f"https://reddit.com{post.get('permalink', '')}",
                    },
                )

        return await fetch_arxiv_by_ids(
            list(id_to_extras.keys()), source_name="reddit", extras_per_id=id_to_extras
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
        if not cfg.sources.reddit.enabled:
            print("reddit source disabled")
            return
        src = RedditSource(
            subreddit=cfg.sources.reddit.subreddit,
            top_window=cfg.sources.reddit.top_window,
        )
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            try:
                papers = asyncio.run(src.fetch(target))
            except Exception as e:
                print(f"reddit: skip {target.date()} due to {e}")
                continue
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "reddit.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"reddit: wrote {len(papers)} papers for {target.date()}")

    main()
