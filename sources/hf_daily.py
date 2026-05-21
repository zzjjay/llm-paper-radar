from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx

from sources._arxiv_lookup import fetch_arxiv_by_ids
from sources.base import Paper, Source, SourceRecord

TRENDING_LINK_RE = re.compile(r'href="/papers/(\d{4}\.\d{4,5})"')


def parse_trending_ranks(html: str) -> dict[str, int]:
    """Extract arXiv IDs from the trending page HTML in order of appearance.
    Returns mapping arxiv_id -> rank (1-indexed, lowest = hottest)."""
    seen: dict[str, int] = {}
    for m in TRENDING_LINK_RE.finditer(html):
        aid = m.group(1)
        if aid not in seen:
            seen[aid] = len(seen) + 1
    return seen


class HFDailySource(Source):
    name = "hf_daily"
    DAILY_URL = "https://huggingface.co/api/daily_papers"
    TRENDING_URL = "https://huggingface.co/papers/trending"

    async def fetch(self, target_date: datetime) -> list[Paper]:
        date_str = target_date.strftime("%Y-%m-%d")
        async with httpx.AsyncClient(
            timeout=30.0, headers={"User-Agent": "Mozilla/5.0 llm-paper-radar"}
        ) as client:
            daily_resp = await client.get(self.DAILY_URL, params={"date": date_str})
            daily_resp.raise_for_status()
            daily_items = daily_resp.json()

            trending_ranks: dict[str, int] = {}
            try:
                tr_resp = await client.get(self.TRENDING_URL)
                if tr_resp.status_code == 200:
                    trending_ranks = parse_trending_ranks(tr_resp.text)
            except httpx.HTTPError as e:
                print(f"hf_daily: trending fetch failed ({e}); continuing with daily only")

        now = datetime.now(UTC)
        papers: dict[str, Paper] = {}

        for item in daily_items:
            paper_obj = item.get("paper", {})
            arxiv_id = paper_obj.get("id") or item.get("id")
            if not arxiv_id:
                continue
            published_str = paper_obj.get("publishedAt") or item.get("publishedAt")
            try:
                published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except (AttributeError, ValueError):
                published_at = target_date

            extras = {
                "upvotes": paper_obj.get("upvotes", item.get("upvotes", 0)),
                "num_comments": paper_obj.get("numComments", item.get("numComments", 0)),
            }

            papers[arxiv_id] = Paper(
                id=arxiv_id,
                title=paper_obj.get("title", item.get("title", "")).strip(),
                authors=[a.get("name", "") for a in paper_obj.get("authors", [])],
                abstract=paper_obj.get("summary", item.get("summary", "")).strip(),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=published_at,
                primary_category="cs.CL",
                categories=[],
                sources=[SourceRecord(name="hf_daily", fetched_at=now, extras=extras)],
            )

        # IDs in trending that we already have full metadata for: just record rank.
        missing_ids: list[str] = []
        for arxiv_id, rank in trending_ranks.items():
            tr_record = SourceRecord(
                name="hf_daily", fetched_at=now, extras={"trending_rank": rank}
            )
            if arxiv_id in papers:
                papers[arxiv_id].sources.append(tr_record)
            else:
                missing_ids.append(arxiv_id)

        # Enrich trending-only IDs via arXiv lookup so we don't emit blank stubs
        # (the filter would just drop them, producing an empty digest — common on
        # weekends when HF's daily_papers endpoint returns nothing).
        if missing_ids:
            extras_per_id = {aid: {"trending_rank": trending_ranks[aid]} for aid in missing_ids}
            try:
                enriched = await fetch_arxiv_by_ids(
                    missing_ids, source_name="hf_daily", extras_per_id=extras_per_id
                )
            except httpx.HTTPError as e:
                print(
                    f"hf_daily: arxiv lookup for {len(missing_ids)} trending IDs failed ({e});"
                    f" dropping those entries"
                )
                enriched = []
            for p in enriched:
                papers[p.id] = p

        return list(papers.values())


if __name__ == "__main__":
    import asyncio
    import json
    from datetime import timedelta
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.hf_daily.enabled:
            print("hf_daily source disabled")
            return
        src = HFDailySource()
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            if backfill_days > 0 and (
                Path("digests") / f"{target.strftime('%Y-%m-%d')}.md"
            ).exists():
                print(f"hf_daily: skip {target.date()} (digest exists)")
                continue
            papers = asyncio.run(src.fetch(target))
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "hf_daily.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"hf_daily: wrote {len(papers)} papers for {target.date()}")

    main()
