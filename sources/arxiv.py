from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import feedparser
import httpx

from sources.base import Paper, Source, SourceRecord

ARXIV_ID_RE = re.compile(r"abs/([\d.]+)(?:v\d+)?$")


class ArxivSource(Source):
    name = "arxiv"
    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, categories: list[str], page_size: int = 200, max_pages: int = 5):
        self.categories = categories
        self.page_size = page_size
        self.max_pages = max_pages

    async def fetch(self, target_date: datetime) -> list[Paper]:
        """Fetch papers submitted within ~24h ending at target_date (UTC)."""
        # arXiv expects literal '+' between OR'd terms. httpx's params= encoder
        # turns '+' into '%2B', which arXiv treats as part of a literal string
        # match — the query then matches nothing. Build the URL by hand instead.
        cat_query = "+OR+".join(f"cat:{c}" for c in self.categories)
        all_entries: list[dict] = []
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            start = 0
            for _ in range(self.max_pages):
                url = (
                    f"{self.BASE_URL}?search_query={cat_query}&start={start}"
                    f"&max_results={self.page_size}"
                    f"&sortBy=submittedDate&sortOrder=descending"
                )
                # arXiv occasionally 429s shared CI IPs. Back off and retry a few times.
                for attempt in range(4):
                    resp = await client.get(url)
                    if resp.status_code != 429:
                        break
                    wait = 5 * (2**attempt)  # 5s, 10s, 20s, 40s
                    print(f"arxiv: 429 on page start={start}, sleeping {wait}s before retry")
                    await asyncio.sleep(wait)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
                if not feed.entries:
                    break
                all_entries.extend(feed.entries)
                start += self.page_size
                # Stop early if oldest entry on page is older than window
                oldest = _entry_published(feed.entries[-1])
                if oldest < target_date - timedelta(days=2):
                    break
                await asyncio.sleep(3.0)  # arXiv rate limit guidance

        return list(self._to_papers(all_entries, target_date))

    def _to_papers(self, entries: list[dict], target_date: datetime) -> Iterable[Paper]:
        window_start = target_date - timedelta(hours=24)
        now = datetime.now(UTC)
        for e in entries:
            pub = _entry_published(e)
            if not (window_start <= pub <= target_date + timedelta(hours=1)):
                continue
            arxiv_id = _extract_arxiv_id(e.get("id", ""))
            if not arxiv_id:
                continue
            categories = [t["term"] for t in e.get("tags", [])]
            primary = categories[0] if categories else "unknown"
            pdf_link = next(
                (lnk["href"] for lnk in e.get("links", []) if lnk.get("type") == "application/pdf"),
                None,
            )
            yield Paper(
                id=arxiv_id,
                title=e.get("title", "").strip().replace("\n ", " "),
                authors=[a.get("name", "") for a in e.get("authors", [])],
                abstract=e.get("summary", "").strip(),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_link or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=pub,
                primary_category=primary,
                categories=categories,
                sources=[SourceRecord(name="arxiv", fetched_at=now)],
            )


def _extract_arxiv_id(url: str) -> str | None:
    m = ARXIV_ID_RE.search(url)
    return m.group(1) if m else None


def _entry_published(entry: dict) -> datetime:
    s = entry.get("published") or entry.get("updated")
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


if __name__ == "__main__":
    import json
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.arxiv.enabled:
            print("arxiv source disabled")
            return
        src = ArxivSource(categories=cfg.sources.arxiv.categories)
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            papers = asyncio.run(src.fetch(target))
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "arxiv.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"arxiv: wrote {len(papers)} papers for {target.date()}")

    main()
