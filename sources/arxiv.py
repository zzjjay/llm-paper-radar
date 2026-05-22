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
    # Switch from descending-pagination to date-range query when the target
    # day is older than this. 5 days matches the ~1000-paper-page-cap reach
    # of cs.CL/cs.LG/cs.AR fetches in May 2026; bump if arXiv volume drops.
    RECENT_DAYS = 5

    def __init__(self, categories: list[str], page_size: int = 200, max_pages: int = 5):
        self.categories = categories
        self.page_size = page_size
        self.max_pages = max_pages

    async def fetch(self, target_date: datetime) -> list[Paper]:
        """Fetch papers submitted within ~24h ending at target_date (UTC).

        Uses two query strategies depending on how recent `target_date` is:
        - **Recent (≤ RECENT_DAYS old)**: descending-by-submittedDate pagination
          from "now", early-break when the page is older than the window. Cheap
          and matches the cron's daily usage pattern.
        - **Old (> RECENT_DAYS)**: explicit `submittedDate:[start TO end]`
          range query, so the fetch lands on the correct historical window
          regardless of how much later we run it. Critical for backfills —
          without this, sub-1000-paper-page-cap means anything older than
          ~5 days returns 0 papers.
        """
        now = datetime.now(UTC)
        age = (now - target_date).total_seconds() / 86400
        cat_query = "+OR+".join(f"cat:{c}" for c in self.categories)
        all_entries: list[dict] = []
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            start = 0
            for _ in range(self.max_pages):
                if age > self.RECENT_DAYS:
                    # Historical backfill: pin the query to the exact 24h
                    # window so we don't depend on pagination depth.
                    ws = target_date - timedelta(hours=24)
                    we = target_date + timedelta(hours=24)
                    range_q = (
                        f"submittedDate:[{ws.strftime('%Y%m%d%H%M')}+TO+"
                        f"{we.strftime('%Y%m%d%H%M')}]"
                    )
                    full_query = f"({cat_query})+AND+{range_q}"
                else:
                    full_query = cat_query
                url = (
                    f"{self.BASE_URL}?search_query={full_query}&start={start}"
                    f"&max_results={self.page_size}"
                    f"&sortBy=submittedDate&sortOrder=descending"
                )
                # arXiv occasionally 429s shared CI IPs and intermittently times
                # out / drops the connection. Back off and retry on both.
                resp = None
                last_exc: Exception | None = None
                for attempt in range(4):
                    try:
                        resp = await client.get(url)
                        if resp.status_code != 429:
                            last_exc = None
                            break
                        wait = 5 * (2**attempt)  # 5s, 10s, 20s, 40s
                        print(f"arxiv: 429 on page start={start}, sleeping {wait}s before retry")
                    except (httpx.TimeoutException, httpx.TransportError) as e:
                        last_exc = e
                        wait = 5 * (2**attempt)
                        print(
                            f"arxiv: {type(e).__name__} on page start={start}, "
                            f"sleeping {wait}s before retry"
                        )
                    await asyncio.sleep(wait)
                if last_exc is not None:
                    raise last_exc
                assert resp is not None
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

        is_historical = age > self.RECENT_DAYS
        return list(self._to_papers(all_entries, target_date, is_historical))

    def _to_papers(
        self,
        entries: list[dict],
        target_date: datetime,
        is_historical: bool = False,
    ) -> Iterable[Paper]:
        # Recent (cron): window = [target - 24h, target + 1h]. Matches the
        # "noon UTC cron grabs yesterday's papers" pattern: target = today
        # 00:00, fetcher grabs papers from [yesterday 00:00, today 01:00).
        # The trailing +1h cushion is so a cron started a bit late still
        # gets that early-morning trickle. Folder convention: papers fetched
        # on day D land in folder D, even if their publication date is D-1.
        #
        # Historical (backfill): window = [target 00:00, target+1 00:00).
        # When a user asks for "papers from 4/20", they mean papers
        # *published on 4/20*, not papers ending at 4/20 00:00 (which would
        # be 4/19's batch). This is the intuitive semantics for backfill;
        # we accept the asymmetry with recent-mode to keep cron behavior
        # unchanged.
        if is_historical:
            window_start = target_date
            window_end = target_date + timedelta(hours=24)
        else:
            window_start = target_date - timedelta(hours=24)
            window_end = target_date + timedelta(hours=1)
        now = datetime.now(UTC)
        for e in entries:
            pub = _entry_published(e)
            if not (window_start <= pub <= window_end):
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
            if backfill_days > 0 and (
                Path("digests") / f"{target.strftime('%Y-%m-%d')}.md"
            ).exists():
                print(f"arxiv: skip {target.date()} (digest exists)")
                continue
            papers = asyncio.run(src.fetch(target))
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            out_path = day_dir / "arxiv.json"
            # Defensive: a fetch that returns 0 papers must NOT overwrite an
            # existing non-empty arxiv.json. Without this guard, backfills
            # that fail to reach old days (e.g. the source's pagination depth
            # was too shallow) silently destroy historical data. Either a
            # real "no papers today" outcome is rare for cs.CL/cs.LG/cs.AR,
            # or a failure mode worth a loud WARN.
            if not papers and out_path.exists() and out_path.stat().st_size > 2:
                print(
                    f"arxiv: WARN skip write for {target.date()} — fetch returned 0 papers "
                    f"but existing file has data ({out_path.stat().st_size} bytes). "
                    f"Refusing to clobber. Investigate upstream."
                )
                continue
            out_path.write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"arxiv: wrote {len(papers)} papers for {target.date()}")

    main()
