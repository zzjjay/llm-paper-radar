from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import feedparser
import httpx

from sources._arxiv_lookup import ArxivEmptyResponse, arxiv_get_with_retry
from sources._arxiv_oai import fetch_via_oai
from sources.base import ARXIV_USER_AGENT, Paper, Source, SourceRecord

ARXIV_ID_RE = re.compile(r"abs/([\d.]+)(?:v\d+)?$")


class ArxivSource(Source):
    name = "arxiv"
    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, categories: list[str], page_size: int = 200, max_pages: int = 5):
        self.categories = categories
        self.page_size = page_size
        self.max_pages = max_pages

    async def fetch(self, target_date: datetime) -> list[Paper]:
        """Fetch papers whose arxiv `<published>` falls on `target_date`
        (UTC). Window: [target 00:00, target+24h 00:00).

        `target_date` is interpreted as the **arxiv publication date**.
        The fetched papers' `published_at` will all fall inside the day,
        and downstream folder/digest/README naming uses the same date,
        so everything stays consistent.

        Primary path is the `api/query` endpoint. If that endpoint fails
        (throttle exhaustion, suspected-throttle empty feed) we fall back to
        the OAI-PMH harvesting endpoint, which arxiv throttles far less
        aggressively — that is the exact failure that was zeroing out whole
        days (see scripts/log/2026-06-0*.log). A *genuine* zero day
        (opensearch:totalResults=0) returns [] from the primary without
        raising, so it does NOT trigger the fallback.
        """
        try:
            return await self._fetch_via_api(target_date)
        except (
            ArxivEmptyResponse,
            httpx.HTTPStatusError,
            httpx.TransportError,
            httpx.TimeoutException,
        ) as e:
            print(
                f"arxiv: api/query failed for {target_date.date()} "
                f"({type(e).__name__}: {e}); falling back to OAI-PMH"
            )
            papers = await fetch_via_oai(target_date, self.categories)
            print(
                f"arxiv: OAI-PMH fallback got {len(papers)} papers "
                f"for {target_date.date()}"
            )
            return papers

    async def _fetch_via_api(self, target_date: datetime) -> list[Paper]:
        cat_query = "+OR+".join(f"cat:{c}" for c in self.categories)
        ws = target_date
        we = target_date + timedelta(hours=24)
        range_q = (
            f"submittedDate:[{ws.strftime('%Y%m%d%H%M')}+TO+"
            f"{we.strftime('%Y%m%d%H%M')}]"
        )
        full_query = f"({cat_query})+AND+{range_q}"
        all_entries: list[dict] = []
        async with httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": ARXIV_USER_AGENT},
        ) as client:
            start = 0
            for _ in range(self.max_pages):
                url = (
                    f"{self.BASE_URL}?search_query={full_query}&start={start}"
                    f"&max_results={self.page_size}"
                    f"&sortBy=submittedDate&sortOrder=descending"
                )

                # arXiv 429s/503s shared CI IPs; back off + jitter and honor
                # Retry-After. Shared helper handles 429/503/Timeout/TransportError.
                #
                # On page 0, also disambiguate "200 + 0 entries":
                #   - genuine zero day (Sat/Sun UTC, or any quiet day):
                #     the Atom feed contains <opensearch:totalResults>0
                #     and we should accept the empty result, write 0
                #     papers, and let downstream steps proceed.
                #   - throttle / proxy bug serving a truncated feed:
                #     totalResults is missing or > 0 while entries == 0.
                #     Retry inside the same backoff budget; if it never
                #     recovers, raise so daily.sh's main() skips the day
                #     rather than silently overwriting with 0 papers.
                #
                # Later pages (start > 0) can legitimately be empty
                # (pagination exhausted), so the validator only runs for
                # page 0.
                def _validate_page0(resp: httpx.Response) -> None:
                    feed = feedparser.parse(resp.text)
                    if feed.entries:
                        return
                    total = feed.feed.get("opensearch_totalresults")
                    if total is not None and str(total).strip() == "0":
                        # arxiv explicitly says "this query has zero results";
                        # accept it as a real empty day.
                        print(
                            f"arxiv(page 0): 0 papers for {target_date.date()} "
                            f"(opensearch:totalResults=0; arxiv published nothing matching the query)"
                        )
                        return
                    raise ArxivEmptyResponse(
                        f"arxiv returned 0 entries on page 0 for {target_date.date()} "
                        f"with no opensearch:totalResults=0 marker — suspected throttle"
                    )

                resp = await arxiv_get_with_retry(
                    client,
                    url,
                    context=f"arxiv(page {start})",
                    validate=_validate_page0 if start == 0 else None,
                )
                feed = feedparser.parse(resp.text)
                if not feed.entries:
                    break
                all_entries.extend(feed.entries)
                start += self.page_size
                await asyncio.sleep(3.0)  # arXiv rate limit guidance

        return list(self._to_papers(all_entries, target_date))

    def _to_papers(
        self,
        entries: list[dict],
        target_date: datetime,
    ) -> Iterable[Paper]:
        # Window = [target 00:00, target+24h 00:00). All emitted papers'
        # published_at live inside this day, so folder/digest/README
        # naming stays consistent with the actual arxiv publication date.
        window_start = target_date
        window_end = target_date + timedelta(hours=24)
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
    import time
    from pathlib import Path

    import click

    from pipeline._clock import today_utc
    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int, help="Process today + N days back. Default 0 = today only. Each day is fetched/processed independently.")
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    @click.option(
        "--batch-size",
        default=10,
        type=int,
        help="Days per batch before the long inter-batch sleep. "
             "Picked so total requests per batch stays well under arxiv's soft-ban threshold.",
    )
    @click.option(
        "--batch-pause",
        default=300,
        type=int,
        help="Seconds to sleep between batches (default 300 = 5 min). "
             "Long enough for arxiv to fully cool any per-IP throttle counters.",
    )
    @click.option(
        "--day-pause",
        default=25,
        type=int,
        help="Seconds to sleep between days within a batch. "
             "25s matches _backfill_6mo.py's polite pace.",
    )
    @click.option("--force", is_flag=True, default=False, help="Re-fetch even if the day's digest already exists.")
    def main(
        backfill_days: int,
        out_dir: Path,
        batch_size: int,
        batch_pause: int,
        day_pause: int,
        force: bool,
    ):
        cfg = load_config()
        if not cfg.sources.arxiv.enabled:
            print("arxiv source disabled")
            return
        src = ArxivSource(categories=cfg.sources.arxiv.categories)
        today = today_utc()

        # Build the list of days to actually fetch (skipping those whose
        # digest already exists). Done up front so batching boundaries land
        # on real work, not on a run of skips.
        targets: list[datetime] = []
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            if not force and backfill_days > 0 and (
                Path("digests") / f"{target.strftime('%Y-%m-%d')}.md"
            ).exists():
                print(f"arxiv: skip {target.date()} (digest exists)")
                continue
            targets.append(target)

        total = len(targets)
        fetched = 0
        for idx, target in enumerate(targets):
            try:
                papers = asyncio.run(src.fetch(target))
            except Exception as e:
                # A 429 storm or transient error on one day must not kill
                # the whole backfill — log and continue to the next day so
                # we don't lose progress on 30 days because of day 5.
                print(f"arxiv: skip {target.date()} due to {type(e).__name__}: {e}")
            else:
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
                else:
                    out_path.write_text(
                        json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
                    )
                    print(f"arxiv: wrote {len(papers)} papers for {target.date()}")

            fetched += 1
            is_last = idx == total - 1
            if is_last:
                continue
            # Inter-batch pause lands AFTER every batch_size-th fetched day
            # (counted across both successes and failures, so a single bad
            # day doesn't shift the batch boundary). Skipped days don't
            # count — they made no requests.
            if fetched % batch_size == 0:
                print(
                    f"arxiv: batch boundary ({fetched}/{total} done), "
                    f"sleeping {batch_pause}s before next batch"
                )
                time.sleep(batch_pause)
            else:
                time.sleep(day_pause)

    main()
