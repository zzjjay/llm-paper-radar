"""Fetch arXiv papers by author for a curated watchlist.

Each watched author is queried independently via the arXiv API
(`au:"Name" AND (cat:cs.LG OR ...)`), the results are deduped by arXiv ID,
and the merged SourceRecord carries `matched_authors` + per-author
`affiliation` in `extras`. These get surfaced as a top-level "Watched authors"
section in the digest, bypassing the score threshold.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

import feedparser
import httpx

from sources._arxiv_lookup import arxiv_get_with_retry
from sources.base import ARXIV_USER_AGENT, Paper, Source, SourceRecord

ARXIV_ID_RE = re.compile(r"abs/([\d.]+)(?:v\d+)?$")


class WatchedAuthorSpec:
    def __init__(self, name: str, affiliation: str = ""):
        self.name = name
        self.affiliation = affiliation


class ArxivAuthorsSource(Source):
    name = "arxiv_authors"
    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(
        self,
        authors: list[WatchedAuthorSpec],
        categories: list[str],
        window_days: int = 7,
        page_size: int = 50,
        max_pages: int = 3,
    ):
        self.authors = authors
        self.categories = categories
        self.window_days = window_days
        self.page_size = page_size
        self.max_pages = max_pages

    async def fetch(self, target_date: datetime) -> list[Paper]:
        window_start = target_date - timedelta(days=self.window_days)
        # arxiv_id -> (Paper, [matched_specs])
        merged: dict[str, tuple[Paper, list[WatchedAuthorSpec]]] = {}
        async with httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": ARXIV_USER_AGENT},
        ) as client:
            entries = await self._fetch_all_authors(client, window_start)
            for p in self._to_papers(entries, window_start, target_date):
                # arXiv's au: is a prefix/substring search and over-matches
                # common names ("Song Han"), so re-check each watched author
                # against the paper's real author list and attribute the match.
                for spec in self.authors:
                    if _author_matches(p.authors, spec.name):
                        if p.id in merged:
                            merged[p.id][1].append(spec)
                        else:
                            merged[p.id] = (p, [spec])

        out: list[Paper] = []
        for paper, specs in merged.values():
            paper.sources = [
                SourceRecord(
                    name="arxiv_authors",
                    fetched_at=datetime.now(UTC),
                    extras={
                        "matched_authors": [s.name for s in specs],
                        "affiliations": sorted({s.affiliation for s in specs if s.affiliation}),
                    },
                )
            ]
            out.append(paper)
        return out

    async def _fetch_all_authors(
        self,
        client: httpx.AsyncClient,
        window_start: datetime,
    ) -> list[dict]:
        """One paginated query for ALL watched authors instead of one request
        per author. Previously each author was a separate request hitting
        export.arxiv.org from the same IP within seconds, which reliably tripped
        arxiv's per-IP 429 throttle (observed losing the whole watchlist). A
        single `(au:"A" OR au:"B" ...) AND (cat...)` query cuts that to a
        handful of paginated requests sharing the same backoff helper as the
        rest of the pipeline.
        """
        # Build query by hand: arXiv treats `+` inside an encoded query as a
        # literal AND/OR token; httpx's params= would percent-encode it.
        author_part = "+OR+".join(
            f'au:"{spec.name.replace(" ", "+")}"' for spec in self.authors
        )
        cat_part = "+OR+".join(f"cat:{c}" for c in self.categories)
        query = f"({author_part})+AND+({cat_part})"
        all_entries: list[dict] = []
        start = 0
        for _ in range(self.max_pages):
            url = (
                f"{self.BASE_URL}?search_query={query}&start={start}"
                f"&max_results={self.page_size}"
                f"&sortBy=submittedDate&sortOrder=descending"
            )
            resp = await arxiv_get_with_retry(
                client, url, context=f"arxiv_authors(start={start})"
            )
            feed = feedparser.parse(resp.text)
            if not feed.entries:
                break
            all_entries.extend(feed.entries)
            oldest = _entry_published(feed.entries[-1])
            if oldest < window_start:
                break
            start += self.page_size
        return all_entries

    def _to_papers(
        self, entries: list[dict], window_start: datetime, target_date: datetime
    ) -> Iterable[Paper]:
        now = datetime.now(UTC)
        window_end = target_date + timedelta(hours=1)
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
                sources=[SourceRecord(name="arxiv_authors", fetched_at=now)],
            )


def _author_matches(paper_authors: list[str], target: str) -> bool:
    """Case-insensitive exact full-name or 'Last, First'-style match."""
    target_norm = _normalize(target)
    for a in paper_authors:
        if _normalize(a) == target_norm:
            return True
    return False


def _normalize(name: str) -> str:
    # "van Baalen, Mart" → "mart van baalen"
    s = name.strip().lower()
    if "," in s:
        last, first = (p.strip() for p in s.split(",", 1))
        s = f"{first} {last}"
    return " ".join(s.split())


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

    from pipeline._clock import today_utc
    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int, help="Loop the wrapper N+1 times (default 0 = once). The actual fetch is a single windowed query — see --window-days.")
    @click.option(
        "--window-days",
        default=None,
        type=int,
        help="Single query spanning the last N days (overrides config window_days). 1 API call, output written under today's data/raw/ dir.",
    )
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, window_days: int | None, out_dir: Path):
        cfg = load_config()
        sub = cfg.sources.arxiv_authors
        if not sub.enabled:
            print("arxiv_authors source disabled")
            return
        if not sub.authors:
            print("arxiv_authors: no authors configured, skipping")
            return
        specs = [WatchedAuthorSpec(name=a.name, affiliation=a.affiliation) for a in sub.authors]
        effective_window = window_days if window_days is not None else sub.window_days
        src = ArxivAuthorsSource(
            authors=specs,
            categories=sub.categories,
            window_days=effective_window,
        )
        today = today_utc()
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            if backfill_days > 0 and (
                Path("digests") / f"{target.strftime('%Y-%m-%d')}.md"
            ).exists():
                print(f"arxiv_authors: skip {target.date()} (digest exists)")
                continue
            papers = asyncio.run(src.fetch(target))
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "arxiv_authors.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"arxiv_authors: wrote {len(papers)} papers for {target.date()}")

    main()
