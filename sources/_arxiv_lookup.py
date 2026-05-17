"""Shared: extract arXiv IDs from text, refetch full metadata via arXiv API."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime

import feedparser
import httpx

from sources.base import Paper, SourceName, SourceRecord

ARXIV_LINK_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")


def extract_arxiv_ids(text: str) -> set[str]:
    return set(ARXIV_LINK_RE.findall(text or ""))


async def fetch_arxiv_by_ids(
    ids: list[str],
    source_name: SourceName,
    extras_per_id: dict[str, dict] | None = None,
) -> list[Paper]:
    """Look up arXiv metadata for given IDs, return Paper objects with given source_name."""
    if not ids:
        return []
    extras_per_id = extras_per_id or {}
    id_query = ",".join(ids)
    url = "http://export.arxiv.org/api/query"
    params = {"id_list": id_query, "max_results": len(ids)}
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        last_exc: Exception | None = None
        resp = None
        for attempt in range(4):
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                last_exc = None
                break
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                wait = 5 * (2**attempt)
                print(
                    f"_arxiv_lookup: {type(e).__name__} fetching {len(ids)} ids,"
                    f" sleeping {wait}s before retry"
                )
                await asyncio.sleep(wait)
        if last_exc is not None:
            raise last_exc
        assert resp is not None
    feed = feedparser.parse(resp.text)
    now = datetime.now(UTC)
    papers: list[Paper] = []
    for e in feed.entries:
        m = re.search(r"abs/([\d.]+)(?:v\d+)?$", e.get("id", ""))
        if not m:
            continue
        arxiv_id = m.group(1)
        categories = [t["term"] for t in e.get("tags", [])]
        primary = categories[0] if categories else "unknown"
        pdf_link = next(
            (lnk["href"] for lnk in e.get("links", []) if lnk.get("type") == "application/pdf"),
            None,
        )
        try:
            pub = datetime.fromisoformat(e["published"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            pub = now
        papers.append(
            Paper(
                id=arxiv_id,
                title=e.get("title", "").strip().replace("\n ", " "),
                authors=[a.get("name", "") for a in e.get("authors", [])],
                abstract=e.get("summary", "").strip(),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_link or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=pub,
                primary_category=primary,
                categories=categories,
                sources=[
                    SourceRecord(
                        name=source_name,
                        fetched_at=now,
                        extras=extras_per_id.get(arxiv_id, {}),
                    )
                ],
            )
        )
    if len(papers) < len(ids):
        missing = set(ids) - {p.id for p in papers}
        print(
            f"_arxiv_lookup: arxiv returned {len(papers)}/{len(ids)} requested papers;"
            f" missing: {sorted(missing)}"
        )
    return papers
