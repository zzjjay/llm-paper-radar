"""Shared: extract arXiv IDs from text, refetch full metadata via arXiv API."""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Callable
from datetime import UTC, datetime

import feedparser
import httpx

from sources.base import ARXIV_USER_AGENT, Paper, SourceName, SourceRecord

ARXIV_LINK_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")

# Retry budget for arxiv.org/api/query. arXiv's per-IP throttle often
# stays hot for several minutes once tripped, so 4 attempts at max 40s
# (the old budget) was too short — a 429 storm would just exhaust retries
# and the day's fetch would be dropped. New schedule: 5,10,20,40,80,160,300
# (with ±20% jitter) = up to ~10 min total. Honor Retry-After header when
# present.
RETRY_WAITS = [5, 10, 20, 40, 80, 160, 300]


def _backoff_wait(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    base = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
    return base * random.uniform(0.8, 1.2)


class ArxivEmptyResponse(Exception):
    """200 OK but the response body failed a caller-supplied sanity check
    (e.g. arxiv returned an empty Atom feed for a query that must not be
    empty). arxiv is known to serve empty feeds in lieu of 429 when its
    per-IP throttle is hot, so this is treated as a transient failure
    worth retrying inside the same backoff budget."""


async def arxiv_get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    context: str = "arxiv",
    validate: Callable[[httpx.Response], None] | None = None,
) -> httpx.Response:
    """GET against arxiv API with exponential backoff + jitter, honoring
    Retry-After. Retries 429, 503, and connection-level errors. Raises
    the last exception (or HTTPStatusError) if all attempts fail.

    `validate`, if provided, is called on each 2xx response. It may raise
    `ArxivEmptyResponse` to mark the response as a "soft failure" worth
    retrying — arxiv occasionally serves empty Atom feeds instead of 429
    while throttled, and treating those as legitimate "no results" loses
    a whole day's batch. Any other exception from `validate` propagates."""
    last_exc: Exception | None = None
    for attempt in range(len(RETRY_WAITS)):
        try:
            resp = await client.get(url, params=params)
            if resp.status_code not in (429, 503):
                resp.raise_for_status()
                if validate is not None:
                    validate(resp)
                return resp
            wait = _backoff_wait(attempt, resp.headers.get("Retry-After"))
            print(
                f"{context}: {resp.status_code} on attempt {attempt + 1}, "
                f"sleeping {wait:.0f}s before retry"
            )
            last_exc = httpx.HTTPStatusError(
                f"{resp.status_code}", request=resp.request, response=resp
            )
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            wait = _backoff_wait(attempt, None)
            print(
                f"{context}: {type(e).__name__} on attempt {attempt + 1}, "
                f"sleeping {wait:.0f}s before retry"
            )
        except ArxivEmptyResponse as e:
            last_exc = e
            wait = _backoff_wait(attempt, None)
            print(
                f"{context}: empty-feed (suspected throttle) on attempt "
                f"{attempt + 1}, sleeping {wait:.0f}s before retry"
            )
        await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


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
    async with httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=True,
        headers={"User-Agent": ARXIV_USER_AGENT},
    ) as client:
        resp = await arxiv_get_with_retry(
            client, url, params=params, context=f"_arxiv_lookup({len(ids)} ids)"
        )
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
