"""OAI-PMH fallback for the primary arXiv day-fetch.

`sources.arxiv` fetches via the `api/query` endpoint, which is the one that
keeps 429-ing our shared corporate egress IP. arXiv's officially-sanctioned
bulk harvesting channel is OAI-PMH (`export.arxiv.org/oai2`), which is far more
tolerant of volume. This module is the fallback path used by `ArxivSource.fetch`
when the primary endpoint fails or returns a suspected-throttle empty feed.

Two semantic differences from `api/query`, both handled here so the fallback's
output is drop-in compatible:

1. OAI's `from`/`until` filter on `datestamp` (announcement/update date), NOT
   `submittedDate`. arXiv announces ~1-2 days after submission (observed: for
   datestamp 2026-06-04, the cs.* papers' `<created>` clustered on 06-03/06-02).
   So we harvest a widened datestamp window [target, target+OAI_LAG_DAYS] and
   then locally keep only records whose `<created>` == target. This reproduces
   the submittedDate semantics the rest of the pipeline assumes.

2. OAI's `cs` set is all of computer science (~1300/day), not just the three
   configured categories. We filter to the configured cats locally.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

import httpx

from sources._arxiv_lookup import arxiv_get_with_retry
from sources.base import ARXIV_USER_AGENT, Paper, SourceRecord

OAI_URL = "https://export.arxiv.org/oai2"
# How many extra days past `target` to scan on the datestamp axis to catch the
# announce-after-submit lag. 3 covers weekday lag (1-2d) plus weekend pileup.
OAI_LAG_DAYS = 3
# arXiv's OAI set covering all of CS. We filter to configured cats afterwards.
OAI_SET = "cs"

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "ar": "http://arxiv.org/OAI/arXiv/",
}


async def fetch_via_oai(
    target_date: datetime,
    categories: list[str],
    *,
    max_pages: int = 20,
) -> list[Paper]:
    """Harvest `target_date`'s submissions via OAI-PMH.

    Returns Papers whose `<created>` (submission date) == target_date and whose
    categories intersect `categories`. Raises on hard transport failure (so the
    caller can decide what to do); an empty harvest returns [].
    """
    cat_set = set(categories)
    target_d = target_date.date()
    ws = target_d.strftime("%Y-%m-%d")
    we = (target_d + timedelta(days=OAI_LAG_DAYS)).strftime("%Y-%m-%d")

    params: dict = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "set": OAI_SET,
        "from": ws,
        "until": we,
    }
    out: list[Paper] = []
    async with httpx.AsyncClient(
        timeout=90.0,
        follow_redirects=True,
        headers={"User-Agent": ARXIV_USER_AGENT},
    ) as client:
        for page in range(max_pages):
            # resumptionToken is exclusive of all other args once present.
            url = OAI_URL
            resp = await arxiv_get_with_retry(
                client, url, params=params, context=f"arxiv_oai(page {page})"
            )
            root = ET.fromstring(resp.text)
            out.extend(_parse_records(root, target_d, cat_set))
            tok_el = root.find(".//oai:resumptionToken", NS)
            if tok_el is None or not (tok_el.text or "").strip():
                break
            params = {"verb": "ListRecords", "resumptionToken": tok_el.text.strip()}
            await asyncio.sleep(2.0)  # polite pacing between OAI pages
    return out


def _parse_records(
    root: ET.Element, target_d: date, cat_set: set[str]
) -> Iterable[Paper]:
    now = datetime.now(UTC)
    for rec in root.findall(".//oai:record", NS):
        # Skip deleted records (header status="deleted", no metadata).
        hdr = rec.find("oai:header", NS)
        if hdr is not None and hdr.get("status") == "deleted":
            continue
        meta = rec.find(".//ar:arXiv", NS)
        if meta is None:
            continue
        created = (meta.findtext("ar:created", default="", namespaces=NS) or "").strip()
        # submittedDate semantics: only keep papers actually submitted on target.
        if created != target_d.strftime("%Y-%m-%d"):
            continue
        categories = (
            meta.findtext("ar:categories", default="", namespaces=NS) or ""
        ).split()
        if not (cat_set & set(categories)):
            continue
        arxiv_id = (meta.findtext("ar:id", default="", namespaces=NS) or "").strip()
        if not arxiv_id:
            continue
        primary = categories[0] if categories else "unknown"
        try:
            pub = datetime.fromisoformat(created).replace(tzinfo=UTC)
        except ValueError:
            pub = now
        yield Paper(
            id=arxiv_id,
            title=" ".join(
                (meta.findtext("ar:title", default="", namespaces=NS) or "").split()
            ),
            authors=_authors(meta),
            abstract=(
                meta.findtext("ar:abstract", default="", namespaces=NS) or ""
            ).strip(),
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            published_at=pub,
            primary_category=primary,
            categories=categories,
            sources=[SourceRecord(name="arxiv", fetched_at=now)],
        )


def _authors(meta: ET.Element) -> list[str]:
    names: list[str] = []
    authors_el = meta.find("ar:authors", NS)
    if authors_el is None:
        return names
    for a in authors_el.findall("ar:author", NS):
        key = (a.findtext("ar:keyname", default="", namespaces=NS) or "").strip()
        fore = (a.findtext("ar:forenames", default="", namespaces=NS) or "").strip()
        full = f"{fore} {key}".strip()
        if full:
            names.append(full)
    return names
