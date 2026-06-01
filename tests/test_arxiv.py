from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from httpx import Response

import sources._arxiv_lookup as _arxiv_lookup
from sources.arxiv import ArxivSource

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_response.xml"
EMPTY_FEED = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns="http://www.w3.org/2005/Atom">'
    "<title>empty</title>"
    "</feed>"
)


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_fetch_parses_entries():
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    src = ArxivSource(categories=["cs.CL"])
    # Fixture has papers published 2026-05-10. Target the day they were
    # published so the historical window [target, target+24h) includes them
    # — the source switches to historical mode for any target older than
    # ~5 days, which any real "now" will trigger.
    papers = await src.fetch(datetime(2026, 5, 10, tzinfo=UTC))
    assert len(papers) >= 1
    p = papers[0]
    assert p.id  # arXiv ID like "2405.xxxxx"
    assert p.title
    assert p.abstract
    assert p.url.startswith("https://arxiv.org/abs/")
    assert p.pdf_url and p.pdf_url.startswith("https://arxiv.org/pdf/")
    assert p.sources[0].name == "arxiv"
    assert p.primary_category


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_or_query_uses_literal_plus():
    """Regression: httpx's params= encoder turns '+' into '%2B', breaking arXiv's
    OR operator. The URL we send must contain literal '+OR+' between cat: terms."""
    route = respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    src = ArxivSource(categories=["cs.CL", "cs.LG", "cs.AR"])
    await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    sent_url = str(route.calls[0].request.url)
    assert "cat:cs.CL+OR+cat:cs.LG+OR+cat:cs.AR" in sent_url, sent_url
    assert "%2B" not in sent_url, f"'+' got URL-encoded: {sent_url}"


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_retries_on_page0_empty_feed(monkeypatch):
    """Regression: arxiv sometimes returns HTTP 200 with an empty Atom feed
    while throttled instead of 429. Page 0 of a cs.LG-style query must not
    be empty for a real day, so an empty page 0 should consume the same
    backoff budget as a 429 rather than silently writing 0 papers."""
    # Don't actually sleep through the backoff in tests.
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(_arxiv_lookup.asyncio, "sleep", _no_sleep)

    route = respx.get("http://export.arxiv.org/api/query").mock(
        side_effect=[
            Response(200, text=EMPTY_FEED),           # page 0 attempt 1: throttled
            Response(200, text=EMPTY_FEED),           # page 0 attempt 2: throttled
            Response(200, text=FIXTURE.read_text()),  # page 0 attempt 3: real data
            Response(200, text=EMPTY_FEED),           # page 1: pagination end
        ]
    )
    src = ArxivSource(categories=["cs.CL"])
    papers = await src.fetch(datetime(2026, 5, 10, tzinfo=UTC))
    assert route.call_count == 4, "should retry empties on page 0, then paginate to end"
    assert len(papers) >= 1, "real feed on attempt 3 should produce papers"


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_raises_after_persistent_empty_feeds(monkeypatch):
    """When every retry attempt returns an empty feed, the budget is
    exhausted and fetch raises — the daily.sh main() catches that and
    skips the day rather than writing 0 papers as if the day were empty."""
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(_arxiv_lookup.asyncio, "sleep", _no_sleep)

    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=EMPTY_FEED)
    )
    src = ArxivSource(categories=["cs.CL"])
    with pytest.raises(_arxiv_lookup.ArxivEmptyResponse):
        await src.fetch(datetime(2026, 5, 10, tzinfo=UTC))


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_later_page_empty_is_legitimate_pagination_end(monkeypatch):
    """The empty-feed validator only runs on page 0. A later page returning
    empty is the normal end-of-pagination signal and must not trigger
    retries — otherwise every fetch would waste the retry budget at the
    natural pagination boundary."""
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(_arxiv_lookup.asyncio, "sleep", _no_sleep)

    route = respx.get("http://export.arxiv.org/api/query").mock(
        side_effect=[
            Response(200, text=FIXTURE.read_text()),  # page 0: real data
            Response(200, text=EMPTY_FEED),           # page 1: pagination end
        ]
    )
    src = ArxivSource(categories=["cs.CL"], page_size=1)
    papers = await src.fetch(datetime(2026, 5, 10, tzinfo=UTC))
    assert route.call_count == 2, "should stop at first empty later page, not retry it"
    assert len(papers) >= 1
