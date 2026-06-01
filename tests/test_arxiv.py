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

# Real arxiv answer for a query with no matches: empty entries list
# PLUS an explicit <opensearch:totalResults>0</opensearch:totalResults>.
GENUINE_ZERO_FEED = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"'
    ' xmlns="http://www.w3.org/2005/Atom">'
    "<opensearch:totalResults>0</opensearch:totalResults>"
    "<opensearch:startIndex>0</opensearch:startIndex>"
    "</feed>"
)


def _nonempty_feed_on(date_str: str) -> str:
    """Minimal valid Atom feed with one entry inside the [date, date+24h) window."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry>"
        "<id>http://arxiv.org/abs/2605.00001v1</id>"
        "<title>Stub</title>"
        "<summary>stub</summary>"
        f"<published>{date_str}T12:00:00Z</published>"
        '<link rel="alternate" type="text/html" href="https://arxiv.org/abs/2605.00001"/>'
        '<link type="application/pdf" href="https://arxiv.org/pdf/2605.00001"/>'
        '<category term="cs.CL"/>'
        "</entry>"
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
async def test_arxiv_retries_on_suspect_empty_feed(monkeypatch):
    """Regression: arxiv sometimes returns HTTP 200 with a truncated empty
    Atom feed while throttled (no <opensearch:totalResults> at all). That
    is NOT a real zero day and must consume the backoff budget the same
    way a 429 would, instead of silently writing 0 papers."""
    # Don't actually sleep through the backoff in tests.
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(_arxiv_lookup.asyncio, "sleep", _no_sleep)

    route = respx.get("http://export.arxiv.org/api/query").mock(
        side_effect=[
            Response(200, text=EMPTY_FEED),                       # attempt 1: throttle (no totalResults)
            Response(200, text=EMPTY_FEED),                       # attempt 2: throttle
            Response(200, text=_nonempty_feed_on("2026-05-11")),  # attempt 3: real data
            Response(200, text=GENUINE_ZERO_FEED),                # page 1: pagination end
        ]
    )
    src = ArxivSource(categories=["cs.CL"])
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert route.call_count == 4, "should retry suspect empties on page 0, then paginate to end"
    assert len(papers) == 1, "real feed on attempt 3 should produce one paper"


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_raises_after_persistent_suspect_empty_feeds(monkeypatch):
    """When every retry attempt returns a suspect empty feed (no
    totalResults marker), the budget is exhausted and fetch raises so
    daily.sh's main() skips the day instead of overwriting with 0 papers."""
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(_arxiv_lookup.asyncio, "sleep", _no_sleep)

    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=EMPTY_FEED)
    )
    src = ArxivSource(categories=["cs.CL"])
    with pytest.raises(_arxiv_lookup.ArxivEmptyResponse):
        await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_genuine_zero_day_accepted_immediately(capsys, monkeypatch):
    """When arxiv's response carries <opensearch:totalResults>0</...> the
    query genuinely matched nothing (e.g. a quiet weekend day). Accept it
    in a single request, write 0 papers, and log the reason so the day's
    "why no papers" is visible without grepping for throttle errors."""
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(_arxiv_lookup.asyncio, "sleep", _no_sleep)

    route = respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=GENUINE_ZERO_FEED)
    )
    src = ArxivSource(categories=["cs.CL"])
    papers = await src.fetch(datetime(2026, 5, 10, tzinfo=UTC))  # Sunday
    assert route.call_count == 1, "totalResults=0 must not trigger retries"
    assert papers == []
    out = capsys.readouterr().out
    assert "opensearch:totalResults=0" in out, (
        "should explain in stdout why we got 0 papers"
    )


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_later_page_empty_is_legitimate_pagination_end(monkeypatch):
    """The empty-feed validator only runs on page 0. A later page returning
    empty (even without totalResults, since respx mock returns it as-is)
    is the normal end-of-pagination signal and must not trigger retries."""
    async def _no_sleep(_):
        return None
    monkeypatch.setattr(_arxiv_lookup.asyncio, "sleep", _no_sleep)

    route = respx.get("http://export.arxiv.org/api/query").mock(
        side_effect=[
            Response(200, text=_nonempty_feed_on("2026-05-11")),  # page 0: real data
            Response(200, text=EMPTY_FEED),                       # page 1: pagination end
        ]
    )
    src = ArxivSource(categories=["cs.CL"], page_size=1)
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert route.call_count == 2, "should stop at first empty later page, not retry it"
    assert len(papers) == 1
