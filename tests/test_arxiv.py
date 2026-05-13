from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.arxiv import ArxivSource

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_response.xml"


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_fetch_parses_entries():
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    src = ArxivSource(categories=["cs.CL"])
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
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
