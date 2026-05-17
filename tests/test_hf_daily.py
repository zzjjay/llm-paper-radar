from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.hf_daily import HFDailySource, parse_trending_ranks

FIXTURE_DAILY = Path(__file__).parent / "fixtures" / "hf_daily_response.json"
FIXTURE_TREND = Path(__file__).parent / "fixtures" / "hf_trending.html"


def _arxiv_feed(ids: list[str]) -> str:
    entries = "\n".join(
        f"""<entry>
  <id>http://arxiv.org/abs/{aid}v1</id>
  <title>Title for {aid}</title>
  <summary>Abstract for {aid}.</summary>
  <published>2026-05-11T00:00:00Z</published>
  <author><name>Author {aid}</name></author>
  <link href="http://arxiv.org/pdf/{aid}.pdf" type="application/pdf"/>
  <category term="cs.LG"/>
</entry>"""
        for aid in ids
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
{entries}
</feed>"""


@respx.mock
@pytest.mark.asyncio
async def test_hf_daily_fetch_includes_daily_and_trending():
    respx.get("https://huggingface.co/api/daily_papers").mock(
        return_value=Response(200, text=FIXTURE_DAILY.read_text())
    )
    respx.get("https://huggingface.co/papers/trending").mock(
        return_value=Response(200, text=FIXTURE_TREND.read_text())
    )
    # Trending-only IDs (not in the daily fixture) are now enriched via arXiv.
    trending_only = ["2605.04567", "2605.03891", "2605.02345", "2605.01789", "2605.00912"]
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=_arxiv_feed(trending_only))
    )
    src = HFDailySource()
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert len(papers) >= 1

    has_daily = any("upvotes" in s.extras for p in papers for s in p.sources)
    assert has_daily

    has_trending = any("trending_rank" in s.extras for p in papers for s in p.sources)
    assert has_trending

    # Trending-only papers should now carry real metadata, not blank stubs.
    by_id = {p.id: p for p in papers}
    for aid in trending_only:
        assert by_id[aid].title, f"{aid} should have a title from arxiv lookup"
        assert by_id[aid].abstract, f"{aid} should have an abstract from arxiv lookup"


def test_parse_trending_ranks_extracts_arxiv_ids_with_position():
    html = """<a href="/papers/2402.17764">A</a>
<div>noise</div>
<a href="/papers/2503.12345">B</a>
<a href="/papers/2402.17764">A again</a>"""
    ranks = parse_trending_ranks(html)
    assert ranks["2402.17764"] == 1
    assert ranks["2503.12345"] == 2


@respx.mock
@pytest.mark.asyncio
async def test_hf_daily_continues_when_trending_fails():
    respx.get("https://huggingface.co/api/daily_papers").mock(
        return_value=Response(200, text=FIXTURE_DAILY.read_text())
    )
    respx.get("https://huggingface.co/papers/trending").mock(
        return_value=Response(500, text="oops")
    )
    src = HFDailySource()
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert len(papers) >= 1
