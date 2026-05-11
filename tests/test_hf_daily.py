from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.hf_daily import HFDailySource, parse_trending_ranks

FIXTURE_DAILY = Path(__file__).parent / "fixtures" / "hf_daily_response.json"
FIXTURE_TREND = Path(__file__).parent / "fixtures" / "hf_trending.html"


@respx.mock
@pytest.mark.asyncio
async def test_hf_daily_fetch_includes_daily_and_trending():
    respx.get("https://huggingface.co/api/daily_papers").mock(
        return_value=Response(200, text=FIXTURE_DAILY.read_text())
    )
    respx.get("https://huggingface.co/papers/trending").mock(
        return_value=Response(200, text=FIXTURE_TREND.read_text())
    )
    src = HFDailySource()
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert len(papers) >= 1

    has_daily = any("upvotes" in s.extras for p in papers for s in p.sources)
    assert has_daily

    has_trending = any("trending_rank" in s.extras for p in papers for s in p.sources)
    assert has_trending


def test_parse_trending_ranks_extracts_arxiv_ids_with_position():
    html = '''<a href="/papers/2402.17764">A</a>
<div>noise</div>
<a href="/papers/2503.12345">B</a>
<a href="/papers/2402.17764">A again</a>'''
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
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert len(papers) >= 1
