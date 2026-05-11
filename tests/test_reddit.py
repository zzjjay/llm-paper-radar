from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.reddit import RedditSource

FIXTURE = Path(__file__).parent / "fixtures" / "reddit_top_response.json"


@respx.mock
@pytest.mark.asyncio
async def test_reddit_extracts_arxiv_links_only(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "fake")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "fake")

    respx.post("https://www.reddit.com/api/v1/access_token").mock(
        return_value=Response(200, json={"access_token": "x", "expires_in": 3600})
    )
    respx.get("https://oauth.reddit.com/r/LocalLLaMA/top.json").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2402.17764v1</id>
    <title>BitNet b1.58</title>
    <summary>1-bit LLM.</summary>
    <published>2026-05-10T00:00:00Z</published>
    <author><name>Shuming Ma</name></author>
    <category term="cs.CL"/>
    <link href="http://arxiv.org/pdf/2402.17764v1.pdf" type="application/pdf"/>
  </entry>
</feed>"""
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=arxiv_xml)
    )

    src = RedditSource(subreddit="LocalLLaMA", top_window="day")
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert len(papers) == 1
    p = papers[0]
    assert p.id == "2402.17764"
    assert p.sources[0].name == "reddit"
    assert p.sources[0].extras["score"] == 230
    assert p.sources[0].extras["num_comments"] == 67
    assert "thread_url" in p.sources[0].extras


@respx.mock
@pytest.mark.asyncio
async def test_reddit_returns_empty_when_no_arxiv_links(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "fake")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "fake")

    respx.post("https://www.reddit.com/api/v1/access_token").mock(
        return_value=Response(200, json={"access_token": "x", "expires_in": 3600})
    )
    no_links_fixture = {
        "data": {
            "children": [
                {"data": {"id": "p1", "title": "How to fine-tune", "selftext": "no link here",
                          "score": 10, "num_comments": 1, "permalink": "/r/x/p1/"}},
                {"data": {"id": "p2", "title": "Random thread", "selftext": "",
                          "score": 5, "num_comments": 0, "permalink": "/r/x/p2/"}},
            ]
        }
    }
    respx.get("https://oauth.reddit.com/r/LocalLLaMA/top.json").mock(
        return_value=Response(200, json=no_links_fixture)
    )
    arxiv_route = respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(500, text="should not be called")
    )

    src = RedditSource(subreddit="LocalLLaMA", top_window="day")
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert papers == []
    assert arxiv_route.called is False  # early-return prevents wasteful arxiv call
