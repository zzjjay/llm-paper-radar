from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.twitter_rsshub import TwitterRSSHubSource

FIXTURE = Path(__file__).parent / "fixtures" / "rsshub_twitter.xml"


@respx.mock
@pytest.mark.asyncio
async def test_twitter_rsshub_collects_arxiv_links(monkeypatch):
    monkeypatch.setenv("RSSHUB_BASE_URL", "https://rsshub.test")
    respx.get("https://rsshub.test/twitter/user/_akhaliq").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2402.17764v1</id>
    <title>BitNet</title>
    <summary>1-bit.</summary>
    <published>2026-05-10T00:00:00Z</published>
    <author><name>X</name></author>
    <category term="cs.CL"/>
    <link href="http://arxiv.org/pdf/2402.17764v1.pdf" type="application/pdf"/>
  </entry>
</feed>"""
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=arxiv_xml)
    )

    src = TwitterRSSHubSource(accounts=["_akhaliq"])
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert len(papers) == 1
    assert papers[0].id == "2402.17764"
    assert papers[0].sources[0].name == "twitter_rsshub"
    assert "_akhaliq" in papers[0].sources[0].extras.get("accounts", [])


@pytest.mark.asyncio
async def test_twitter_skips_when_base_url_missing(monkeypatch):
    monkeypatch.delenv("RSSHUB_BASE_URL", raising=False)
    src = TwitterRSSHubSource(accounts=["_akhaliq"])
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert papers == []


@pytest.mark.asyncio
async def test_twitter_skips_unreachable_account(monkeypatch):
    monkeypatch.setenv("RSSHUB_BASE_URL", "https://rsshub.test")
    with respx.mock:
        respx.get("https://rsshub.test/twitter/user/down").mock(
            return_value=Response(503, text="dead")
        )
        src = TwitterRSSHubSource(accounts=["down"])
        papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
        assert papers == []
