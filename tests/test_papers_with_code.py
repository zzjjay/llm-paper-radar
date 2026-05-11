from datetime import UTC, datetime
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.papers_with_code import PapersWithCodeSource

FIXTURE = Path(__file__).parent / "fixtures" / "pwc_rss.xml"


@respx.mock
@pytest.mark.asyncio
async def test_pwc_extracts_arxiv_and_code():
    respx.get("https://paperswithcode.com/latest/rss.xml").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2402.17764v1</id>
    <title>BitNet b1.58</title>
    <summary>1-bit LLM.</summary>
    <published>2026-05-10T00:00:00Z</published>
    <author><name>X</name></author>
    <category term="cs.CL"/>
    <link href="http://arxiv.org/pdf/2402.17764v1.pdf" type="application/pdf"/>
  </entry>
</feed>"""
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=arxiv_xml)
    )

    src = PapersWithCodeSource()
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=UTC))
    assert len(papers) == 1
    p = papers[0]
    assert p.id == "2402.17764"
    assert p.code_url == "https://github.com/example/bitnet"
    assert p.sources[0].name == "papers_with_code"
