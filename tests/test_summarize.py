import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.summarize import summarize_papers
from sources.base import Paper, SourceRecord


def _mk(id_, score, hard_gate: bool = False):
    p = Paper(
        id=id_,
        title="t",
        authors=[],
        abstract="a",
        url="https://x",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(UTC))],
    )
    p.relevance_score = score
    p.relevance_breakdown = {"hard_gate": hard_gate, "topic_bucket": "ptq"}
    return p


@pytest.mark.asyncio
async def test_summarize_skips_hard_gated(tmp_path: Path, monkeypatch):
    """No more numeric threshold — every non-hard-gated paper is summarized."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    scored = [
        _mk("hi", 9),
        _mk("gated", 0, hard_gate=True),
        _mk("mid", 6),  # below the OLD threshold, but should still summarize
    ]
    in_path = tmp_path / "scored.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in scored]))
    out_path = tmp_path / "summarized.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.return_value = {
        "summary": "中文摘要",
        "highlights": ["🎯 a", "📊 b"],
    }
    n = await summarize_papers(in_path, out_path, prompt_path, fake, concurrency=2)
    assert n == 3
    out = json.loads(out_path.read_text())
    by_id = {p["id"]: p for p in out}
    assert by_id["hi"]["summary"] == "中文摘要"
    assert by_id["mid"]["summary"] == "中文摘要"
    assert by_id["gated"]["summary"] is None
    assert fake.call_json.call_count == 2
