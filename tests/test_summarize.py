import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.summarize import summarize_papers
from sources.base import Paper, SourceRecord


def _mk(id_, score):
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
    return p


@pytest.mark.asyncio
async def test_summarize_only_runs_for_above_threshold(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    scored = [_mk("hi", 9), _mk("low", 5), _mk("hi2", 8)]
    in_path = tmp_path / "scored.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in scored]))
    out_path = tmp_path / "summarized.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.return_value = {
        "summary": "English summary",
        "highlights": ["🎯 a", "📊 b"],
    }
    n = await summarize_papers(in_path, out_path, prompt_path, fake, threshold=7, concurrency=2)
    assert n == 3
    out = json.loads(out_path.read_text())
    by_id = {p["id"]: p for p in out}
    assert by_id["hi"]["summary"] == "English summary"
    assert by_id["hi2"]["summary"] == "English summary"
    assert by_id["low"]["summary"] is None
    assert fake.call_json.call_count == 2
