import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.filter import filter_papers
from sources.base import Paper, SourceRecord


def _mk(id_, title, abstract):
    return Paper(
        id=id_,
        title=title,
        authors=[],
        abstract=abstract,
        url="https://x",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(UTC))],
    )


@pytest.mark.asyncio
async def test_filter_assigns_score_and_reason(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    deduped = [_mk("1", "FP4 quant for LLM", "Quantization."), _mk("2", "Random RAG paper", "RAG.")]
    deduped_path = tmp_path / "in.json"
    deduped_path.write_text(json.dumps([p.model_dump(mode="json") for p in deduped]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("test prompt")

    fake = AsyncMock()
    fake.call_json.side_effect = [
        {"relevance_score": 9, "reason": "FP4 quantization method"},
        {"relevance_score": 1, "reason": "RAG, not relevant"},
    ]

    n = await filter_papers(
        deduped_path=deduped_path,
        out_path=out_path,
        prompt_path=prompt_path,
        client=fake,
        concurrency=2,
    )
    assert n == 2
    out = json.loads(out_path.read_text())
    assert out[0]["relevance_score"] == 9
    assert out[0]["relevance_reason"] == "FP4 quantization method"
    assert out[1]["relevance_score"] == 1


@pytest.mark.asyncio
async def test_filter_skips_papers_with_empty_metadata(tmp_path: Path, monkeypatch):
    """Trending-only stubs (e.g. hf_daily trending-rank-only entries) carry
    no title/abstract and must be skipped before reaching the LLM — otherwise
    Haiku invents scores from nothing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    stub = _mk("stub", "", "")
    real = _mk("real", "BitNet b1.58", "1-bit LLM with FP4 quantization.")
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in [stub, real]]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "p.md"
    prompt_path.write_text("score this")

    fake = AsyncMock()
    fake.call_json.return_value = {"relevance_score": 9, "reason": "good"}
    await filter_papers(in_path, out_path, prompt_path, fake, concurrency=2)

    out = json.loads(out_path.read_text())
    by_id = {p["id"]: p for p in out}
    assert by_id["stub"]["relevance_score"] is None
    assert by_id["real"]["relevance_score"] == 9
    assert fake.call_json.call_count == 1  # only the real paper was scored


@pytest.mark.asyncio
async def test_filter_handles_per_paper_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    deduped = [_mk("1", "good", "good"), _mk("2", "bad", "bad")]
    deduped_path = tmp_path / "in.json"
    deduped_path.write_text(json.dumps([p.model_dump(mode="json") for p in deduped]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.side_effect = [
        {"relevance_score": 8, "reason": "ok"},
        Exception("boom"),
    ]
    await filter_papers(deduped_path, out_path, prompt_path, fake, concurrency=2)
    out = json.loads(out_path.read_text())
    assert any(r["relevance_score"] == 8 for r in out)
    assert any(r["relevance_score"] is None for r in out)
