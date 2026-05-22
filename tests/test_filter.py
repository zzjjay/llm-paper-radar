import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.config import KeywordRule, PrefilterConfig
from pipeline.filter import filter_papers, prefilter_verdict
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


# -------- prefilter unit tests --------


def _prefilter() -> PrefilterConfig:
    return PrefilterConfig(
        enabled=True,
        whitelist=[
            KeywordRule(pattern="PTQ", weight=5),
            KeywordRule(pattern="quantization", weight=3),
        ],
        blacklist=[
            KeywordRule(pattern="image classification", weight=-4),
            KeywordRule(pattern="ImageNet", weight=-3),
        ],
        max_blacklist_hits=2,
    )


def test_prefilter_word_boundary_does_not_match_substring():
    """`QuIP` must not match inside `equipping`. The bug this guards against
    sent agent-loop papers into the PTQ bucket on the first dry-run."""
    p = _mk("x", "Equipping LLM agents with skills", "Agent framework for tool use.")
    cfg = PrefilterConfig(
        enabled=True,
        whitelist=[KeywordRule(pattern="QuIP", weight=5)],
        blacklist=[],
    )
    _gate, wl, _bl = prefilter_verdict(p, cfg)
    assert wl == []


def test_prefilter_gates_when_only_blacklist_hits():
    p = _mk(
        "x",
        "Pruning ResNet on ImageNet",
        "We study image classification with ResNet on ImageNet at 4-bit.",
    )
    gate, wl, bl = prefilter_verdict(p, _prefilter())
    assert gate is True
    assert wl == []
    assert "image classification" in bl and "ImageNet" in bl


def test_prefilter_keeps_paper_with_any_whitelist_hit():
    """Even with blacklist hits, a single whitelist match defers to the LLM."""
    p = _mk(
        "x",
        "PTQ on ImageNet for ResNet quantization",
        "We do PTQ on ResNet ImageNet quantization with image classification.",
    )
    gate, wl, bl = prefilter_verdict(p, _prefilter())
    assert gate is False
    assert "PTQ" in wl
    assert "ImageNet" in bl


@pytest.mark.asyncio
async def test_prefilter_skips_llm_call_for_obvious_off_topic(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    off_topic = _mk(
        "off",
        "ImageNet classification with ResNet",
        "We propose ResNet for image classification on ImageNet.",
    )
    on_topic = _mk(
        "on",
        "AWQ-style PTQ for Llama-70B",
        "Post-training quantization for Llama.",
    )
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in [off_topic, on_topic]]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "p.md"
    prompt_path.write_text("score this")

    fake = AsyncMock()
    fake.call_json.return_value = {
        "hard_gate": False,
        "topic_relevance": 5,
        "practicality": 4,
        "topic_bucket": "ptq",
        "reason": "ok",
    }
    await filter_papers(
        in_path,
        out_path,
        prompt_path,
        fake,
        concurrency=2,
        prefilter_cfg=_prefilter(),
    )

    out = {p["id"]: p for p in json.loads(out_path.read_text())}
    assert out["off"]["relevance_breakdown"]["hard_gate"] is True
    assert out["off"]["relevance_score"] == 0
    assert "prefilter" in out["off"]["relevance_reason"]
    assert out["on"]["relevance_score"] == 9
    # off-topic paper never hit the LLM
    assert fake.call_json.call_count == 1


# -------- existing filter behavior --------


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
        {
            "hard_gate": False,
            "topic_relevance": 5,
            "practicality": 4,
            "topic_bucket": "ptq",
            "compression_type": "ptq",
            "reason": "FP4 quantization method",
        },
        {
            "hard_gate": True,
            "topic_relevance": 0,
            "practicality": 0,
            "topic_bucket": "ptq",
            "reason": "RAG, not relevant",
        },
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
    assert out[0]["relevance_score"] == 9  # 5 + 4
    assert out[0]["relevance_reason"] == "FP4 quantization method"
    assert out[0]["relevance_breakdown"]["topic_bucket"] == "ptq"
    assert out[1]["relevance_score"] == 0  # hard_gate forces composite to 0


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
    fake.call_json.return_value = {
        "hard_gate": False,
        "topic_relevance": 5,
        "practicality": 4,
        "topic_bucket": "low_bits",
        "reason": "good",
    }
    await filter_papers(in_path, out_path, prompt_path, fake, concurrency=2)

    out = json.loads(out_path.read_text())
    by_id = {p["id"]: p for p in out}
    assert by_id["stub"]["relevance_score"] is None
    assert by_id["real"]["relevance_score"] == 9  # 5 + 4
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
        {
            "hard_gate": False,
            "topic_relevance": 4,
            "practicality": 4,
            "topic_bucket": "ptq",
            "reason": "ok",
        },
        Exception("boom"),
    ]
    await filter_papers(deduped_path, out_path, prompt_path, fake, concurrency=2)
    out = json.loads(out_path.read_text())
    by_id = {r["id"]: r for r in out}
    # Successful scoring path: composite = 4 + 4 = 8.
    assert by_id["1"]["relevance_score"] == 8
    # Failure path no longer silently drops with score=None — paper is
    # preserved as hard_gate=True with a diagnosable reason so it stays
    # visible downstream and can be retried.
    failed = by_id["2"]
    assert failed["relevance_score"] == 0
    assert failed["relevance_breakdown"]["hard_gate"] is True
    assert "judge unavailable" in failed["relevance_reason"]
