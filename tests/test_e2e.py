import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.config import Config, DedupeConfig
from pipeline.dedupe import dedupe_for_date
from pipeline.filter import filter_papers
from pipeline.render import render_daily
from pipeline.summarize import summarize_papers
from sources.base import Paper, SourceRecord


def _mk_raw(id_, source, **extras):
    return Paper(
        id=id_,
        title=f"Paper {id_}",
        authors=["A"],
        abstract=f"abs {id_}",
        url=f"https://arxiv.org/abs/{id_}",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name=source, fetched_at=datetime.now(UTC), extras=extras)],
    )


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    date = datetime(2026, 5, 11, tzinfo=UTC)
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "2026-05-11"
    raw_dir.mkdir(parents=True)
    (raw_dir / "arxiv.json").write_text(
        json.dumps(
            [
                _mk_raw("X", "arxiv").model_dump(mode="json"),
                _mk_raw("Y", "arxiv").model_dump(mode="json"),
            ]
        )
    )
    (raw_dir / "hf_daily.json").write_text(
        json.dumps(
            [
                _mk_raw("X", "hf_daily", upvotes=20).model_dump(mode="json"),
            ]
        )
    )

    cfg = Config(
        dedupe=DedupeConfig(cross_day_strategy="lenient", source_priority=["hf_daily", "arxiv"])
    )

    deduped_path = tmp_path / "deduped" / "2026-05-11.json"
    deduped_path.parent.mkdir()
    seen_path = tmp_path / "seen.json"
    n = dedupe_for_date(date, raw_root, deduped_path, seen_path, cfg)
    assert n == 2

    scored_path = tmp_path / "scored" / "2026-05-11.json"
    scored_path.parent.mkdir()
    prompt = tmp_path / "p.md"
    prompt.write_text("score this")
    fake_filter = AsyncMock()
    fake_filter.call_json.side_effect = [
        {"relevance_score": 9, "reason": "good"},
        {"relevance_score": 4, "reason": "weak"},
    ]
    await filter_papers(deduped_path, scored_path, prompt, fake_filter, concurrency=2)

    summarized_path = tmp_path / "summarized" / "2026-05-11.json"
    summarized_path.parent.mkdir()
    sum_prompt = tmp_path / "s.md"
    sum_prompt.write_text("summarize")
    fake_sum = AsyncMock()
    fake_sum.call_json.return_value = {
        "summary": "English",
        "highlights": ["🎯 a"],
    }
    await summarize_papers(
        scored_path, summarized_path, sum_prompt, fake_sum, threshold=7, concurrency=2
    )
    assert fake_sum.call_json.call_count == 1

    digests = tmp_path / "digests"
    readme = tmp_path / "README.md"
    index = tmp_path / "INDEX.md"
    render_daily(date, summarized_path, digests, readme, index, full_top_n=10, threshold=7)

    out = (digests / "2026-05-11.md").read_text()
    assert "Paper X" in out
    assert "Paper Y" not in out
    assert readme.read_text() == out
