import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pipeline.weekly import render_weekly
from sources.base import Paper, SourceRecord


def _mk(id_, score, day_offset, hard_gate: bool = False):
    p = Paper(
        id=id_,
        title=f"Title {id_}",
        authors=[],
        abstract="a",
        url=f"https://x/{id_}",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC) - timedelta(days=day_offset),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(UTC))],
    )
    p.relevance_score = score
    p.relevance_breakdown = {"hard_gate": hard_gate, "topic_bucket": "ptq"}
    p.summary = "e"
    return p


def test_weekly_aggregates_past_seven_days(tmp_path: Path):
    summarized = tmp_path / "summarized"
    summarized.mkdir()
    for d in range(7):
        date = datetime(2026, 5, 11, tzinfo=UTC) - timedelta(days=d)
        papers = [_mk(f"d{d}p{i}", 9 - i, day_offset=d) for i in range(5)]
        (summarized / f"{date.strftime('%Y-%m-%d')}.json").write_text(
            json.dumps([p.model_dump(mode="json") for p in papers])
        )

    out_dir = tmp_path / "weekly"
    render_weekly(
        end_date=datetime(2026, 5, 11, tzinfo=UTC),
        summarized_root=summarized,
        out_dir=out_dir,
        top_n=20,
    )

    files = list(out_dir.glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "Top 20" in text
    assert "Per-source" in text
