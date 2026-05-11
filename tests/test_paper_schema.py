from datetime import datetime, timezone
import pytest
from sources.base import Paper, SourceRecord


def test_paper_minimal_construction():
    p = Paper(
        id="2402.17764",
        title="BitNet b1.58",
        authors=["Shuming Ma"],
        abstract="We introduce a 1-bit LLM variant.",
        url="https://arxiv.org/abs/2402.17764",
        pdf_url="https://arxiv.org/pdf/2402.17764.pdf",
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL",
        categories=["cs.CL", "cs.LG"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(timezone.utc))],
    )
    assert p.id == "2402.17764"
    assert p.relevance_score is None
    assert p.highlights_zh == []
    assert p.seen_before is False
    assert p.code_url is None


def test_source_record_with_extras():
    r = SourceRecord(
        name="reddit",
        fetched_at=datetime.now(timezone.utc),
        extras={"score": 230, "num_comments": 67},
    )
    assert r.extras["score"] == 230


def test_paper_round_trip_json():
    p = Paper(
        id="x",
        title="t",
        authors=[],
        abstract="a",
        url="https://x",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime(2026, 5, 11, tzinfo=timezone.utc))],
    )
    js = p.model_dump_json()
    p2 = Paper.model_validate_json(js)
    assert p2.id == "x"
    assert p2.sources[0].name == "arxiv"


def test_invalid_source_name_rejected():
    with pytest.raises(ValueError):
        SourceRecord(name="not_a_source", fetched_at=datetime.now(timezone.utc))
