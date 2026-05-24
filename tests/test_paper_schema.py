from datetime import UTC, datetime

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
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL", "cs.LG"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(UTC))],
    )
    assert p.id == "2402.17764"
    assert p.relevance_score is None
    assert p.highlights == []
    assert p.seen_before is False
    assert p.code_url is None


def test_source_record_with_extras():
    r = SourceRecord(
        name="hf_daily",
        fetched_at=datetime.now(UTC),
        extras={"upvotes": 230, "num_comments": 67},
    )
    assert r.extras["upvotes"] == 230


def test_paper_round_trip_json():
    p = Paper(
        id="x",
        title="t",
        authors=[],
        abstract="a",
        url="https://x",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime(2026, 5, 11, tzinfo=UTC))],
    )
    js = p.model_dump_json()
    p2 = Paper.model_validate_json(js)
    assert p2.id == "x"
    assert p2.sources[0].name == "arxiv"


def test_invalid_source_name_rejected():
    with pytest.raises(ValueError):
        SourceRecord(name="not_a_source", fetched_at=datetime.now(UTC))


def test_relevance_breakdown_round_trips():
    breakdown = {
        "hard_gate": False,
        "topic_relevance": 5,
        "practicality": 4,
        "topic_bucket": "ptq",
        "compression_type": "quantization",
        "format_or_method": "MXFP4",
        "largest_model_tested": "Llama-3.1-70B",
        "accuracy_benchmarks": "MMLU,HellaSwag",
        "accuracy_summary": "+0.3 MMLU vs AWQ",
        "inference_perf": "1.8x A100",
        "calibration_cost": "128 samples",
        "peak_memory": "<24GB",
    }
    p = Paper(
        id="x",
        title="t",
        authors=[],
        abstract="a",
        url="https://x",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime(2026, 5, 11, tzinfo=UTC))],
    )
    p.relevance_breakdown = breakdown
    p2 = Paper.model_validate_json(p.model_dump_json())
    assert p2.relevance_breakdown == breakdown
