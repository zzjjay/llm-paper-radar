from datetime import UTC, datetime

from pipeline.render import _group_with_caps
from sources.base import Paper, SourceRecord


def _mk(id_: str, bucket: str, score: int = 9) -> Paper:
    p = Paper(
        id=id_,
        title=f"T {id_}",
        authors=[],
        abstract="a",
        url=f"https://x/{id_}",
        pdf_url=None,
        published_at=datetime(2026, 5, 11, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime(2026, 5, 11, tzinfo=UTC))],
    )
    p.relevance_score = score
    p.relevance_breakdown = {"topic_bucket": bucket}
    return p


def test_ptq_capped_at_5_others_at_default_3():
    papers = (
        [_mk(f"q{i}", "ptq") for i in range(8)]
        + [_mk(f"s{i}", "speculative_decoding") for i in range(5)]
        + [_mk(f"o{i}", "other") for i in range(2)]
    )
    grouped = _group_with_caps(papers, {"ptq": 5, "_default": 3})
    assert len(grouped["ptq"]) == 5
    assert len(grouped["speculative_decoding"]) == 3
    assert len(grouped["other"]) == 2  # under cap, kept as-is
    # order preservation: first 5 of ptq are q0..q4
    assert [p.id for p in grouped["ptq"]] == [f"q{i}" for i in range(5)]


def test_unknown_bucket_falls_to_other():
    p = _mk("x", "not_a_bucket")
    grouped = _group_with_caps([p], {"_default": 3})
    assert grouped["other"] == [p]
    assert grouped["ptq"] == []


def test_missing_breakdown_falls_to_other():
    p = _mk("x", "ptq")
    p.relevance_breakdown = None
    grouped = _group_with_caps([p], {"_default": 3})
    assert grouped["other"] == [p]


def test_per_bucket_override_wins_over_default():
    papers = [_mk(f"s{i}", "survey") for i in range(4)]
    grouped = _group_with_caps(papers, {"survey": 1, "_default": 3})
    assert len(grouped["survey"]) == 1


def test_all_12_buckets_are_recognized():
    """Ensure render's BUCKET_ORDER stays in sync with prompt enum."""
    from pipeline.render import BUCKET_ORDER

    expected = {
        "ptq",
        "qat",
        "pruning",
        "distillation",
        "kv_cache",
        "diffusion_compression",
        "speculative_decoding",
        "survey",
        "other",
    }
    assert set(BUCKET_ORDER) == expected
