from datetime import UTC, datetime

from pipeline.render import _group_with_caps
from sources.base import Paper, SourceRecord


def _mk(id_: str, bucket: str, score: int = 9, hard_gate: bool = False) -> Paper:
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
    p.relevance_breakdown = {"topic_bucket": bucket, "hard_gate": hard_gate}
    return p


def test_ptq_capped_at_5_others_at_default_3():
    papers = (
        [_mk(f"q{i}", "ptq") for i in range(8)]
        + [_mk(f"k{i}", "kv_cache") for i in range(5)]
    )
    grouped = _group_with_caps(papers, {"ptq": 5, "_default": 3})
    assert len(grouped["ptq"]) == 5
    assert len(grouped["kv_cache"]) == 3
    # order preservation: first 5 of ptq are q0..q4
    assert [p.id for p in grouped["ptq"]] == [f"q{i}" for i in range(5)]


def test_unknown_bucket_is_dropped_from_grouped_view():
    """`other` and `speculative_decoding` are not valid buckets; papers
    carrying these tags should be dropped from the bucketed highlights
    section. They still flow through `surviving` for the compact table
    (tested separately in test_render). `survey` IS a valid bucket as
    of the methodology/comparison expansion."""
    legacy = _mk("legacy", "speculative_decoding")
    valid = _mk("v", "ptq")
    grouped = _group_with_caps([legacy, valid], {"_default": 3})
    assert grouped["ptq"] == [valid]
    assert "speculative_decoding" not in grouped
    assert "other" not in grouped


def test_missing_breakdown_is_dropped_from_grouped_view():
    p = _mk("x", "ptq")
    p.relevance_breakdown = None
    grouped = _group_with_caps([p], {"_default": 3})
    assert all(len(ps) == 0 for ps in grouped.values())


def test_per_bucket_override_wins_over_default():
    papers = [_mk(f"q{i}", "qat") for i in range(4)]
    grouped = _group_with_caps(papers, {"qat": 1, "_default": 3})
    assert len(grouped["qat"]) == 1


def test_bucket_order_matches_bucket_enum():
    """Ensure render's BUCKET_ORDER stays in sync with prompts/relevance.md.

    `trending` is render-only (manual override target) and not part of the
    LLM classification enum, but it still belongs in BUCKET_ORDER so the
    renderer can emit a Trending section."""
    from pipeline.render import BUCKET_ORDER

    expected = {
        "ptq",
        "low_bits",
        "qat",
        "kv_cache",
        "pruning_distill",
        "diffusion",
        "survey",
        "trending",
    }
    assert set(BUCKET_ORDER) == expected
