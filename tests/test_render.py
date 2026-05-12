import json
import math
from datetime import UTC, datetime
from pathlib import Path

from pipeline.render import heat_score, render_daily, render_index_line, sort_papers
from sources.base import Paper, SourceRecord


def _mk(
    id_,
    score,
    hf_upvotes=0,
    reddit_score=0,
    trending_rank=None,
    twitter_accounts=None,
    summary="en",
):  # noqa: E501
    now = datetime.now(UTC)
    sources = [
        SourceRecord(name="arxiv", fetched_at=now),
        SourceRecord(
            name="hf_daily",
            fetched_at=now,
            extras={"upvotes": hf_upvotes, "num_comments": 0},
        ),
        SourceRecord(
            name="reddit",
            fetched_at=now,
            extras={"score": reddit_score, "num_comments": 0},
        ),
    ]
    if trending_rank is not None:
        sources.append(
            SourceRecord(
                name="hf_daily",
                fetched_at=now,
                extras={"trending_rank": trending_rank},
            )
        )
    if twitter_accounts:
        sources.append(
            SourceRecord(
                name="twitter_rsshub",
                fetched_at=now,
                extras={"accounts": twitter_accounts},
            )
        )
    p = Paper(
        id=id_,
        title=f"Title {id_}",
        authors=["A"],
        abstract="abs",
        url=f"https://arxiv.org/abs/{id_}",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=sources,
    )
    p.relevance_score = score
    p.summary = summary
    p.highlights = ["🎯 hl"]
    return p


def test_heat_score_combines_all_signals():
    p = _mk("x", 9, hf_upvotes=10, reddit_score=99, trending_rank=2, twitter_accounts=["a", "b"])
    expected = (100 / 2) + 10 + math.log(100) * 5 + 10 * 2
    assert abs(heat_score(p) - expected) < 0.01


def test_heat_score_zero_when_no_signal():
    p = _mk("x", 9)
    assert heat_score(p) == 0.0


def test_trending_rank_above_30_gives_no_bonus():
    p = _mk("x", 9, trending_rank=50)
    assert heat_score(p) == 0.0


def test_sort_papers_heat_primary_relevance_tiebreaker():
    a = _mk("a", 9, hf_upvotes=0)
    b = _mk("b", 7, trending_rank=1)
    c = _mk("c", 9, hf_upvotes=5)
    sorted_ = sort_papers([a, b, c])
    assert [p.id for p in sorted_] == ["b", "c", "a"]


def test_sort_papers_relevance_outweighs_modest_heat():
    """A 9/10 paper with low heat should outrank an 8/10 paper with modest heat
    (composite score = heat + relevance*RELEVANCE_WEIGHT pulls relevance up)."""
    high_rel_low_heat = _mk("hr", 9, hf_upvotes=2)
    mid_rel_modest_heat = _mk("mr", 8, hf_upvotes=18)
    sorted_ = sort_papers([mid_rel_modest_heat, high_rel_low_heat])
    assert [p.id for p in sorted_] == ["hr", "mr"]


def test_render_daily_writes_top10_full_then_table(tmp_path: Path):
    summarized_path = tmp_path / "summarized.json"
    papers = [_mk(f"id{i}", 9, trending_rank=i + 1) for i in range(15)]
    below = _mk("low", 5)
    summarized_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers + [below]]))

    digests_dir = tmp_path / "digests"
    readme = tmp_path / "README.md"
    index = tmp_path / "INDEX.md"

    render_daily(
        date=datetime(2026, 5, 11, tzinfo=UTC),
        summarized_path=summarized_path,
        digests_dir=digests_dir,
        readme_path=readme,
        index_path=index,
        full_top_n=10,
        threshold=7,
    )

    out = (digests_dir / "2026-05-11.md").read_text()
    assert "## 🔥 Top 10" in out
    assert out.count("#### Summary") == 10
    assert "## 📚 Full List" in out
    assert "id|low" not in out and "Title low" not in out
    assert readme.read_text() == out
    assert "[05-11](digests/2026-05-11.md)" in index.read_text()


def test_render_index_line_includes_summary_stats():
    line = render_index_line(
        datetime(2026, 5, 11, tzinfo=UTC),
        scanned=487,
        passed=38,
        top_title="BitNet b1.58",
    )
    assert "[05-11](digests/2026-05-11.md)" in line
    assert "487" in line and "38" in line and "BitNet" in line
