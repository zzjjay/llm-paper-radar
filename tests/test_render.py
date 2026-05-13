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
    topic_bucket="ptq",
    topic_relevance=5,
    practicality=4,
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
    p.relevance_breakdown = {
        "topic_bucket": topic_bucket,
        "topic_relevance": topic_relevance,
        "practicality": practicality,
        "compression_type": "quantization",
        "format_or_method": "W4A16",
        "largest_model_tested": "Llama-3-70B",
        "calibration_cost": "data-free",
        "inference_perf": "1.5x",
    }
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


def test_render_daily_groups_by_topic_with_caps(tmp_path: Path):
    summarized_path = tmp_path / "summarized.json"
    # 15 PTQ papers (cap 5) + 1 below threshold
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
        threshold=7,
        topic_caps={"ptq": 5, "_default": 3},
    )

    out = (digests_dir / "2026-05-11.md").read_text()
    assert "## 🔥 Highlights by topic" in out
    assert "### PTQ (post-training quantization) (top 5 of cap 5)" in out
    # Caps apply to highlights, not full list. 15 pass threshold; cap surfaces 5.
    assert out.count("#### Summary") == 5
    assert "## 📚 Full List" in out
    # Below-threshold paper excluded from both highlights and full list.
    assert "Title low" not in out
    # README is bootstrapped from the doc template with the digest spliced in,
    # so it contains the digest body but isn't byte-equal to it.
    readme_text = readme.read_text()
    assert "<!-- LATEST_START -->" in readme_text
    assert "## 🔥 Highlights by topic" in readme_text
    assert "[05-11](digests/2026-05-11.md)" in index.read_text()


def test_render_daily_splices_into_existing_readme_markers(tmp_path: Path):
    """README has docs + LATEST markers; render should splice digest between
    them and leave the docs intact (must not overwrite the whole README)."""
    summarized_path = tmp_path / "summarized.json"
    papers = [_mk(f"id{i}", 9, trending_rank=i + 1) for i in range(3)]
    summarized_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers]))

    readme = tmp_path / "README.md"
    readme.write_text(
        "# My Radar\n\n## Docs above\nintro text\n\n"
        "<!-- LATEST_START -->\nold digest goes here\n<!-- LATEST_END -->\n\n"
        "## Docs below\ntrailing text\n"
    )

    render_daily(
        date=datetime(2026, 5, 11, tzinfo=UTC),
        summarized_path=summarized_path,
        digests_dir=tmp_path / "digests",
        readme_path=readme,
        index_path=tmp_path / "INDEX.md",
        threshold=7,
        topic_caps={"ptq": 3, "_default": 2},
    )

    content = readme.read_text()
    assert content.startswith("# My Radar\n")
    assert "## Docs above\nintro text" in content
    assert "## Docs below\ntrailing text" in content
    assert "old digest goes here" not in content  # replaced
    assert "## 🔥 Highlights by topic" in content  # new digest spliced in


def test_render_index_line_includes_summary_stats():
    line = render_index_line(
        datetime(2026, 5, 11, tzinfo=UTC),
        scanned=487,
        passed=38,
        top_title="BitNet b1.58",
    )
    assert "[05-11](digests/2026-05-11.md)" in line
    assert "487" in line and "38" in line and "BitNet" in line
