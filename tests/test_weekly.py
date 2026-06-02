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
    )

    files = list(out_dir.glob("*.md"))
    assert len(files) == 1
    assert files[0].name == "20260505-20260511.md"
    text = files[0].read_text()

    # Header + compact table, no Top-N truncation, no per-source section.
    assert "Weekly Digest · 2026-05-05 → 2026-05-11" in text
    assert "Surfaced: 35 papers" in text
    assert "| # | Bucket | Paper | Authors | Date | Why |" in text
    assert "Per-source" not in text
    # All 35 papers present: count data rows (skip header + separator).
    data_rows = [
        ln for ln in text.splitlines() if ln.startswith("| ") and "Bucket" not in ln
    ]
    # 35 data rows + 1 separator row "|---|..."
    assert len([ln for ln in data_rows if not ln.startswith("|---")]) == 35

    # Why-column points to ../digests/<date>.md (relative from weekly/).
    assert "../digests/2026-05-11.md#p-d0p0" in text


def test_weekly_drops_hard_gated(tmp_path: Path):
    summarized = tmp_path / "summarized"
    summarized.mkdir()
    date = datetime(2026, 5, 11, tzinfo=UTC)
    papers = [
        _mk("keep", 9, day_offset=0),
        _mk("drop", 9, day_offset=0, hard_gate=True),
    ]
    (summarized / f"{date.strftime('%Y-%m-%d')}.json").write_text(
        json.dumps([p.model_dump(mode="json") for p in papers])
    )

    out_dir = tmp_path / "weekly"
    render_weekly(end_date=date, summarized_root=summarized, out_dir=out_dir)
    text = (out_dir / "20260505-20260511.md").read_text()
    assert "keep" in text
    assert "p-drop" not in text


def test_weekly_splices_into_readme(tmp_path: Path):
    summarized = tmp_path / "summarized"
    summarized.mkdir()
    date = datetime(2026, 5, 11, tzinfo=UTC)
    papers = [_mk("p1", 9, day_offset=0)]
    (summarized / f"{date.strftime('%Y-%m-%d')}.json").write_text(
        json.dumps([p.model_dump(mode="json") for p in papers])
    )

    readme = tmp_path / "README.md"
    readme.write_text(
        "# Radar\n\n<!-- LATEST_START -->\n\ndaily here\n\n<!-- LATEST_END -->\n\n"
        "---\n\n## Weekly\n\n<!-- WEEKLY_START -->\n<!-- WEEKLY_END -->\n\n## Scoring\n"
    )

    render_weekly(
        end_date=date,
        summarized_root=summarized,
        out_dir=tmp_path / "weekly",
        readme_path=readme,
    )

    text = readme.read_text()
    # Daily block untouched.
    assert "daily here" in text
    # Weekly table spliced between the WEEKLY markers, with root-relative links.
    weekly = text.partition("<!-- WEEKLY_START -->")[2].partition("<!-- WEEKLY_END -->")[0]
    assert "Weekly Digest · 2026-05-05 → 2026-05-11" in weekly
    assert "digests/2026-05-11.md#p-p1" in weekly
    assert "../digests/" not in weekly


def test_weekly_readme_inserts_block_when_markers_missing(tmp_path: Path):
    summarized = tmp_path / "summarized"
    summarized.mkdir()
    date = datetime(2026, 5, 11, tzinfo=UTC)
    (summarized / f"{date.strftime('%Y-%m-%d')}.json").write_text(
        json.dumps([_mk("p1", 9, day_offset=0).model_dump(mode="json")])
    )

    readme = tmp_path / "README.md"
    readme.write_text("# Radar\n\n<!-- LATEST_START -->\n\ndaily\n\n<!-- LATEST_END -->\n")

    render_weekly(
        end_date=date,
        summarized_root=summarized,
        out_dir=tmp_path / "weekly",
        readme_path=readme,
    )

    text = readme.read_text()
    assert "daily" in text
    assert "<!-- WEEKLY_START -->" in text
    assert "Weekly Digest · 2026-05-05 → 2026-05-11" in text
    # Weekly section comes after the daily block.
    assert text.index("<!-- LATEST_END -->") < text.index("<!-- WEEKLY_START -->")
