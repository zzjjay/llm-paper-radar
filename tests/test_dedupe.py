import json
from datetime import UTC, datetime
from pathlib import Path

from pipeline.config import Config, DedupeConfig
from pipeline.dedupe import dedupe_for_date, merge_papers
from sources.base import Paper, SourceRecord


def _mk(id_, title, source, **extras):
    return Paper(
        id=id_,
        title=title,
        authors=[],
        abstract="abs",
        url=f"https://arxiv.org/abs/{id_}",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name=source, fetched_at=datetime.now(UTC), extras=extras)],
    )


def test_merge_papers_combines_sources_and_uses_priority():
    arxiv_p = _mk("X", "arxiv title", "arxiv")
    arxiv_p.abstract = "arxiv abs"
    hf_p = _mk("X", "hf title", "hf_daily", upvotes=42)
    hf_p.abstract = "hf abs"

    priority = [
        "hf_daily",
        "reddit",
        "arxiv",
    ]
    merged = merge_papers([arxiv_p, hf_p], priority)
    assert len(merged) == 1
    p = merged[0]
    assert p.title == "hf title"
    assert p.abstract == "hf abs"
    assert {s.name for s in p.sources} == {"arxiv", "hf_daily"}


def test_merge_keeps_distinct_papers():
    a = _mk("X", "x", "arxiv")
    b = _mk("Y", "y", "arxiv")
    merged = merge_papers([a, b], ["arxiv"])
    ids = {p.id for p in merged}
    assert ids == {"X", "Y"}


def test_dedupe_for_date_writes_files_and_marks_seen_before(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "2026-05-11"
    raw_dir.mkdir(parents=True)
    (raw_dir / "arxiv.json").write_text(
        json.dumps([_mk("X", "x", "arxiv").model_dump(mode="json")])
    )
    (raw_dir / "hf_daily.json").write_text(
        json.dumps([_mk("X", "x-hf", "hf_daily", upvotes=10).model_dump(mode="json")])
    )

    seen_path = tmp_path / "seen.json"
    out_path = tmp_path / "deduped" / "2026-05-11.json"
    out_path.parent.mkdir(parents=True)

    cfg = Config(
        dedupe=DedupeConfig(cross_day_strategy="lenient", source_priority=["hf_daily", "arxiv"])
    )
    n = dedupe_for_date(
        date=datetime(2026, 5, 11, tzinfo=UTC),
        raw_root=tmp_path / "raw",
        out_path=out_path,
        seen_path=seen_path,
        config=cfg,
    )
    assert n == 1
    data = json.loads(out_path.read_text())
    assert data[0]["title"] == "x-hf"
    assert data[0]["seen_before"] is False

    dedupe_for_date(
        date=datetime(2026, 5, 11, tzinfo=UTC),
        raw_root=tmp_path / "raw",
        out_path=out_path,
        seen_path=seen_path,
        config=cfg,
    )
    data2 = json.loads(out_path.read_text())
    assert data2[0]["seen_before"] is True
