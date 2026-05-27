from datetime import UTC, datetime, timedelta

import pytest
import respx
from httpx import Response

from sources.openreview import (
    OpenReviewSource,
    _content_value,
    _expand_year_templates,
    _note_pub,
    _note_to_paper,
)


def _make_note(note_id: str, title: str, days_ago: int, *, v2_format: bool = True) -> dict:
    """Build a fake OpenReview note. cdate is ms-since-epoch."""
    target = datetime.now(UTC) - timedelta(days=days_ago)
    cdate_ms = int(target.timestamp() * 1000)
    if v2_format:
        content = {
            "title": {"value": title},
            "abstract": {"value": f"Abstract for {title}."},
            "authors": {"value": ["Alice", "Bob"]},
        }
    else:
        content = {
            "title": title,
            "abstract": f"Abstract for {title}.",
            "authors": ["Alice", "Bob"],
        }
    return {"id": note_id, "cdate": cdate_ms, "content": content}


@respx.mock
@pytest.mark.asyncio
async def test_openreview_returns_papers_in_window():
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    inside = [_make_note(f"id-{i}", f"Inside {i}", days_ago=2) for i in range(3)]
    outside = [_make_note("id-old", "Old paper", days_ago=30)]
    respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}&offset=0&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(200, json={"notes": inside + outside}))

    src = OpenReviewSource(venues=[venue], window_days=7, max_pages=1)
    papers = await src.fetch(datetime.now(UTC))

    titles = {p.title for p in papers}
    assert titles == {"Inside 0", "Inside 1", "Inside 2"}
    assert "Old paper" not in titles
    # IDs are prefixed with "or-" to keep them distinct from arxiv ids during dedupe.
    assert all(p.id.startswith("or-") for p in papers)
    # Each paper carries the venue label in its source record extras.
    assert all(p.sources[0].extras["venue"] == venue for p in papers)
    # Venue label inferred from the first dotted segment, lowercased.
    assert all(p.primary_category == "test" for p in papers)


@respx.mock
@pytest.mark.asyncio
async def test_openreview_skips_failing_venue_continues_others():
    """A 500 on one venue must not block the others."""
    bad = "BAD.cc/2026/Conference"
    good = "GOOD.cc/2026/Conference"
    respx.get(
        f"https://api2.openreview.net/notes?invitation={bad}/-/Submission&offset=0&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(500, text="oops"))
    respx.get(
        f"https://api2.openreview.net/notes?invitation={good}/-/Submission&offset=0&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(200, json={"notes": [_make_note("ok1", "Good paper", days_ago=1)]}))

    src = OpenReviewSource(venues=[bad, good], window_days=7, max_pages=1)
    papers = await src.fetch(datetime.now(UTC))
    assert [p.title for p in papers] == ["Good paper"]


def test_content_value_unwraps_v2_and_passes_through_v1():
    assert _content_value({"title": {"value": "X"}}, "title") == "X"
    assert _content_value({"title": "X"}, "title") == "X"
    assert _content_value({}, "missing", default="d") == "d"


def test_note_pub_prefers_cdate_then_pdate_then_mdate():
    ts = 1758370688811  # 2025-09-20 UTC
    assert _note_pub({"cdate": ts}).year == 2025
    assert _note_pub({"pdate": ts}).year == 2025
    assert _note_pub({"mdate": ts}).year == 2025
    assert _note_pub({}) is None


def test_note_to_paper_skips_when_title_missing():
    note = {"id": "x", "content": {"abstract": {"value": "abs only"}}}
    assert _note_to_paper(note, "V.cc/2026/Conference", datetime.now(UTC)) is None


def test_expand_year_templates_expands_to_current_and_next_year():
    """`{year}` placeholder is filled with current year and current+1 so a venue
    config survives the calendar rollover without manual edits."""
    out = _expand_year_templates(
        ["ICLR.cc/{year}/Conference", "MLSys.org/{year}/Conference"],
        current_year=2026,
    )
    assert out == [
        "ICLR.cc/2026/Conference",
        "ICLR.cc/2027/Conference",
        "MLSys.org/2026/Conference",
        "MLSys.org/2027/Conference",
    ]


def test_expand_year_templates_passes_through_pinned_literals():
    """Literal venue strings (no `{year}`) are kept verbatim so a user can pin
    a specific past conference alongside templated current ones."""
    out = _expand_year_templates(
        ["NeurIPS.cc/2025/Conference", "ICLR.cc/{year}/Conference"],
        current_year=2026,
    )
    assert out == [
        "NeurIPS.cc/2025/Conference",
        "ICLR.cc/2026/Conference",
        "ICLR.cc/2027/Conference",
    ]


def test_expand_year_templates_dedups_overlap_preserving_order():
    """A literal that happens to equal the template's expansion is dropped on
    the second occurrence — first-seen wins so configured order is preserved."""
    out = _expand_year_templates(
        ["ICLR.cc/2026/Conference", "ICLR.cc/{year}/Conference"],
        current_year=2026,
    )
    assert out == ["ICLR.cc/2026/Conference", "ICLR.cc/2027/Conference"]
