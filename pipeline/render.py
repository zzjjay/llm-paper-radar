from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from pipeline.config import load_config
from sources.base import Paper

REPO_URL = "https://github.com/zhaolin-amd/llm-paper-radar"
TRENDING_BONUS_CAP = 30
RELEVANCE_WEIGHT = 20

BUCKET_ORDER = [
    "ptq",
    "qat",
    "kv_cache",
    "speculative_decoding",
    "distillation",
    "pruning",
    "diffusion_compression",
    "survey",
    "other",
]
BUCKET_TITLES = {
    "ptq": "PTQ (post-training quantization)",
    "qat": "QAT / low-bit pretraining",
    "pruning": "Pruning / sparsity",
    "distillation": "Knowledge distillation",
    "kv_cache": "KV cache compression",
    "diffusion_compression": "Diffusion compression",
    "speculative_decoding": "Speculative decoding",
    "survey": "Survey",
    "other": "Other",
}


def _bucket_of(p: Paper) -> str:
    bd = p.relevance_breakdown or {}
    bucket = bd.get("topic_bucket")
    if bucket in BUCKET_TITLES:
        return bucket
    return "other"


def _cap_for(bucket: str, caps: dict[str, int]) -> int:
    return caps.get(bucket, caps.get("_default", 3))


def _caps_summary(caps: dict[str, int]) -> str:
    """One-line human summary of the cap config, e.g. 'Up to 3 PTQ, 2 others'."""
    default = caps.get("_default", 3)
    overrides = [
        (bucket, n)
        for bucket, n in caps.items()
        if bucket != "_default" and n != default and bucket in BUCKET_TITLES
    ]
    overrides.sort(key=lambda kv: BUCKET_ORDER.index(kv[0]))
    parts = [f"{n} {bucket.upper().replace('_', ' ')}" for bucket, n in overrides]
    parts.append(f"{default} others")
    return "Up to " + ", ".join(parts)


def heat_score(p: Paper) -> float:
    """Heat = trending_bonus + hf_upvotes + log(reddit_score+1)*5 + twitter_account_bonus.
    Trending bonus = 100/rank for rank 1..30, else 0.
    Twitter bonus = 10 per distinct account that linked it."""
    trending_bonus = 0.0
    hf_upvotes = 0
    reddit_score = 0
    twitter_accounts: set[str] = set()
    for s in p.sources:
        if s.name == "hf_daily":
            if "trending_rank" in s.extras:
                rank = s.extras["trending_rank"]
                if rank and rank <= TRENDING_BONUS_CAP:
                    trending_bonus = max(trending_bonus, 100.0 / rank)
            if "upvotes" in s.extras:
                hf_upvotes = max(hf_upvotes, s.extras.get("upvotes", 0) or 0)
        elif s.name == "reddit":
            reddit_score = max(reddit_score, s.extras.get("score", 0) or 0)
        elif s.name == "twitter_rsshub":
            twitter_accounts.update(s.extras.get("accounts", []))
    return trending_bonus + hf_upvotes + math.log(reddit_score + 1) * 5 + 10 * len(twitter_accounts)


def composite_score(p: Paper) -> float:
    """Heat + relevance*WEIGHT. Lets a 9/10 paper outweigh ~20 HF upvotes,
    while a viral paper (trending #1 = +100 heat) can still surface lower-relevance work."""
    return heat_score(p) + (p.relevance_score or 0) * RELEVANCE_WEIGHT


def sort_papers(papers: list[Paper]) -> list[Paper]:
    """Composite primary; heat as tiebreaker."""
    return sorted(
        papers,
        key=lambda p: (composite_score(p), heat_score(p)),
        reverse=True,
    )


def _source_badge(p: Paper) -> str:
    """One badge per distinct source name; consolidate hf_daily's two extras shapes."""
    hf_up = hf_cm = 0
    hf_trend = None
    reddit_sc = reddit_cm = 0
    twitter_accs: set[str] = set()
    watched_authors: list[str] = []
    other: set[str] = set()
    for s in p.sources:
        if s.name == "hf_daily":
            if "upvotes" in s.extras:
                hf_up = max(hf_up, s.extras.get("upvotes", 0) or 0)
                hf_cm = max(hf_cm, s.extras.get("num_comments", 0) or 0)
            if "trending_rank" in s.extras:
                rank = s.extras["trending_rank"]
                hf_trend = rank if hf_trend is None else min(hf_trend, rank)
        elif s.name == "reddit":
            reddit_sc = max(reddit_sc, s.extras.get("score", 0) or 0)
            reddit_cm = max(reddit_cm, s.extras.get("num_comments", 0) or 0)
        elif s.name == "twitter_rsshub":
            twitter_accs.update(s.extras.get("accounts", []))
        elif s.name == "arxiv_authors":
            for n in s.extras.get("matched_authors", []):
                if n not in watched_authors:
                    watched_authors.append(n)
        else:
            other.add(s.name)
    parts = []
    if hf_up or hf_trend is not None or hf_cm:
        bits = []
        if hf_trend is not None:
            bits.append(f"📈 trending #{hf_trend}")
        if hf_up:
            bits.append(f"👍 {hf_up}")
        if hf_cm:
            bits.append(f"💬 {hf_cm}")
        parts.append(f"hf_daily ({', '.join(bits)})")
    if reddit_sc or reddit_cm:
        parts.append(f"reddit (🔥 {reddit_sc}, 💬 {reddit_cm})")
    if twitter_accs:
        parts.append(f"twitter ({', '.join(sorted(twitter_accs))})")
    if watched_authors:
        parts.append(f"watched ({', '.join(watched_authors)})")
    for o in sorted(other):
        parts.append(o)
    return ", ".join(parts)


def _watched_meta(p: Paper) -> tuple[list[str], list[str]] | None:
    """Return (matched_authors, affiliations) if paper is from arxiv_authors source."""
    authors: list[str] = []
    affs: list[str] = []
    for s in p.sources:
        if s.name != "arxiv_authors":
            continue
        for n in s.extras.get("matched_authors", []):
            if n not in authors:
                authors.append(n)
        for a in s.extras.get("affiliations", []):
            if a not in affs:
                affs.append(a)
    return (authors, affs) if authors else None


def _summary_block(summary: str | None, highlights: list[str]) -> str:
    if not summary:
        return "#### Summary\n*(Summary generation failed)*\n"
    hl = "\n".join(f"- {h}" for h in highlights)
    return f"#### Summary\n{summary}\n\n{hl}\n" if hl else f"#### Summary\n{summary}\n"


def _related_methods_block(p: Paper) -> str:
    if not p.related_methods:
        return ""
    lines = ["#### 📎 Related / compared methods"]
    for m in p.related_methods:
        name = m.get("name", "").strip()
        if not name:
            continue
        relation = (m.get("relation") or "").strip()
        arxiv_id = m.get("arxiv_id")
        head = f"[{name}](https://arxiv.org/abs/{arxiv_id})" if arxiv_id else name
        lines.append(f"- {head}{' — ' + relation if relation else ''}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _why_selected_line(p: Paper) -> str:
    """One-line rationale derived from the Haiku rubric breakdown."""
    bd = p.relevance_breakdown or {}
    bucket = BUCKET_TITLES.get(_bucket_of(p), "Other")
    topic = bd.get("topic_relevance")
    pract = bd.get("practicality")
    reason = (p.relevance_reason or "").strip()
    pieces = [f"**{bucket}**"]
    if topic is not None and pract is not None:
        pieces.append(f"topic {topic}/5 · practicality {pract}/5")
    score = p.relevance_score
    if score is not None:
        pieces.append(f"composite {score}/10")
    if reason:
        pieces.append(reason)
    return "#### 🧭 Why this paper\n" + " · ".join(pieces) + "\n"


def _signal_line(p: Paper) -> str:
    bd = p.relevance_breakdown or {}
    parts = []
    ctype = bd.get("compression_type")
    if ctype and ctype != "unknown":
        parts.append(str(ctype))
    fmt = bd.get("format_or_method")
    if fmt and fmt != "unknown":
        parts.append(str(fmt))
    largest = bd.get("largest_model_tested")
    if largest and largest != "unknown":
        parts.append(str(largest))
    cal = bd.get("calibration_cost")
    if cal and cal != "unknown":
        parts.append(f"cal: {cal}")
    perf = bd.get("inference_perf")
    if perf and perf != "unknown":
        parts.append(f"perf: {perf}")
    return f"🧪 {' · '.join(parts)}\n" if parts else ""


def _watched_block(rank: int, p: Paper) -> str:
    """Same as _full_block but prefixed with the watched-author/affiliation tag."""
    meta = _watched_meta(p)
    if meta is None:
        return _full_block(rank, p)
    matched, affs = meta
    tag_authors = ", ".join(matched)
    tag_affs = " — " + "; ".join(affs) if affs else ""
    header = f"> 👤 {tag_authors}{tag_affs}\n"
    block = _full_block(rank, p)
    # Splice the header after the title line so it sits with the metadata.
    title_line, _, rest = block.partition("\n")
    return f"{title_line}\n{header}{rest}"


def _full_block(rank: int, p: Paper) -> str:
    code_part = f" · [GitHub]({p.code_url})" if p.code_url else ""
    pdf_part = f" · [PDF]({p.pdf_url})" if p.pdf_url else ""
    revisited = " 🔁" if p.seen_before else ""
    primary_source = p.sources[0].name if p.sources else "unknown"
    cats = ", ".join(p.categories) if p.categories else p.primary_category
    authors_short = ", ".join(p.authors[:3]) + ("..." if len(p.authors) > 3 else "")
    summary_block = _summary_block(p.summary, p.highlights)
    signal_line = _signal_line(p)
    why_line = _why_selected_line(p)
    related = _related_methods_block(p)
    anchor = f'<a id="{_paper_anchor(p)}"></a>\n'
    return f"""{anchor}### {rank}. {p.title} ({p.relevance_score}/10){revisited}
**{primary_source}** · `{p.id}` · {p.published_at.date()}
👥 {authors_short} · 🏷 {cats}
🔗 [arXiv]({p.url}){pdf_part}{code_part}
📡 Sources: {_source_badge(p)}
{signal_line}
{why_line}
{summary_block}
{related}---
"""


def _paper_anchor(p: Paper) -> str:
    # arXiv IDs contain dots; build an explicit HTML-anchor-safe slug.
    return "p-" + re.sub(r"[^a-zA-Z0-9_-]", "-", p.id)


def _authors_short(p: Paper, limit: int = 3) -> str:
    if not p.authors:
        return "—"
    head = ", ".join(p.authors[:limit])
    return head + (" et al." if len(p.authors) > limit else "")


def _details_link(p: Paper, digest_filename: str) -> str:
    return f"[📄]({digest_filename}#{_paper_anchor(p)})"


def _compact_row(rank: int, p: Paper, digest_filename: str) -> str:
    """README/compact view: # | Paper | Authors | Date | Bucket | Details."""
    bucket = BUCKET_TITLES.get(_bucket_of(p), "Other")
    revisited = " 🔁" if p.seen_before else ""
    title_cell = f"[{p.title}]({p.url}){revisited}"
    date_str = p.published_at.strftime("%Y-%m-%d")
    return (
        f"| {rank} | {title_cell} | {_authors_short(p)} | {date_str}"
        f" | {bucket} | {_details_link(p, digest_filename)} |"
    )


def _compact_watched_row(rank: int, p: Paper, digest_filename: str) -> str:
    """Watched-author table adds a Matched column in place of Bucket."""
    meta = _watched_meta(p)
    matched = ", ".join(meta[0]) if meta else "—"
    revisited = " 🔁" if p.seen_before else ""
    title_cell = f"[{p.title}]({p.url}){revisited}"
    date_str = p.published_at.strftime("%Y-%m-%d")
    return (
        f"| {rank} | {title_cell} | {_authors_short(p)} | {date_str}"
        f" | {matched} | {_details_link(p, digest_filename)} |"
    )


LATEST_START = "<!-- LATEST_START -->"
LATEST_END = "<!-- LATEST_END -->"


def _splice_into_readme(readme_path: Path, digest_text: str) -> None:
    """Inject the rendered digest between the LATEST markers in README.md.
    If the markers are missing (or README is the legacy whole-digest layout),
    rewrite the README to the doc template with the markers in place."""
    if readme_path.exists():
        existing = readme_path.read_text()
    else:
        existing = ""
    if LATEST_START in existing and LATEST_END in existing:
        before, _, rest = existing.partition(LATEST_START)
        _, _, after = rest.partition(LATEST_END)
        new = f"{before}{LATEST_START}\n\n{digest_text}\n\n{LATEST_END}{after}"
        readme_path.write_text(new)
        return
    # Bootstrap: write the doc README with the digest spliced in. The static
    # template lives next to render.py so we don't need a separate doc system.
    template_path = Path(__file__).parent / "readme_template.md"
    template = template_path.read_text()
    new = template.replace(
        f"{LATEST_START}\n{LATEST_END}",
        f"{LATEST_START}\n\n{digest_text}\n\n{LATEST_END}",
    )
    readme_path.write_text(new)


def render_index_line(date: datetime, scanned: int, passed: int, top_title: str) -> str:
    day = date.strftime("%m-%d")
    full = date.strftime("%Y-%m-%d")
    return f"- [{day}](digests/{full}.md) — {scanned} scanned, {passed} passed, top: {top_title}"


def _group_with_caps(
    papers: list[Paper], topic_caps: dict[str, int]
) -> dict[str, list[Paper]]:
    """Bucket sorted papers and apply per-bucket caps. Input order is preserved."""
    grouped: dict[str, list[Paper]] = {b: [] for b in BUCKET_ORDER}
    for p in papers:
        grouped[_bucket_of(p)].append(p)
    return {b: ps[: _cap_for(b, topic_caps)] for b, ps in grouped.items()}


WATCHED_TABLE_HEADER = (
    "| # | Paper | Authors | Date | Matched | Details |\n"
    "|---|-------|---------|------|---------|---------|"
)
MAIN_TABLE_HEADER = (
    "| # | Paper | Authors | Date | Bucket | Details |\n"
    "|---|-------|---------|------|--------|---------|"
)


def _render_detail_md(
    date: datetime,
    scanned: int,
    threshold: int,
    watched_papers: list[Paper],
    grouped: dict[str, list[Paper]],
    topic_caps: dict[str, int],
    surviving: list[Paper],
    highlighted_total: int,
) -> str:
    body: list[str] = []
    body.append(f"# LLM Inference Optimization Daily · {date.strftime('%Y-%m-%d')}\n")
    body.append(f"> 📅 Window: {date.strftime('%Y-%m-%d')} (UTC daily)")
    body.append(
        f"> 📊 Scanned {scanned} papers → passed filter {len(surviving)}"
        f" → highlighted {highlighted_total} (threshold ≥{threshold})"
        f" · 👤 {len(watched_papers)} from watched authors"
    )
    body.append("")
    body.append(f"> Auto-generated daily digest from [llm-paper-radar]({REPO_URL}).")
    body.append(
        "> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml)"
        " · Powered by Claude Sonnet 4.6 · Compact tables live in"
        " [README.md](../README.md).\n"
    )

    rank = 0
    if watched_papers:
        body.append("## 👤 Watched authors\n")
        body.append(
            "_Papers by authors on the watchlist in [`config.yaml`](../config.yaml)"
            " under `sources.arxiv_authors.authors`. Surfaced regardless of score._\n"
        )
        for p in watched_papers:
            rank += 1
            body.append(_watched_block(rank, p))

    body.append("## 🔥 Highlights by topic\n")
    body.append(
        f"_{_caps_summary(topic_caps)} per topic — change in"
        " [`config.yaml`](../config.yaml) under `render.topic_caps`._\n"
    )
    for bucket in BUCKET_ORDER:
        papers = grouped[bucket]
        if not papers:
            continue
        body.append(f"### {BUCKET_TITLES[bucket]}\n")
        for p in papers:
            rank += 1
            body.append(_full_block(rank, p))

    return "\n".join(body)


def _render_compact_md(
    date: datetime,
    scanned: int,
    threshold: int,
    watched_papers: list[Paper],
    surviving: list[Paper],
    digest_filename: str,
) -> str:
    """Compact README view: header + watched table + main table + Revisited."""
    watched_ids = {p.id for p in watched_papers}
    main_rows = [p for p in surviving if p.id not in watched_ids]
    revisited = [p for p in main_rows if p.seen_before]

    body: list[str] = []
    body.append(f"# LLM Inference Optimization Daily · {date.strftime('%Y-%m-%d')}\n")
    body.append(f"> 📅 Window: {date.strftime('%Y-%m-%d')} (UTC daily)")
    body.append(
        f"> 📊 Scanned {scanned} papers → passed filter {len(surviving)}"
        f" (threshold ≥{threshold}) · 👤 {len(watched_papers)} from watched authors"
    )
    body.append("")
    body.append(
        f"> Tables only — full summaries / why-selected / related methods live in"
        f" the [detail page]({digest_filename})."
        f" History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml)"
        f" · Generated by [llm-paper-radar]({REPO_URL}).\n"
    )

    if watched_papers:
        body.append("## 👤 Watched authors\n")
        body.append(WATCHED_TABLE_HEADER)
        for i, p in enumerate(watched_papers, start=1):
            body.append(_compact_watched_row(i, p, digest_filename))
        body.append("")
    else:
        body.append("## 👤 Watched authors\n")
        body.append("_No papers from the watchlist in this window._\n")

    if main_rows:
        body.append("## 🔥 Highlighted papers\n")
        body.append(MAIN_TABLE_HEADER)
        for i, p in enumerate(main_rows, start=1):
            body.append(_compact_row(i, p, digest_filename))
        body.append("")
    else:
        body.append("## 🔥 Highlighted papers\n")
        body.append("_No papers passed the threshold today._\n")

    if revisited:
        body.append("## 🔁 Revisited\n")
        for p in revisited[:5]:
            body.append(f"- [{p.title}]({p.url}) — score {p.relevance_score}")
        body.append("")

    return "\n".join(body)


def render_daily(
    date: datetime,
    summarized_path: Path,
    digests_dir: Path,
    readme_path: Path,
    index_path: Path,
    threshold: int,
    topic_caps: dict[str, int] | None = None,
) -> None:
    if topic_caps is None:
        topic_caps = {"ptq": 3, "_default": 2}
    all_papers = [Paper.model_validate(p) for p in json.loads(summarized_path.read_text())]
    scanned = len(all_papers)
    watched_papers = sort_papers([p for p in all_papers if _watched_meta(p) is not None])
    watched_ids = {p.id for p in watched_papers}
    surviving = [p for p in all_papers if (p.relevance_score or 0) >= threshold]
    surviving = sort_papers(surviving)
    # Don't repeat watched papers in the topic-bucket highlights; they already
    # have a dedicated section above.
    topic_pool = [p for p in surviving if p.id not in watched_ids]
    grouped = _group_with_caps(topic_pool, topic_caps)
    highlighted_total = sum(len(ps) for ps in grouped.values())

    detail_text = _render_detail_md(
        date,
        scanned,
        threshold,
        watched_papers,
        grouped,
        topic_caps,
        surviving,
        highlighted_total,
    )
    digests_dir.mkdir(parents=True, exist_ok=True)
    digest_filename = f"{date.strftime('%Y-%m-%d')}.md"
    digest_path = digests_dir / digest_filename
    digest_path.write_text(detail_text)

    # README compact view links into the detail file via a relative path that
    # works from the repo root.
    compact_text = _render_compact_md(
        date,
        scanned,
        threshold,
        watched_papers,
        surviving,
        digest_filename=f"{digests_dir.name}/{digest_filename}",
    )
    _splice_into_readme(readme_path, compact_text)

    top_title = surviving[0].title if surviving else "(no papers passed)"
    new_line = render_index_line(date, scanned, len(surviving), top_title[:50])
    if index_path.exists():
        existing = index_path.read_text()
        marker = f"](digests/{date.strftime('%Y-%m-%d')}.md)"
        if marker in existing:
            updated_lines = []
            for ln in existing.splitlines():
                updated_lines.append(new_line if marker in ln else ln)
            index_path.write_text("\n".join(updated_lines) + "\n")
        else:
            index_path.write_text(existing + "\n" + new_line + "\n")
    else:
        index_path.write_text("# Digest History Index\n\n" + new_line + "\n")


if __name__ == "__main__":

    @click.command()
    @click.option("--date", default=None)
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--in-root", default="data/summarized", type=click.Path(path_type=Path))
    @click.option("--digests-dir", default="digests", type=click.Path(path_type=Path))
    @click.option("--readme", default="README.md", type=click.Path(path_type=Path))
    @click.option("--index", default="INDEX.md", type=click.Path(path_type=Path))
    def main(date, backfill_days, in_root, digests_dir, readme, index):
        cfg = load_config()
        if date:
            base = datetime.fromisoformat(date).replace(tzinfo=UTC)
        else:
            base = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            in_path = in_root / f"{target.strftime('%Y-%m-%d')}.json"
            if not in_path.exists():
                print(f"render: skip {target.date()}")
                continue
            render_daily(
                target,
                in_path,
                digests_dir,
                readme,
                index,
                cfg.filter.threshold,
                cfg.render.topic_caps,
            )
            print(f"render: wrote digest for {target.date()}")

    main()
