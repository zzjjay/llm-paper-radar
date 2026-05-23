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
RELEVANCE_WEIGHT = 30
# Star bonus = log(stars+1) * STAR_WEIGHT, capped at STAR_BONUS_CAP.
# Tuned so that ~1k-star repo ≈ HF trending #5; framework repos (10k+) cap
# out at 25 so they can't outweigh a 9/10 relevance paper (RELEVANCE_WEIGHT=30).
STAR_WEIGHT = 3.0
STAR_BONUS_CAP = 25.0

# Bucket enum. Kept in sync with prompts/relevance.md and tests/test_render_grouping.py.
# `trending` is the soft 8th catch-all for compression-adjacent decoding-
# acceleration work (parallel/dual-view drafters, etc.) that doesn't fit
# the seven strict compression buckets; the LLM can pick it directly, and
# it's also the landing spot for manual overrides of hf_daily-popular papers.
BUCKET_ORDER = [
    "ptq",
    "low_bits",
    "qat",
    "kv_cache",
    "pruning_distill",
    "diffusion",
    "survey",
    "trending",
]
BUCKET_TITLES = {
    "ptq": "PTQ",
    "low_bits": "Low-bit (≤ 2-bit)",
    "qat": "QAT",
    "kv_cache": "KV cache",
    "pruning_distill": "Pruning & distillation",
    "diffusion": "Diffusion",
    "survey": "Survey",
    "trending": "Trending",
}
# Detail-page section headers — tech terms stay English, glue words Chinese.
BUCKET_TITLES_CN = {
    "ptq": "PTQ（训练后量化）",
    "low_bits": "Low-bit（≤ 2-bit）",
    "qat": "QAT（量化感知训练）",
    "kv_cache": "KV cache 压缩",
    "pruning_distill": "Pruning / 蒸馏",
    "diffusion": "Diffusion",
    "survey": "Survey / 方法论与对比",
    "trending": "Trending / 高热度但未归桶",
}

# Sentinel returned by `_bucket_of` for papers whose `topic_bucket` is not
# one of the eight valid values in BUCKET_ORDER. These papers should have
# been hard-gated by the LLM, but defensively we still surface them in the
# compact table while excluding them from the bucketed "Highlights" section.
UNBUCKETED = "_unbucketed"


def _bucket_of(p: Paper) -> str:
    bd = p.relevance_breakdown or {}
    bucket = bd.get("topic_bucket")
    if bucket in BUCKET_TITLES:
        return bucket
    return UNBUCKETED


def _passed_gate(p: Paper) -> bool:
    """True when the paper has a real score and was not hard-gated.

    Replaces the old `relevance_score >= threshold` filter. There is no
    numeric threshold anymore — anything not hard-gated bubbles up; bucket
    caps in `render.topic_caps` control digest length."""
    if p.relevance_score is None:
        return False
    bd = p.relevance_breakdown or {}
    return not bd.get("hard_gate", False)


def _cap_for(bucket: str, caps: dict[str, int]) -> int:
    return caps.get(bucket, caps.get("_default", 3))


def _caps_summary(caps: dict[str, int]) -> str:
    """One-line CN summary of the cap config, e.g. 'PTQ 3 篇，其它 2 篇'."""
    default = caps.get("_default", 3)
    overrides = [
        (bucket, n)
        for bucket, n in caps.items()
        if bucket != "_default" and n != default and bucket in BUCKET_TITLES
    ]
    overrides.sort(key=lambda kv: BUCKET_ORDER.index(kv[0]))
    parts = [f"{bucket.upper().replace('_', ' ')} {n} 篇" for bucket, n in overrides]
    parts.append(f"其它 {default} 篇")
    return "，".join(parts)


def _star_bonus(p: Paper) -> float:
    """Capped log-scaled bonus from GitHub stars of the paper's official repo.
    Only HF-sourced papers carry `code_meta.stars` today (HF API includes it
    for free). arXiv/openreview/reddit-only papers contribute 0 here."""
    meta = p.code_meta or {}
    stars = meta.get("stars")
    if not isinstance(stars, int) or stars <= 0:
        return 0.0
    return min(math.log(stars + 1) * STAR_WEIGHT, STAR_BONUS_CAP)


def heat_score(p: Paper) -> float:
    """Heat = trending_bonus + hf_upvotes + log(reddit_score+1)*5 + star_bonus.
    Trending bonus = 100/rank for rank 1..30, else 0."""
    trending_bonus = 0.0
    hf_upvotes = 0
    reddit_score = 0
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
    return trending_bonus + hf_upvotes + math.log(reddit_score + 1) * 5 + _star_bonus(p)


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
        return "#### 摘要\n*（摘要生成失败）*\n"
    hl = "\n".join(f"- {h}" for h in highlights)
    return f"#### 摘要\n{summary}\n\n{hl}\n" if hl else f"#### 摘要\n{summary}\n"


def _related_methods_block(p: Paper) -> str:
    if not p.related_methods:
        return ""
    lines = ["#### 📎 相关方法 / 对比基线"]
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
    bucket = BUCKET_TITLES_CN.get(_bucket_of(p), "其它")
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
    return "#### 🧭 为什么推送这篇\n" + " · ".join(pieces) + "\n"


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


ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def _links_line(p: Paper) -> str:
    """Detail-page link row. arXiv-shaped IDs get arXiv + alphaxiv; HF is
    only added when the paper was actually surfaced via hf_daily (otherwise
    huggingface.co/papers/<id> 404s). OpenReview-native papers (id starts
    with `or-`) get the forum link. GitHub is appended whenever known."""
    parts: list[str] = []
    if ARXIV_ID_RE.match(p.id):
        parts.append(f"[arXiv]({p.url})")
        parts.append(f"[alphaxiv](https://www.alphaxiv.org/abs/{p.id})")
        if any(s.name == "hf_daily" for s in p.sources):
            parts.append(f"[HF](https://huggingface.co/papers/{p.id})")
    else:
        parts.append(f"[OpenReview]({p.url})")
    if p.code_url:
        parts.append(f"[GitHub]({p.code_url})")
    return " · ".join(parts)


def _full_block(rank: int, p: Paper) -> str:
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
🔗 {_links_line(p)}
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


def _bucket_cell(p: Paper) -> str:
    """Bucket column value: the topic bucket title. Watched-author papers
    sort into their topic bucket alongside everyone else (the detail page's
    dedicated 👤 section still highlights them)."""
    b = _bucket_of(p)
    if b == UNBUCKETED:
        # Show the raw bucket the LLM returned, so debugging stale enums
        # is obvious. Falls back to "—" if no breakdown at all.
        raw = (p.relevance_breakdown or {}).get("topic_bucket") or "—"
        return f"? {raw}"
    return BUCKET_TITLES[b]


def _compact_row(rank: int, p: Paper, digest_filename: str) -> str:
    """README/compact view: # | Bucket | Paper | Authors | Date | Why."""
    revisited = " 🔁" if p.seen_before else ""
    title_cell = f"[{p.title}]({p.url}){revisited}"
    date_str = p.published_at.strftime("%Y-%m-%d")
    return (
        f"| {rank} | {_bucket_cell(p)} | {title_cell} | {_authors_short(p)}"
        f" | {date_str} | {_details_link(p, digest_filename)} |"
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
    """Bucket sorted papers and apply per-bucket caps. Input order is preserved.
    Papers whose bucket is not one of the eight valid values in BUCKET_ORDER
    are dropped from the bucketed view (they still appear in the compact table)."""
    grouped: dict[str, list[Paper]] = {b: [] for b in BUCKET_ORDER}
    for p in papers:
        b = _bucket_of(p)
        if b == UNBUCKETED:
            continue
        grouped[b].append(p)
    return {b: ps[: _cap_for(b, topic_caps)] for b, ps in grouped.items()}


MAIN_TABLE_HEADER = (
    "| # | Bucket | Paper | Authors | Date | Why |\n"
    "|---|--------|-------|---------|------|---------|"
)


def _render_detail_md(
    date: datetime,
    scanned: int,
    watched_papers: list[Paper],
    grouped: dict[str, list[Paper]],
    topic_caps: dict[str, int],
    surviving: list[Paper],
    highlighted_total: int,
) -> str:
    body: list[str] = []
    body.append(f"# LLM 推理优化日报 · {date.strftime('%Y-%m-%d')}\n")
    body.append(f"> 📅 窗口：{date.strftime('%Y-%m-%d')}（UTC daily）")
    body.append(
        f"> 📊 扫描 {scanned} 篇 → 未被 hard_gate {len(surviving)}"
        f" → 精选 {highlighted_total}（按主题上限）"
        f" · 👤 {len(watched_papers)} 篇来自关注作者"
    )
    body.append("")
    body.append(f"> 自动生成自 [llm-paper-radar]({REPO_URL}).")
    body.append(
        "> 历史归档：[INDEX.md](INDEX.md) · 配置：[config.yaml](../config.yaml)"
        " · 摘要模型 Claude Opus 4.7 · 紧凑表格视图见"
        " [README.md](../README.md)。\n"
    )

    rank = 0
    if watched_papers:
        body.append("## 👤 关注作者\n")
        body.append(
            "_作者白名单配置在 [`config.yaml`](../config.yaml) 的"
            " `sources.arxiv_authors.authors`；这些论文绕过主题上限，全部展示。_\n"
        )
        for p in watched_papers:
            rank += 1
            body.append(_watched_block(rank, p))

    body.append("## 🔥 主题精选\n")
    body.append(
        f"_每个主题最多展示：{_caps_summary(topic_caps)}；可在"
        " [`config.yaml`](../config.yaml) 的 `render.topic_caps` 修改。_\n"
    )
    for bucket in BUCKET_ORDER:
        papers = grouped[bucket]
        if not papers:
            continue
        body.append(f"### {BUCKET_TITLES_CN[bucket]}\n")
        for p in papers:
            rank += 1
            body.append(_full_block(rank, p))

    return "\n".join(body)


def _bucket_sort_key(p: Paper) -> tuple[int, float, float]:
    """Sort key for the compact README table: bucket position first (in
    BUCKET_ORDER), then composite score desc, then heat desc. Papers with
    an unknown/legacy bucket sink to the end."""
    b = _bucket_of(p)
    bucket_idx = BUCKET_ORDER.index(b) if b in BUCKET_ORDER else len(BUCKET_ORDER)
    return (bucket_idx, -composite_score(p), -heat_score(p))


def _render_compact_md(
    date: datetime,
    scanned: int,
    watched_papers: list[Paper],
    surviving: list[Paper],
    digest_filename: str,
) -> str:
    """Compact README view: header + one bucketed table covering everything
    that passed hard_gate. Watched-author papers sort into their topic bucket
    alongside everyone else; the detail page's dedicated 👤 section still
    highlights them. The 🔁 marker on titles signals previously-seen papers."""
    # Watched-author papers are visibility insurance ("never miss"), but a
    # paper the LLM judge has marked hard_gate=True (out of scope) is
    # noise — keep it out of the main table even if a watched author is on
    # the author list. The detail page still lists watched papers in its
    # dedicated section.
    combined = sorted(surviving, key=_bucket_sort_key)
    watched_count = sum(1 for p in combined if _watched_meta(p) is not None)

    body: list[str] = []
    body.append(f"# LLM Inference Optimization Daily · {date.strftime('%Y-%m-%d')}\n")
    body.append(f"> 📅 Window: {date.strftime('%Y-%m-%d')} (UTC daily)")
    body.append(
        f"> 📊 Scanned {scanned} papers → passed hard_gate {len(surviving)}"
        f" · 👤 {watched_count} from watched authors"
    )
    body.append("")
    body.append(
        f"> Table only — full summaries / why-selected / related methods live in"
        f" the [detail page]({digest_filename})."
        f" History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml)"
        f" · Generated by [llm-paper-radar]({REPO_URL}).\n"
    )

    body.append("## 📚 Papers\n")
    if combined:
        body.append(MAIN_TABLE_HEADER)
        for i, p in enumerate(combined, start=1):
            body.append(_compact_row(i, p, digest_filename))
        body.append("")
    else:
        body.append("_Nothing surfaced today (everything was hard-gated)._\n")

    return "\n".join(body)


def _compute_day(
    summarized_path: Path,
) -> tuple[int, list[Paper], list[Paper]]:
    """Read a summarized JSON and return (scanned, watched, surviving).
    Side-effect free — does NOT write a digest file. Use this from any
    caller that only needs the in-memory triple (e.g. rollup queries)."""
    all_papers = [Paper.model_validate(p) for p in json.loads(summarized_path.read_text())]
    scanned = len(all_papers)
    watched_papers = sort_papers([p for p in all_papers if _watched_meta(p) is not None])
    surviving = sort_papers([p for p in all_papers if _passed_gate(p)])
    return scanned, watched_papers, surviving


def _load_day(
    date: datetime,
    summarized_path: Path,
    digests_dir: Path,
    topic_caps: dict[str, int],
) -> tuple[int, list[Paper], list[Paper]]:
    """Write the per-day digest file and return (scanned, watched, surviving)
    for downstream README rendering."""
    scanned, watched_papers, surviving = _compute_day(summarized_path)
    watched_ids = {p.id for p in watched_papers}
    topic_pool = [p for p in surviving if p.id not in watched_ids]
    grouped = _group_with_caps(topic_pool, topic_caps)
    highlighted_total = sum(len(ps) for ps in grouped.values())

    detail_text = _render_detail_md(
        date, scanned, watched_papers, grouped, topic_caps, surviving, highlighted_total
    )
    digests_dir.mkdir(parents=True, exist_ok=True)
    (digests_dir / f"{date.strftime('%Y-%m-%d')}.md").write_text(detail_text)
    return scanned, watched_papers, surviving


def _update_index(
    date: datetime, index_path: Path, scanned: int, surviving: list[Paper]
) -> None:
    top_title = surviving[0].title if surviving else "(no papers passed)"
    new_line = render_index_line(date, scanned, len(surviving), top_title[:50])
    if index_path.exists():
        existing = index_path.read_text()
        marker = f"](digests/{date.strftime('%Y-%m-%d')}.md)"
        if marker in existing:
            updated_lines = [new_line if marker in ln else ln for ln in existing.splitlines()]
            index_path.write_text("\n".join(updated_lines) + "\n")
        else:
            index_path.write_text(existing + "\n" + new_line + "\n")
    else:
        index_path.write_text("# Digest History Index\n\n" + new_line + "\n")


def render_daily(
    date: datetime,
    summarized_path: Path,
    digests_dir: Path,
    readme_path: Path,
    index_path: Path,
    topic_caps: dict[str, int] | None = None,
) -> None:
    if topic_caps is None:
        topic_caps = {"ptq": 5, "_default": 2}
    scanned, watched_papers, surviving = _load_day(
        date, summarized_path, digests_dir, topic_caps
    )
    digest_filename = f"{date.strftime('%Y-%m-%d')}.md"
    compact_text = _render_compact_md(
        date,
        scanned,
        watched_papers,
        surviving,
        digest_filename=f"{digests_dir.name}/{digest_filename}",
    )
    _splice_into_readme(readme_path, compact_text)
    _update_index(date, index_path, scanned, surviving)


def _render_aggregated_compact_md(
    days: list[tuple[datetime, int, list[Paper], list[Paper]]],
    digests_dir_name: str,
) -> str:
    """N-day rollup compact view: one merged table sorted by bucket (then the
    per-bucket sort key). Watched-author papers sort into their topic bucket
    alongside everyone else. Each row links to its own day's digest page."""
    days_sorted = sorted(days, key=lambda x: x[0], reverse=True)  # newest first
    start_date = min(d[0] for d in days)
    end_date = max(d[0] for d in days)
    total_scanned = sum(s for _, s, _, _ in days)

    pairs: list[tuple[datetime, Paper]] = []
    seen_ids: set[str] = set()
    # Watched-author papers are visibility insurance ("never miss"), but a
    # paper the LLM judge has marked hard_gate=True (out of scope) is
    # noise — keep it out of the rollup table even if a watched author is
    # on the author list.
    for date, _, _watched, surviving in days_sorted:
        for p in surviving:
            if p.id in seen_ids:
                continue
            seen_ids.add(p.id)
            pairs.append((date, p))
    pairs.sort(key=lambda dp: _bucket_sort_key(dp[1]))
    watched_count = sum(1 for _, p in pairs if _watched_meta(p) is not None)

    span = f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"
    body: list[str] = []
    body.append(f"# LLM Inference Optimization Daily · {len(days)}-day · {span}\n")
    body.append(f"> 📅 Window: {span}")
    body.append(
        f"> 📊 Scanned {total_scanned} papers → surfaced {len(pairs)}"
        f" · 👤 {watched_count} from watched authors"
    )
    body.append("")
    body.append(
        "> Table only — full summaries / why-selected / related methods live in the"
        f" per-day detail pages under [`{digests_dir_name}/`]({digests_dir_name}/)."
        " History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml)"
        f" · Generated by [llm-paper-radar]({REPO_URL}).\n"
    )
    body.append(f"## 📚 Papers ({len(days)}-day rollup)\n")
    if pairs:
        body.append(MAIN_TABLE_HEADER)
        for i, (date, p) in enumerate(pairs, start=1):
            digest_filename = f"{digests_dir_name}/{date.strftime('%Y-%m-%d')}.md"
            body.append(_compact_row(i, p, digest_filename))
        body.append("")
    else:
        body.append("_Nothing surfaced in this window._\n")
    return "\n".join(body)


def render_aggregated(
    targets: list[tuple[datetime, Path]],
    digests_dir: Path,
    readme_path: Path,
    index_path: Path,
    topic_caps: dict[str, int] | None = None,
) -> None:
    """Render each day's digest + INDEX entry, then splice one merged
    N-day table into README."""
    if topic_caps is None:
        topic_caps = {"ptq": 5, "_default": 2}
    days: list[tuple[datetime, int, list[Paper], list[Paper]]] = []
    for date, in_path in targets:
        scanned, watched, surviving = _load_day(date, in_path, digests_dir, topic_caps)
        _update_index(date, index_path, scanned, surviving)
        days.append((date, scanned, watched, surviving))
        print(f"render: wrote digest for {date.date()}")
    if not days:
        return
    aggregated = _render_aggregated_compact_md(days, digests_dir.name)
    _splice_into_readme(readme_path, aggregated)


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

        targets: list[tuple[datetime, Path]] = []
        for delta in reversed(range(backfill_days + 1)):
            target = base - timedelta(days=delta)
            in_path = in_root / f"{target.strftime('%Y-%m-%d')}.json"
            if not in_path.exists():
                print(f"render: skip {target.date()}")
                continue
            targets.append((target, in_path))

        if backfill_days > 0:
            render_aggregated(targets, digests_dir, readme, index, cfg.render.topic_caps)
        else:
            for target, in_path in targets:
                render_daily(target, in_path, digests_dir, readme, index, cfg.render.topic_caps)
                print(f"render: wrote digest for {target.date()}")

    main()
