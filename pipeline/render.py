from __future__ import annotations

import json
import math
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
    "pruning",
    "distillation",
    "kv_cache",
    "diffusion_compression",
    "speculative_decoding",
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
    for o in sorted(other):
        parts.append(o)
    return ", ".join(parts)


def _summary_block(summary: str | None, highlights: list[str]) -> str:
    if not summary:
        return "#### Summary\n*(Summary generation failed)*\n"
    hl = "\n".join(f"- {h}" for h in highlights)
    return f"#### Summary\n{summary}\n\n{hl}\n" if hl else f"#### Summary\n{summary}\n"


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


def _full_block(rank: int, p: Paper) -> str:
    code_part = f" · [GitHub]({p.code_url})" if p.code_url else ""
    pdf_part = f" · [PDF]({p.pdf_url})" if p.pdf_url else ""
    revisited = " 🔁" if p.seen_before else ""
    primary_source = p.sources[0].name if p.sources else "unknown"
    cats = ", ".join(p.categories) if p.categories else p.primary_category
    authors_short = ", ".join(p.authors[:3]) + ("..." if len(p.authors) > 3 else "")
    summary_block = _summary_block(p.summary, p.highlights)
    signal_line = _signal_line(p)
    return f"""### {rank}. {p.title} ({p.relevance_score}/10){revisited}
**{primary_source}** · `{p.id}` · {p.published_at.date()}
👥 {authors_short} · 🏷 {cats}
🔗 [arXiv]({p.url}){pdf_part}{code_part}
📡 Sources: {_source_badge(p)}
{signal_line}
{summary_block}
---
"""


def _table_row(rank: int, p: Paper) -> str:
    sources = "+".join({s.name for s in p.sources})
    code = "✅" if p.code_url else "—"
    date_str = p.published_at.strftime("%m-%d")
    bd = p.relevance_breakdown or {}
    bucket = BUCKET_TITLES.get(_bucket_of(p), "Other")
    topic = bd.get("topic_relevance")
    pract = bd.get("practicality")
    topic_str = "—" if topic is None else str(topic)
    pract_str = "—" if pract is None else str(pract)
    return (
        f"| {rank} | [{p.title}]({p.url}) | {p.relevance_score} | {topic_str} | {pract_str}"
        f" | {bucket} | {sources} | {code} | {date_str} |"
    )


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
        topic_caps = {"_default": 3}
    all_papers = [Paper.model_validate(p) for p in json.loads(summarized_path.read_text())]
    scanned = len(all_papers)
    surviving = [p for p in all_papers if (p.relevance_score or 0) >= threshold]
    surviving = sort_papers(surviving)
    grouped = _group_with_caps(surviving, topic_caps)
    highlighted_total = sum(len(ps) for ps in grouped.values())
    revisited = [p for ps in grouped.values() for p in ps if p.seen_before]

    body = []
    body.append(f"# LLM Inference Optimization Daily · {date.strftime('%Y-%m-%d')}\n")
    body.append(f"> 📅 Window: {date.strftime('%Y-%m-%d')} (UTC daily)")
    body.append(
        f"> 📊 Scanned {scanned} papers → passed filter {len(surviving)}"
        f" → highlighted {highlighted_total} (threshold ≥{threshold})"
    )
    body.append("")
    body.append(f"> Auto-generated daily digest from [llm-paper-radar]({REPO_URL}).")
    body.append(
        "> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml)"
        " · Powered by Claude Sonnet 4.6\n"
    )

    body.append("## 🔥 Highlights by topic\n")
    rank = 0
    for bucket in BUCKET_ORDER:
        papers = grouped[bucket]
        if not papers:
            continue
        cap = _cap_for(bucket, topic_caps)
        body.append(f"### {BUCKET_TITLES[bucket]} (top {len(papers)} of cap {cap})\n")
        for p in papers:
            rank += 1
            body.append(_full_block(rank, p))

    body.append("## 📚 Full List (by score, descending)\n")
    body.append("| # | Title | Score | Topic | Pract | Bucket | Sources | Code | Date |")
    body.append("|---|-------|-------|-------|-------|--------|---------|------|------|")
    for i, p in enumerate(surviving, start=1):
        body.append(_table_row(i, p))
    body.append("")

    if revisited:
        body.append("\n## 🔁 Revisited\n")
        for p in revisited[:5]:
            body.append(f"- [{p.title}]({p.url}) — score {p.relevance_score}")

    text = "\n".join(body)
    digests_dir.mkdir(parents=True, exist_ok=True)
    digest_path = digests_dir / f"{date.strftime('%Y-%m-%d')}.md"
    digest_path.write_text(text)
    readme_path.write_text(text)

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
