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


def sort_papers(papers: list[Paper]) -> list[Paper]:
    """Heat-primary; relevance breaks ties."""
    return sorted(
        papers,
        key=lambda p: (heat_score(p), p.relevance_score or 0),
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


def _full_block(rank: int, p: Paper) -> str:
    code_part = f" · [GitHub]({p.code_url})" if p.code_url else ""
    revisited = " 🔁" if p.seen_before else ""
    hl_zh = "\n".join(f"- {h}" for h in p.highlights_zh)
    hl_en = "\n".join(f"- {h}" for h in p.highlights_en)
    primary_source = p.sources[0].name if p.sources else "unknown"
    return f"""### {rank}. {p.title} ({p.relevance_score}/10){revisited}
**{primary_source}** · `{p.id}` · {p.published_at.date()}
👥 {", ".join(p.authors[:3])}{"..." if len(p.authors) > 3 else ""} · 🏷 {", ".join(p.categories)}
🔗 [arXiv]({p.url}) · [PDF]({p.pdf_url}){code_part}
📡 来源: {_source_badge(p)}

#### 中文摘要
{p.summary_zh or ''}

{hl_zh}

#### English Summary
{p.summary_en or ''}

{hl_en}

---
"""


def _table_row(rank: int, p: Paper) -> str:
    sources = "+".join({s.name for s in p.sources})
    code = "✅" if p.code_url else "—"
    date_str = p.published_at.strftime("%m-%d")
    return (
        f"| {rank} | [{p.title}]({p.url}) | {p.relevance_score} | {sources} | {code} | {date_str} |"
    )


def render_index_line(date: datetime, scanned: int, passed: int, top_title: str) -> str:
    day = date.strftime("%m-%d")
    full = date.strftime("%Y-%m-%d")
    return f"- [{day}](digests/{full}.md) — {scanned} 扫描, {passed} 通过, top: {top_title}"


def render_daily(
    date: datetime,
    summarized_path: Path,
    digests_dir: Path,
    readme_path: Path,
    index_path: Path,
    full_top_n: int,
    threshold: int,
) -> None:
    all_papers = [Paper.model_validate(p) for p in json.loads(summarized_path.read_text())]
    scanned = len(all_papers)
    surviving = [p for p in all_papers if (p.relevance_score or 0) >= threshold]
    surviving = sort_papers(surviving)
    revisited = [p for p in surviving if p.seen_before]

    body = []
    body.append(f"# LLM 推理优化日报 · {date.strftime('%Y-%m-%d')}\n")
    body.append(f"> 📅 抓取窗口: {date.strftime('%Y-%m-%d')} (UTC daily window)")
    body.append(f"> 📊 共扫描 {scanned} 篇 → 通过过滤 {len(surviving)} 篇 (阈值 ≥{threshold})")
    body.append("")
    body.append(f"> 这是 [llm-paper-radar]({REPO_URL}) 自动生成的最新一日 digest。")
    body.append(
        "> 历史: [INDEX.md](INDEX.md) · 配置: [config.yaml](config.yaml)"
        " · Powered by Claude Sonnet 4.6\n"
    )

    body.append(f"## 🔥 Top {min(full_top_n, len(surviving))} (Full Detail)\n")
    for i, p in enumerate(surviving[:full_top_n], start=1):
        body.append(_full_block(i, p))

    body.append("## 📚 完整列表 (按分数降序)\n")
    body.append("| # | Title | Score | Sources | Code | Date |")
    body.append("|---|-------|-------|---------|------|------|")
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
        index_path.write_text("# 历史 Digest 索引\n\n" + new_line + "\n")


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
                target, in_path, digests_dir, readme, index,
                cfg.render.full_top_n, cfg.filter.threshold,
            )
            print(f"render: wrote digest for {target.date()}")

    main()
