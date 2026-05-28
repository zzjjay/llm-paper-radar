"""Keyword prefilter + Claude Sonnet two-axis scoring + milestone trending override; emits data/scored/<date>.json."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import yaml

from pipeline._clock import today_utc
from pipeline.config import PrefilterConfig, load_config
from pipeline.llm_client import LLMClient, load_prompt
from sources.base import Paper

# Milestone-override thresholds. A paper in seeds.yaml under category=trending
# bypasses any LLM hard_gate verdict when EITHER heat signal fires.
MILESTONE_TRENDING_RANK_MAX = 20
MILESTONE_STARS_MIN = 5000
# Don't re-surface the same milestone paper if it already surfaced within
# this many days. Without a cooldown a perennially hot paper (vLLM, FlashAttn)
# would appear in the daily digest every day it's on HF trending.
MILESTONE_COOLDOWN_DAYS = 14


@lru_cache(maxsize=1)
def _milestone_trending_ids() -> frozenset[str]:
    """Load arxiv IDs flagged for the trending milestone override from
    seeds.yaml. Cached so we don't re-parse on every paper."""
    seeds_path = Path("seeds.yaml")
    if not seeds_path.exists():
        return frozenset()
    try:
        cfg = yaml.safe_load(seeds_path.read_text()) or {}
    except yaml.YAMLError:
        return frozenset()
    out = set()
    for s in cfg.get("seeds", []) or []:
        if not isinstance(s, dict) or s.get("category") != "trending":
            continue
        sid = (s.get("id") or "").strip()
        if not sid:
            continue
        # seeds.yaml uses "arXiv:2309.06180"; Paper.id is the bare "2309.06180"
        # for arxiv-sourced papers — normalize both ends.
        out.add(sid.split(":")[-1])
    return frozenset(out)


def _recently_surfaced_milestones(
    scored_root: Path,
    target_date: date,
    cooldown_days: int,
) -> frozenset[str]:
    """IDs of milestone-trending papers that already surfaced (passed hard_gate
    in the trending bucket) within the prior `cooldown_days` days strictly
    before `target_date`. Used to suppress repeat overrides so a perennially
    hot paper doesn't appear day after day."""
    if cooldown_days <= 0:
        return frozenset()
    milestone_ids = _milestone_trending_ids()
    if not milestone_ids:
        return frozenset()
    seen: set[str] = set()
    for delta in range(1, cooldown_days + 1):
        prev = target_date - timedelta(days=delta)
        p = scored_root / f"{prev.strftime('%Y-%m-%d')}.json"
        if not p.exists():
            continue
        try:
            entries = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            aid = (e.get("id") or "").split(":")[-1]
            if aid not in milestone_ids:
                continue
            br = e.get("relevance_breakdown") or {}
            if br.get("topic_bucket") == "trending" and br.get("hard_gate") is False:
                seen.add(aid)
    return frozenset(seen)


def _milestone_override(
    paper: Paper, cooldown_ids: frozenset[str] = frozenset()
) -> dict | None:
    """If the paper is on the milestone-trending whitelist AND meets either
    heat threshold (HF trending rank ≤ 20, or code_meta.stars ≥ 5000),
    return a breakdown dict that bypasses the LLM verdict and routes the
    paper to the `trending` bucket. Otherwise return None.

    `cooldown_ids` are milestone IDs that surfaced recently and should be
    suppressed regardless of current heat — see `_recently_surfaced_milestones`."""
    arxiv_id = paper.id.split(":")[-1]
    if arxiv_id not in _milestone_trending_ids():
        return None
    if arxiv_id in cooldown_ids:
        return None
    has_hot_trending = any(
        s.name == "hf_daily"
        and isinstance(s.extras.get("trending_rank"), int)
        and s.extras["trending_rank"] <= MILESTONE_TRENDING_RANK_MAX
        for s in paper.sources
    )
    meta = paper.code_meta or {}
    stars = meta.get("stars")
    has_hot_stars = isinstance(stars, int) and stars >= MILESTONE_STARS_MIN
    if not (has_hot_trending or has_hot_stars):
        return None
    return {
        "hard_gate": False,
        "topic_relevance": 4,
        "practicality": 4,
        "compression_type": "trending",
        "topic_bucket": "trending",
        "model_domain": "language",
        "format_or_method": "milestone framework",
        "largest_model_tested": "production-deployed",
        "accuracy_benchmarks": "n/a (infra paper)",
        "accuracy_summary": "milestone framework, no compression accuracy metric",
        "inference_perf": "real-world deployed at scale",
        "calibration_cost": "n/a",
        "peak_memory": "n/a",
        "reason": (
            "milestone-override: seeds.yaml-curated trending milestone "
            "(vLLM/FlashAttention/EAGLE/Medusa/Lookahead/SGLang class) "
            "with HF trending rank ≤ 20 or code_meta.stars ≥ 5000"
        ),
    }

_BREAKDOWN_FIELDS = (
    "hard_gate",
    "topic_relevance",
    "practicality",
    "compression_type",
    "topic_bucket",
    "model_domain",
    "format_or_method",
    "largest_model_tested",
    "accuracy_benchmarks",
    "accuracy_summary",
    "inference_perf",
    "calibration_cost",
    "peak_memory",
)


def _clip(value, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def _composite(result: dict) -> int:
    if result.get("hard_gate"):
        return 0
    topic = _clip(result.get("topic_relevance"), 0, 5)
    pract = _clip(result.get("practicality"), 0, 5)
    return topic + pract


# ---------------------------------------------------------------------------
# Keyword prefilter
#
# Word-boundary aware so e.g. "QuIP" does not match inside "equipping" and
# "MIT" does not match inside "Amit". A pattern's left/right boundary is
# only enforced on the side that ends in an alphanumeric character; this
# lets patterns like "GPT-" or "1.58-bit" match next to other characters.


_PATTERN_CACHE: dict[str, re.Pattern] = {}


def _compiled(pattern: str) -> re.Pattern:
    cached = _PATTERN_CACHE.get(pattern)
    if cached is not None:
        return cached
    pl = pattern.lower()
    left = r"(?<![a-z0-9])" if pl and pl[0].isalnum() else ""
    right = r"(?![a-z0-9])" if pl and pl[-1].isalnum() else ""
    compiled = re.compile(left + re.escape(pl) + right)
    _PATTERN_CACHE[pattern] = compiled
    return compiled


def _matches(pattern: str, text: str) -> bool:
    return _compiled(pattern).search(text) is not None


def prefilter_verdict(
    paper: Paper, cfg: PrefilterConfig
) -> tuple[bool, list[str], list[str]]:
    """Return (should_hard_gate, whitelist_hits, blacklist_hits).

    A paper is locally hard-gated when:
      - it has zero whitelist hits, AND
      - it has >= cfg.max_blacklist_hits blacklist hits.

    Otherwise the LLM judge gets the final say."""
    text = f"{paper.title}\n{paper.abstract}".lower()
    wl = [r.pattern for r in cfg.whitelist if _matches(r.pattern, text)]
    bl = [r.pattern for r in cfg.blacklist if _matches(r.pattern, text)]
    should_gate = (not wl) and (len(bl) >= cfg.max_blacklist_hits)
    return should_gate, wl, bl


def _prefilter_hard_gate_result(blacklist_hits: list[str]) -> dict:
    """Build a `_BREAKDOWN_FIELDS`-shaped dict for papers killed by the
    keyword prefilter. Looks indistinguishable from a Sonnet hard_gate
    response so downstream stages don't special-case it."""
    reason = f"prefilter: 命中黑名单 {', '.join(blacklist_hits[:3])}"
    return _hard_gate_result(reason)


def _judge_unavailable_result(error: Exception) -> dict:
    """`_BREAKDOWN_FIELDS`-shaped dict for papers where the Sonnet judge
    failed every retry (empty response, persistent JSON parse error, API
    outage). Stored as hard_gate=True so the paper is preserved with a
    diagnosable reason rather than silently disappearing with score=None.
    Reprocessed on next cron run."""
    reason = f"judge unavailable: {type(error).__name__}: {error}"[:200]
    return _hard_gate_result(reason)


def _hard_gate_result(reason: str) -> dict:
    return {
        "hard_gate": True,
        "topic_relevance": 0,
        "practicality": 0,
        "compression_type": "unknown",
        "topic_bucket": "unknown",
        "model_domain": "unknown",
        "format_or_method": "unknown",
        "largest_model_tested": "unknown",
        "accuracy_benchmarks": "none",
        "accuracy_summary": "unknown",
        "inference_perf": "unknown",
        "calibration_cost": "unknown",
        "peak_memory": "unknown",
        "reason": reason,
    }


async def _score_one(
    paper: Paper,
    prompt: str,
    client: LLMClient,
    sem: asyncio.Semaphore,
    prefilter_cfg: PrefilterConfig,
    cooldown_ids: frozenset[str] = frozenset(),
) -> Paper:
    # Skip stub papers (e.g. hf_daily trending-only entries with no metadata).
    # Without a title or abstract there is nothing for the LLM to assess.
    if not paper.title.strip() or not paper.abstract.strip():
        paper.relevance_score = None
        paper.relevance_reason = "skipped: missing title/abstract"
        return paper

    # Cheap keyword prefilter — kill obvious off-topic before burning an LLM call.
    if prefilter_cfg.enabled:
        should_gate, _wl, bl = prefilter_verdict(paper, prefilter_cfg)
        if should_gate:
            result = _prefilter_hard_gate_result(bl)
            paper.relevance_score = _composite(result)
            paper.relevance_reason = result["reason"]
            paper.relevance_breakdown = {k: result.get(k) for k in _BREAKDOWN_FIELDS}
            return paper

    user_msg = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    async with sem:
        try:
            result = await client.call_json(prompt, user_msg, max_tokens=1024)
            paper.relevance_score = _composite(result)
            paper.relevance_reason = str(result.get("reason", ""))
            paper.relevance_breakdown = {k: result.get(k) for k in _BREAKDOWN_FIELDS}
        except Exception as e:
            # tenacity wraps the real failure in RetryError; surface the
            # underlying exception so log lines say what actually went wrong.
            cause = e
            inner = getattr(e, "last_attempt", None)
            if inner is not None and inner.failed:
                try:
                    cause = inner.exception() or e
                except Exception:
                    pass
            print(f"filter: paper {paper.id} failed: {type(cause).__name__}: {cause}")
            # Don't silently drop the paper with score=None — store a
            # hard_gate=True record with a diagnosable reason so the
            # failure is visible downstream and the paper can be retried.
            result = _judge_unavailable_result(cause)
            paper.relevance_score = _composite(result)
            paper.relevance_reason = result["reason"]
            paper.relevance_breakdown = {k: result.get(k) for k in _BREAKDOWN_FIELDS}

    # Post-LLM milestone override: seeds.yaml-curated framework papers
    # (vLLM, FlashAttention, EAGLE/Medusa/Lookahead, SGLang, …) with strong
    # HF heat or GitHub stars get routed to `trending` regardless of what
    # the LLM said. The LLM doesn't see heat signals so it can't make this
    # decision; the override is deterministic and easy to audit.
    override = _milestone_override(paper, cooldown_ids)
    if override is not None:
        paper.relevance_score = _composite(override)
        paper.relevance_reason = override["reason"]
        paper.relevance_breakdown = {k: override.get(k) for k in _BREAKDOWN_FIELDS}
    return paper


async def filter_papers(
    deduped_path: Path,
    out_path: Path,
    prompt_path: Path,
    client: LLMClient,
    concurrency: int = 50,
    prefilter_cfg: PrefilterConfig | None = None,
    cooldown_days: int = MILESTONE_COOLDOWN_DAYS,
) -> int:
    raw = await asyncio.to_thread(Path(deduped_path).read_text)
    papers = [Paper.model_validate(p) for p in json.loads(raw)]
    prompt = load_prompt(prompt_path)
    sem = asyncio.Semaphore(concurrency)
    if prefilter_cfg is None:
        prefilter_cfg = PrefilterConfig(enabled=False)
    # Derive target date from out filename so caller doesn't have to pass it.
    try:
        target_date = date.fromisoformat(Path(out_path).stem)
        cooldown_ids = _recently_surfaced_milestones(
            Path(out_path).parent, target_date, cooldown_days
        )
    except ValueError:
        cooldown_ids = frozenset()
    scored = await asyncio.gather(
        *[_score_one(p, prompt, client, sem, prefilter_cfg, cooldown_ids) for p in papers]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps([p.model_dump(mode="json") for p in scored], indent=2)
    await asyncio.to_thread(out_path.write_text, output)
    return len(scored)


if __name__ == "__main__":
    from datetime import UTC, datetime, timedelta

    import click

    @click.command()
    @click.option("--date", default=None)
    @click.option("--backfill-days", default=0, type=int, help="Process today + N days back. Default 0 = today only. Each day is fetched/processed independently.")
    @click.option("--in-root", default="data/deduped", type=click.Path(path_type=Path))
    @click.option("--out-root", default="data/scored", type=click.Path(path_type=Path))
    @click.option("--prompt-path", default="prompts/relevance.md", type=click.Path(path_type=Path))
    @click.option("--force", is_flag=True, default=False, help="Re-run even if the day's digest already exists.")
    def main(date, backfill_days, in_root, out_root, prompt_path, force):
        cfg = load_config()
        client = LLMClient(model=cfg.filter.model)
        if date:
            base = datetime.fromisoformat(date).replace(tzinfo=UTC)
        else:
            base = today_utc()
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            if not force and backfill_days > 0 and (
                Path("digests") / f"{target.strftime('%Y-%m-%d')}.md"
            ).exists():
                print(f"filter: skip {target.date()} (digest exists)")
                continue
            in_path = in_root / f"{target.strftime('%Y-%m-%d')}.json"
            out_path = out_root / f"{target.strftime('%Y-%m-%d')}.json"
            if not in_path.exists():
                print(f"filter: skip {target.date()} (no deduped input)")
                continue
            n = asyncio.run(
                filter_papers(
                    in_path,
                    out_path,
                    prompt_path,
                    client,
                    cfg.filter.concurrency,
                    cfg.filter.prefilter,
                )
            )
            print(f"filter: scored {n} papers for {target.date()}")

    main()
