from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pipeline.config import load_config
from pipeline.llm_client import LLMClient, load_prompt
from sources.base import Paper

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


async def _score_one(paper: Paper, prompt: str, client: LLMClient, sem: asyncio.Semaphore) -> Paper:
    # Skip stub papers (e.g. hf_daily trending-only entries with no metadata).
    # Without a title or abstract there is nothing for the LLM to assess.
    if not paper.title.strip() or not paper.abstract.strip():
        paper.relevance_score = None
        paper.relevance_reason = "skipped: missing title/abstract"
        return paper
    user_msg = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    async with sem:
        try:
            result = await client.call_json(prompt, user_msg, max_tokens=600)
            paper.relevance_score = _composite(result)
            paper.relevance_reason = str(result.get("reason", ""))[:160]
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
            paper.relevance_score = None
            paper.relevance_reason = None
            paper.relevance_breakdown = None
    return paper


async def filter_papers(
    deduped_path: Path,
    out_path: Path,
    prompt_path: Path,
    client: LLMClient,
    concurrency: int = 50,
) -> int:
    raw = await asyncio.to_thread(Path(deduped_path).read_text)
    papers = [Paper.model_validate(p) for p in json.loads(raw)]
    prompt = load_prompt(prompt_path)
    sem = asyncio.Semaphore(concurrency)
    scored = await asyncio.gather(*[_score_one(p, prompt, client, sem) for p in papers])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps([p.model_dump(mode="json") for p in scored], indent=2)
    await asyncio.to_thread(out_path.write_text, output)
    return len(scored)


if __name__ == "__main__":
    from datetime import UTC, datetime, timedelta

    import click

    @click.command()
    @click.option("--date", default=None)
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--in-root", default="data/deduped", type=click.Path(path_type=Path))
    @click.option("--out-root", default="data/scored", type=click.Path(path_type=Path))
    @click.option("--prompt-path", default="prompts/relevance.md", type=click.Path(path_type=Path))
    def main(date, backfill_days, in_root, out_root, prompt_path):
        cfg = load_config()
        client = LLMClient(model=cfg.filter.model)
        if date:
            base = datetime.fromisoformat(date).replace(tzinfo=UTC)
        else:
            base = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            in_path = in_root / f"{target.strftime('%Y-%m-%d')}.json"
            out_path = out_root / f"{target.strftime('%Y-%m-%d')}.json"
            if not in_path.exists():
                print(f"filter: skip {target.date()} (no deduped input)")
                continue
            n = asyncio.run(
                filter_papers(in_path, out_path, prompt_path, client, cfg.filter.concurrency)
            )
            print(f"filter: scored {n} papers for {target.date()}")

    main()
