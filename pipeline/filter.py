from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pipeline.config import load_config
from pipeline.llm_client import LLMClient, load_prompt
from sources.base import Paper


async def _score_one(paper: Paper, prompt: str, client: LLMClient, sem: asyncio.Semaphore) -> Paper:
    user_msg = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    async with sem:
        try:
            result = await client.call_json(prompt, user_msg, max_tokens=200)
            paper.relevance_score = int(result.get("relevance_score", 0))
            paper.relevance_reason = str(result.get("reason", ""))[:120]
        except Exception as e:
            print(f"filter: paper {paper.id} failed: {e}")
            paper.relevance_score = None
            paper.relevance_reason = None
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
