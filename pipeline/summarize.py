from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pipeline.config import load_config
from pipeline.llm_client import LLMClient, load_prompt
from sources.base import Paper


async def _summarize_one(
    paper: Paper, prompt: str, client: LLMClient, sem: asyncio.Semaphore
) -> Paper:
    user = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    async with sem:
        try:
            r = await client.call_json(prompt, user, max_tokens=1100)
            paper.summary = r.get("summary")
            paper.highlights = r.get("highlights") or []
            related = r.get("related_methods") or []
            # Defensive: drop malformed entries and cap to 3.
            paper.related_methods = [
                {
                    "name": str(m.get("name", "")).strip(),
                    "relation": str(m.get("relation", "")).strip(),
                    "arxiv_id": (m.get("arxiv_id") or None),
                }
                for m in related
                if isinstance(m, dict) and m.get("name")
            ][:3]
        except Exception as e:
            print(f"summarize: paper {paper.id} failed: {e}")
    return paper


def _passed_gate(p: Paper) -> bool:
    if p.relevance_score is None:
        return False
    bd = p.relevance_breakdown or {}
    return not bd.get("hard_gate", False)


async def summarize_papers(
    in_path: Path,
    out_path: Path,
    prompt_path: Path,
    client: LLMClient,
    concurrency: int,
) -> int:
    raw = await asyncio.to_thread(Path(in_path).read_text)
    papers = [Paper.model_validate(p) for p in json.loads(raw)]
    prompt = load_prompt(prompt_path)
    sem = asyncio.Semaphore(concurrency)
    # Summarize everything that wasn't hard-gated. Watched-author papers
    # always qualify (they're not hard-gated by definition unless prefilter
    # caught them, and the watchlist exists precisely to never miss those).
    targets = [
        p
        for p in papers
        if _passed_gate(p) or any(s.name == "arxiv_authors" for s in p.sources)
    ]
    await asyncio.gather(*[_summarize_one(p, prompt, client, sem) for p in targets])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
    await asyncio.to_thread(out_path.write_text, output)
    return len(papers)


if __name__ == "__main__":
    from datetime import UTC, datetime, timedelta

    import click

    @click.command()
    @click.option("--date", default=None)
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--in-root", default="data/scored", type=click.Path(path_type=Path))
    @click.option("--out-root", default="data/summarized", type=click.Path(path_type=Path))
    @click.option("--prompt-path", default="prompts/summary.md", type=click.Path(path_type=Path))
    def main(date, backfill_days, in_root, out_root, prompt_path):
        cfg = load_config()
        client = LLMClient(model=cfg.summarize.model)
        if date:
            base = datetime.fromisoformat(date).replace(tzinfo=UTC)
        else:
            base = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            if backfill_days > 0 and (
                Path("digests") / f"{target.strftime('%Y-%m-%d')}.md"
            ).exists():
                print(f"summarize: skip {target.date()} (digest exists)")
                continue
            in_path = in_root / f"{target.strftime('%Y-%m-%d')}.json"
            out_path = out_root / f"{target.strftime('%Y-%m-%d')}.json"
            if not in_path.exists():
                print(f"summarize: skip {target.date()}")
                continue
            n = asyncio.run(
                summarize_papers(
                    in_path,
                    out_path,
                    prompt_path,
                    client,
                    cfg.summarize.concurrency,
                )
            )
            print(f"summarize: processed {n} papers for {target.date()}")

    main()
