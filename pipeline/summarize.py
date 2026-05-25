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
    bd = paper.relevance_breakdown or {}
    cal_zh = str(bd.get("calibration_cost") or "")
    perf_zh = str(bd.get("inference_perf") or "")
    reason_zh = (paper.relevance_reason or "").strip()
    user = (
        f"Title: {paper.title}\n\n"
        f"Abstract: {paper.abstract}\n\n"
        "Chinese fields to translate (write English siblings, do not re-evaluate):\n"
        f"- relevance_reason: {reason_zh}\n"
        f"- calibration_cost: {cal_zh}\n"
        f"- inference_perf: {perf_zh}"
    )

    def _clean_related(related: list) -> list[dict]:
        return [
            {
                "name": str(m.get("name", "")).strip(),
                "relation": str(m.get("relation", "")).strip(),
                "arxiv_id": (m.get("arxiv_id") or None),
            }
            for m in related
            if isinstance(m, dict) and m.get("name")
        ][:3]

    async with sem:
        try:
            # 3000 token budget covers both summaries + translations of the
            # filter-step Chinese fields (reason / cal / perf). Bumped from
            # 2000 (bilingual summary only) and 1100 (CN-only original).
            r = await client.call_json(prompt, user, max_tokens=3000)
            paper.summary = r.get("summary")
            paper.highlights = r.get("highlights") or []
            paper.related_methods = _clean_related(r.get("related_methods") or [])
            paper.summary_en = r.get("summary_en")
            paper.highlights_en = r.get("highlights_en") or []
            paper.related_methods_en = _clean_related(r.get("related_methods_en") or [])
            # Translations of filter-step Chinese fields. Empty string = the
            # original was empty / missing; render falls back to the CN value
            # only when the _en string is missing (None), not when it's "".
            paper.relevance_reason_en = r.get("relevance_reason_en")
            cal_en = r.get("calibration_cost_en")
            perf_en = r.get("inference_perf_en")
            if cal_en is not None or perf_en is not None:
                # Mutate breakdown in place so render's _signal_line can read
                # `<key>_en` siblings; avoids adding a parallel dict field.
                if paper.relevance_breakdown is None:
                    paper.relevance_breakdown = {}
                if cal_en is not None:
                    paper.relevance_breakdown["calibration_cost_en"] = cal_en
                if perf_en is not None:
                    paper.relevance_breakdown["inference_perf_en"] = perf_en
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
    @click.option("--backfill-days", default=0, type=int, help="Process today + N days back. Default 0 = today only. Each day is fetched/processed independently.")
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
