from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import yaml

from sources.base import Paper, Source, SourceRecord


class SemanticScholarSource(Source):
    name = "semantic_scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    FIELDS = "externalIds,title,abstract,authors,publicationDate,url,openAccessPdf"

    def __init__(self, seeds_file: Path, citation_window_days: int = 7):
        self.seeds_file = Path(seeds_file)
        self.window_days = citation_window_days

    def _seeds(self) -> list[str]:
        data = yaml.safe_load(self.seeds_file.read_text())
        return [s["id"] for s in data.get("seeds", []) if not s["id"].endswith("????")]

    async def fetch(self, target_date: datetime) -> list[Paper]:
        cutoff = target_date - timedelta(days=self.window_days)
        headers = {}
        if key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
            headers["x-api-key"] = key

        all_papers: list[Paper] = []
        seen_arxiv: set[str] = set()
        async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
            for seed in self._seeds():
                try:
                    r = await client.get(
                        f"{self.BASE_URL}/{seed}/citations",
                        params={"fields": f"citingPaper.{self.FIELDS}", "limit": 100},
                    )
                    r.raise_for_status()
                    items = r.json().get("data", [])
                except httpx.HTTPError:
                    continue

                for item in items:
                    cp = item.get("citingPaper", {})
                    arxiv_id = (cp.get("externalIds") or {}).get("ArXiv")
                    if not arxiv_id or arxiv_id in seen_arxiv:
                        continue
                    pub_str = cp.get("publicationDate")
                    if not pub_str:
                        continue
                    try:
                        pub = datetime.fromisoformat(pub_str).replace(tzinfo=UTC)
                    except ValueError:
                        continue
                    if pub < cutoff:
                        continue
                    seen_arxiv.add(arxiv_id)
                    pdf_obj = cp.get("openAccessPdf") or {}
                    all_papers.append(
                        Paper(
                            id=arxiv_id,
                            title=(cp.get("title") or "").strip(),
                            authors=[a.get("name", "") for a in cp.get("authors") or []],
                            abstract=(cp.get("abstract") or "").strip(),
                            url=cp.get("url") or f"https://arxiv.org/abs/{arxiv_id}",
                            pdf_url=pdf_obj.get("url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                            published_at=pub,
                            primary_category="unknown",
                            categories=[],
                            sources=[
                                SourceRecord(
                                    name="semantic_scholar",
                                    fetched_at=datetime.now(UTC),
                                    extras={"seed": seed},
                                )
                            ],
                        )
                    )
                await asyncio.sleep(1.0 if not headers else 0.1)

        return all_papers


if __name__ == "__main__":
    import json
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option(
        "--window-days",
        default=None,
        type=int,
        help="Override citation_window_days from config for this run.",
    )
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, window_days: int | None, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.semantic_scholar.enabled:
            print("semantic_scholar disabled")
            return
        effective_window = (
            window_days if window_days is not None else cfg.sources.semantic_scholar.citation_window_days
        )
        src = SemanticScholarSource(
            seeds_file=Path(cfg.sources.semantic_scholar.seeds_file),
            citation_window_days=effective_window,
        )
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            try:
                papers = asyncio.run(src.fetch(target))
            except Exception as e:
                print(f"semantic_scholar: skip {target.date()}: {e}")
                continue
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "semantic_scholar.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"semantic_scholar: wrote {len(papers)} for {target.date()}")

    main()
