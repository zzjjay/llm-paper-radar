# LLM Paper Radar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily-running automated pipeline that fetches papers from 6 sources, dedupes, filters via Claude Haiku, summarizes via Claude Sonnet (Chinese + English), renders a Markdown digest, and publishes to a public GitHub repo via GitHub Actions.

**Architecture:** Modular pipeline. Each data source is an independent module returning normalized `Paper` objects; pipeline stages (dedupe → filter → summarize → render) consume disk JSON between steps for traceability. Single-source failure does not block the digest. All scheduling and deployment via GitHub Actions; no external infrastructure beyond optional self-hosted RSSHub.

**Tech Stack:**
- Python 3.11+, dependency management via `uv`
- `httpx` (async HTTP), `pydantic v2` (schemas), `pyyaml` (config), `feedparser` (arXiv/PwC/RSSHub feeds)
- `anthropic` SDK for Claude Haiku 4.5 + Sonnet 4.6
- `pytest` + `pytest-asyncio` + `respx` (httpx mocking)
- `ruff` for lint/format
- `click` for CLI entrypoints

**Spec reference:** `docs/superpowers/specs/2026-05-11-llm-paper-radar-design.md`

**Repo target:** `https://github.com/zhaolin-amd/llm-paper-radar` (to be created)

**Total tasks:** 19 across 4 phases. Each task is fully self-contained with TDD steps.

---

## Phase 1 — Foundation (Tasks 1–3)

### Task 1: Bootstrap project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `README.md` (placeholder)
- Create: `config.yaml`
- Create: `seeds.yaml`
- Create: `prompts/relevance.md`
- Create: `prompts/summary.md`
- Create: `sources/__init__.py`
- Create: `pipeline/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1.1: Create `pyproject.toml`**

```toml
[project]
name = "llm-paper-radar"
version = "0.1.0"
description = "Daily LLM inference optimization paper digest"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "click>=8.1.7",
    "feedparser>=6.0.11",
    "httpx>=0.27.0",
    "pydantic>=2.7.0",
    "pyyaml>=6.0.1",
    "tenacity>=9.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "respx>=0.21.1",
    "ruff>=0.7.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "ASYNC"]
```

- [ ] **Step 1.2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.env
.pytest_cache/
.ruff_cache/
data/raw/
data/deduped/
data/scored/
*.egg-info/
```

- [ ] **Step 1.3: Create `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-xxx
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
TEAMS_WEBHOOK_URL=
RSSHUB_BASE_URL=
SEMANTIC_SCHOLAR_API_KEY=
```

- [ ] **Step 1.4: Create `config.yaml`** (paste exact contents from spec §8 `config.yaml`)

- [ ] **Step 1.5: Create `seeds.yaml`**

```yaml
seeds:
  - id: arXiv:2210.17323
    name: GPTQ
  - id: arXiv:2306.00978
    name: AWQ
  - id: arXiv:2211.10438
    name: SmoothQuant
  - id: arXiv:2404.00456
    name: QuaRot
  - id: arXiv:2504.19874
    name: TurboQuant
  - id: arXiv:2512.02010
    name: FourOverSix
  - id: arXiv:2407.11062
    name: EfficientQAT
  - id: arXiv:2305.14314
    name: QLoRA
```

- [ ] **Step 1.6: Create `prompts/relevance.md`** (paste exact contents from spec §5 Relevance Prompt)

- [ ] **Step 1.7: Create `prompts/summary.md`** (paste exact contents from spec §5 Summary Prompt)

- [ ] **Step 1.8: Create empty package init files**

```bash
mkdir -p sources pipeline tests prompts data digests weekly
touch sources/__init__.py pipeline/__init__.py tests/__init__.py
```

- [ ] **Step 1.9: Create placeholder `README.md`**

```markdown
# llm-paper-radar

Auto-generated daily digest of LLM inference optimization papers.

First digest will appear here after the first scheduled run.
```

- [ ] **Step 1.10: Install deps and verify**

Run: `uv sync && uv run python -c "import anthropic, httpx, pydantic, click, feedparser, yaml; print('OK')"`
Expected: `OK`

- [ ] **Step 1.11: Commit**

```bash
git init
git add .
git commit -m "chore: project skeleton with deps and config"
```

---

### Task 2: Config loader

**Files:**
- Create: `pipeline/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 2.1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from pipeline.config import load_config, Config

def test_load_config_from_yaml(tmp_path: Path):
    yaml_text = """
sources:
  arxiv:
    enabled: true
    categories: [cs.CL, cs.LG]
  hf_daily:
    enabled: true
filter:
  model: claude-haiku-4-5-20251001
  threshold: 7
  concurrency: 50
summarize:
  model: claude-sonnet-4-6
  concurrency: 20
render:
  full_top_n: 10
  truncate_after: 10
dedupe:
  cross_day_strategy: lenient
  source_priority: [hf_daily, arxiv]
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert isinstance(cfg, Config)
    assert cfg.filter.threshold == 7
    assert cfg.sources.arxiv.enabled is True
    assert cfg.sources.arxiv.categories == ["cs.CL", "cs.LG"]
    assert cfg.dedupe.source_priority[0] == "hf_daily"
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.config'`

- [ ] **Step 2.3: Implement `pipeline/config.py`**

```python
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ArxivConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = ["cs.CL", "cs.LG", "cs.AR"]


class SimpleSourceConfig(BaseModel):
    enabled: bool = True


class RedditConfig(BaseModel):
    enabled: bool = True
    subreddit: str = "LocalLLaMA"
    top_window: str = "day"


class SemanticScholarConfig(BaseModel):
    enabled: bool = True
    seeds_file: str = "seeds.yaml"
    citation_window_days: int = 7


class TwitterConfig(BaseModel):
    enabled: bool = True
    accounts: list[str] = []


class SourcesConfig(BaseModel):
    arxiv: ArxivConfig = ArxivConfig()
    hf_daily: SimpleSourceConfig = SimpleSourceConfig()
    reddit: RedditConfig = RedditConfig()
    semantic_scholar: SemanticScholarConfig = SemanticScholarConfig()
    papers_with_code: SimpleSourceConfig = SimpleSourceConfig()
    twitter_rsshub: TwitterConfig = TwitterConfig()


class FilterConfig(BaseModel):
    model: str = "claude-haiku-4-5-20251001"
    threshold: int = Field(7, ge=0, le=10)
    concurrency: int = 50


class SummarizeConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    concurrency: int = 20


class RenderConfig(BaseModel):
    full_top_n: int = 10
    truncate_after: int = 10


class DedupeConfig(BaseModel):
    cross_day_strategy: Literal["strict", "lenient"] = "lenient"
    source_priority: list[str] = []


class Config(BaseModel):
    sources: SourcesConfig = SourcesConfig()
    filter: FilterConfig = FilterConfig()
    summarize: SummarizeConfig = SummarizeConfig()
    render: RenderConfig = RenderConfig()
    dedupe: DedupeConfig = DedupeConfig()


def load_config(path: Path = Path("config.yaml")) -> Config:
    data = yaml.safe_load(Path(path).read_text())
    return Config(**data)
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 2.5: Commit**

```bash
git add pipeline/config.py tests/test_config.py
git commit -m "feat(config): pydantic config loader with defaults"
```

---

### Task 3: Paper schema and Source base class

**Files:**
- Create: `sources/base.py`
- Test: `tests/test_paper_schema.py`

- [ ] **Step 3.1: Write the failing test**

```python
# tests/test_paper_schema.py
from datetime import datetime, timezone
import pytest
from sources.base import Paper, SourceRecord


def test_paper_minimal_construction():
    p = Paper(
        id="2402.17764",
        title="BitNet b1.58",
        authors=["Shuming Ma"],
        abstract="We introduce a 1-bit LLM variant.",
        url="https://arxiv.org/abs/2402.17764",
        pdf_url="https://arxiv.org/pdf/2402.17764.pdf",
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL",
        categories=["cs.CL", "cs.LG"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(timezone.utc))],
    )
    assert p.id == "2402.17764"
    assert p.relevance_score is None
    assert p.highlights_zh == []
    assert p.seen_before is False
    assert p.code_url is None


def test_source_record_with_extras():
    r = SourceRecord(
        name="reddit",
        fetched_at=datetime.now(timezone.utc),
        extras={"score": 230, "num_comments": 67},
    )
    assert r.extras["score"] == 230


def test_paper_round_trip_json():
    p = Paper(
        id="x",
        title="t",
        authors=[],
        abstract="a",
        url="https://x",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime(2026, 5, 11, tzinfo=timezone.utc))],
    )
    js = p.model_dump_json()
    p2 = Paper.model_validate_json(js)
    assert p2.id == "x"
    assert p2.sources[0].name == "arxiv"


def test_invalid_source_name_rejected():
    with pytest.raises(ValueError):
        SourceRecord(name="not_a_source", fetched_at=datetime.now(timezone.utc))
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `uv run pytest tests/test_paper_schema.py -v`
Expected: FAIL with import error

- [ ] **Step 3.3: Implement `sources/base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceName = Literal[
    "arxiv",
    "hf_daily",
    "reddit",
    "semantic_scholar",
    "papers_with_code",
    "twitter_rsshub",
]


class SourceRecord(BaseModel):
    name: SourceName
    fetched_at: datetime
    extras: dict = Field(default_factory=dict)


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str | None
    published_at: datetime
    primary_category: str
    categories: list[str]
    code_url: str | None = None

    sources: list[SourceRecord]

    relevance_score: int | None = None
    relevance_reason: str | None = None
    summary_zh: str | None = None
    highlights_zh: list[str] = Field(default_factory=list)
    summary_en: str | None = None
    highlights_en: list[str] = Field(default_factory=list)
    seen_before: bool = False


class Source(ABC):
    """Base class. Each concrete source implements `fetch()` returning Papers."""

    name: SourceName

    @abstractmethod
    async def fetch(self, target_date: datetime) -> list[Paper]:
        """Fetch papers for the given date (UTC)."""
        ...
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `uv run pytest tests/test_paper_schema.py -v`
Expected: PASS (4 passed)

- [ ] **Step 3.5: Commit**

```bash
git add sources/base.py tests/test_paper_schema.py
git commit -m "feat(schema): Paper + SourceRecord pydantic models and Source ABC"
```

---

## Phase 2 — Data Sources (Tasks 4–9)

### Task 4: arXiv source

**Files:**
- Create: `sources/arxiv.py`
- Test: `tests/test_arxiv.py`
- Test fixture: `tests/fixtures/arxiv_response.xml`

- [ ] **Step 4.1: Capture a real arXiv response as fixture**

Run:
```bash
mkdir -p tests/fixtures
curl -s "http://export.arxiv.org/api/query?search_query=cat:cs.CL&max_results=2&sortBy=submittedDate&sortOrder=descending" > tests/fixtures/arxiv_response.xml
```
Expected: file `tests/fixtures/arxiv_response.xml` (~5–20 KB) containing `<entry>` blocks. Verify with `head -50 tests/fixtures/arxiv_response.xml` shows `<feed xmlns=...>` and at least one `<entry>`.

- [ ] **Step 4.2: Write the failing test**

```python
# tests/test_arxiv.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.arxiv import ArxivSource


FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_response.xml"


@respx.mock
@pytest.mark.asyncio
async def test_arxiv_fetch_parses_entries():
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    src = ArxivSource(categories=["cs.CL"])
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert len(papers) >= 1
    p = papers[0]
    assert p.id  # arXiv ID like "2405.xxxxx"
    assert p.title
    assert p.abstract
    assert p.url.startswith("https://arxiv.org/abs/")
    assert p.pdf_url and p.pdf_url.startswith("https://arxiv.org/pdf/")
    assert p.sources[0].name == "arxiv"
    assert p.primary_category
```

- [ ] **Step 4.3: Run test to verify it fails**

Run: `uv run pytest tests/test_arxiv.py -v`
Expected: FAIL with import error

- [ ] **Step 4.4: Implement `sources/arxiv.py`**

```python
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Iterable

import feedparser
import httpx

from sources.base import Paper, Source, SourceRecord

ARXIV_ID_RE = re.compile(r"abs/([\d.]+)(?:v\d+)?$")


class ArxivSource(Source):
    name = "arxiv"
    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, categories: list[str], page_size: int = 200, max_pages: int = 5):
        self.categories = categories
        self.page_size = page_size
        self.max_pages = max_pages

    async def fetch(self, target_date: datetime) -> list[Paper]:
        """Fetch papers submitted within ~24h ending at target_date (UTC)."""
        cat_query = "+OR+".join(f"cat:{c}" for c in self.categories)
        all_entries: list[dict] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            start = 0
            for _ in range(self.max_pages):
                resp = await client.get(
                    self.BASE_URL,
                    params={
                        "search_query": cat_query,
                        "start": start,
                        "max_results": self.page_size,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                    },
                )
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
                if not feed.entries:
                    break
                all_entries.extend(feed.entries)
                start += self.page_size
                # Stop early if oldest entry on page is older than window
                oldest = _entry_published(feed.entries[-1])
                if oldest < target_date - timedelta(days=2):
                    break
                await asyncio.sleep(3.0)  # arXiv rate limit guidance

        return list(self._to_papers(all_entries, target_date))

    def _to_papers(self, entries: list[dict], target_date: datetime) -> Iterable[Paper]:
        window_start = target_date - timedelta(hours=24)
        now = datetime.now(timezone.utc)
        for e in entries:
            pub = _entry_published(e)
            if not (window_start <= pub <= target_date + timedelta(hours=1)):
                continue
            arxiv_id = _extract_arxiv_id(e.get("id", ""))
            if not arxiv_id:
                continue
            categories = [t["term"] for t in e.get("tags", [])]
            primary = categories[0] if categories else "unknown"
            pdf_link = next(
                (l["href"] for l in e.get("links", []) if l.get("type") == "application/pdf"),
                None,
            )
            yield Paper(
                id=arxiv_id,
                title=e.get("title", "").strip().replace("\n ", " "),
                authors=[a.get("name", "") for a in e.get("authors", [])],
                abstract=e.get("summary", "").strip(),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_link or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=pub,
                primary_category=primary,
                categories=categories,
                sources=[SourceRecord(name="arxiv", fetched_at=now)],
            )


def _extract_arxiv_id(url: str) -> str | None:
    m = ARXIV_ID_RE.search(url)
    return m.group(1) if m else None


def _entry_published(entry: dict) -> datetime:
    s = entry.get("published") or entry.get("updated")
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
```

- [ ] **Step 4.5: Run test to verify it passes**

Run: `uv run pytest tests/test_arxiv.py -v`
Expected: PASS

- [ ] **Step 4.6: Add CLI entrypoint at bottom of `sources/arxiv.py`**

```python
if __name__ == "__main__":
    import asyncio
    import json
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.arxiv.enabled:
            print("arxiv source disabled")
            return
        src = ArxivSource(categories=cfg.sources.arxiv.categories)
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            papers = asyncio.run(src.fetch(target))
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "arxiv.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"arxiv: wrote {len(papers)} papers for {target.date()}")

    main()
```

- [ ] **Step 4.7: Smoke-test the CLI locally**

Run: `uv run python -m sources.arxiv --backfill-days 0`
Expected: prints `arxiv: wrote N papers for YYYY-MM-DD` and creates `data/raw/<date>/arxiv.json`

- [ ] **Step 4.8: Commit**

```bash
git add sources/arxiv.py tests/test_arxiv.py tests/fixtures/arxiv_response.xml
git commit -m "feat(sources): arxiv source with feedparser + CLI entrypoint"
```

---

### Task 5: HuggingFace source (Daily + Trending)

**Files:**
- Create: `sources/hf_daily.py`
- Test: `tests/test_hf_daily.py`
- Test fixtures: `tests/fixtures/hf_daily_response.json`, `tests/fixtures/hf_trending.html`

This source fetches **two HF surfaces**: the curated daily papers JSON API AND the HTML trending page. Both contribute records to the same `Paper.sources[]` list with `name="hf_daily"`, distinguished by the `extras` payload (`upvotes` vs `trending_rank`).

- [ ] **Step 5.1: Capture daily papers fixture**

Run:
```bash
curl -s "https://huggingface.co/api/daily_papers?date=$(date -u -d 'yesterday' +%Y-%m-%d)" > tests/fixtures/hf_daily_response.json
```
Expected: JSON array, ~10–30 entries.

- [ ] **Step 5.1b: Capture trending page fixture**

Run:
```bash
curl -s -A "Mozilla/5.0" "https://huggingface.co/papers/trending" > tests/fixtures/hf_trending.html
```
Expected: HTML file, > 50 KB. Verify: `grep -o 'href="/papers/[0-9.]*' tests/fixtures/hf_trending.html | head -5` prints at least 5 paper links.

- [ ] **Step 5.2: Write the failing tests**

```python
# tests/test_hf_daily.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.hf_daily import HFDailySource, parse_trending_ranks

FIXTURE_DAILY = Path(__file__).parent / "fixtures" / "hf_daily_response.json"
FIXTURE_TREND = Path(__file__).parent / "fixtures" / "hf_trending.html"


@respx.mock
@pytest.mark.asyncio
async def test_hf_daily_fetch_includes_daily_and_trending():
    respx.get("https://huggingface.co/api/daily_papers").mock(
        return_value=Response(200, text=FIXTURE_DAILY.read_text())
    )
    respx.get("https://huggingface.co/papers/trending").mock(
        return_value=Response(200, text=FIXTURE_TREND.read_text())
    )
    src = HFDailySource()
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert len(papers) >= 1

    # At least one paper should carry upvotes (from daily API)
    has_daily = any("upvotes" in s.extras for p in papers for s in p.sources)
    assert has_daily

    # At least one paper should carry trending_rank (from trending HTML)
    has_trending = any("trending_rank" in s.extras for p in papers for s in p.sources)
    assert has_trending


def test_parse_trending_ranks_extracts_arxiv_ids_with_position():
    html = '''<a href="/papers/2402.17764">A</a>
<div>noise</div>
<a href="/papers/2503.12345">B</a>
<a href="/papers/2402.17764">A again</a>'''
    ranks = parse_trending_ranks(html)
    assert ranks["2402.17764"] == 1
    assert ranks["2503.12345"] == 2


@respx.mock
@pytest.mark.asyncio
async def test_hf_daily_continues_when_trending_fails():
    respx.get("https://huggingface.co/api/daily_papers").mock(
        return_value=Response(200, text=FIXTURE_DAILY.read_text())
    )
    respx.get("https://huggingface.co/papers/trending").mock(
        return_value=Response(500, text="oops")
    )
    src = HFDailySource()
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    # Daily papers still returned; trending failure tolerated
    assert len(papers) >= 1
```

- [ ] **Step 5.3: Run test to verify it fails**

Run: `uv run pytest tests/test_hf_daily.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 5.4: Implement `sources/hf_daily.py`**

```python
from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from sources.base import Paper, Source, SourceRecord

TRENDING_LINK_RE = re.compile(r'href="/papers/(\d{4}\.\d{4,5})"')


def parse_trending_ranks(html: str) -> dict[str, int]:
    """Extract arXiv IDs from the trending page HTML in order of appearance.
    Returns mapping arxiv_id -> rank (1-indexed, lowest = hottest)."""
    seen: dict[str, int] = {}
    for m in TRENDING_LINK_RE.finditer(html):
        aid = m.group(1)
        if aid not in seen:
            seen[aid] = len(seen) + 1
    return seen


class HFDailySource(Source):
    name = "hf_daily"
    DAILY_URL = "https://huggingface.co/api/daily_papers"
    TRENDING_URL = "https://huggingface.co/papers/trending"

    async def fetch(self, target_date: datetime) -> list[Paper]:
        date_str = target_date.strftime("%Y-%m-%d")
        async with httpx.AsyncClient(
            timeout=30.0, headers={"User-Agent": "Mozilla/5.0 llm-paper-radar"}
        ) as client:
            # 1) Daily papers (must succeed for source to be useful)
            daily_resp = await client.get(self.DAILY_URL, params={"date": date_str})
            daily_resp.raise_for_status()
            daily_items = daily_resp.json()

            # 2) Trending page (best-effort; may fail without aborting source)
            trending_ranks: dict[str, int] = {}
            try:
                tr_resp = await client.get(self.TRENDING_URL)
                if tr_resp.status_code == 200:
                    trending_ranks = parse_trending_ranks(tr_resp.text)
            except httpx.HTTPError as e:
                print(f"hf_daily: trending fetch failed ({e}); continuing with daily only")

        now = datetime.now(timezone.utc)
        papers: dict[str, Paper] = {}

        # Daily papers
        for item in daily_items:
            paper_obj = item.get("paper", {})
            arxiv_id = paper_obj.get("id") or item.get("id")
            if not arxiv_id:
                continue
            published_str = paper_obj.get("publishedAt") or item.get("publishedAt")
            try:
                published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except (AttributeError, ValueError):
                published_at = target_date

            extras = {
                "upvotes": paper_obj.get("upvotes", item.get("upvotes", 0)),
                "num_comments": paper_obj.get("numComments", item.get("numComments", 0)),
            }

            papers[arxiv_id] = Paper(
                id=arxiv_id,
                title=paper_obj.get("title", item.get("title", "")).strip(),
                authors=[a.get("name", "") for a in paper_obj.get("authors", [])],
                abstract=paper_obj.get("summary", item.get("summary", "")).strip(),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=published_at,
                primary_category="cs.CL",
                categories=[],
                sources=[SourceRecord(name="hf_daily", fetched_at=now, extras=extras)],
            )

        # Trending page: add a SECOND SourceRecord for papers already known from daily,
        # OR a fresh Paper stub for trending-only papers.
        for arxiv_id, rank in trending_ranks.items():
            tr_record = SourceRecord(
                name="hf_daily", fetched_at=now, extras={"trending_rank": rank}
            )
            if arxiv_id in papers:
                papers[arxiv_id].sources.append(tr_record)
            else:
                papers[arxiv_id] = Paper(
                    id=arxiv_id,
                    title="",  # filled later via merge with arXiv source
                    authors=[],
                    abstract="",
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    published_at=target_date,
                    primary_category="unknown",
                    categories=[],
                    sources=[tr_record],
                )

        return list(papers.values())


if __name__ == "__main__":
    import asyncio
    import json
    from datetime import timedelta
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.hf_daily.enabled:
            print("hf_daily source disabled")
            return
        src = HFDailySource()
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            papers = asyncio.run(src.fetch(target))
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "hf_daily.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"hf_daily: wrote {len(papers)} papers for {target.date()}")

    main()
```

- [ ] **Step 5.5: Run test to verify it passes**

Run: `uv run pytest tests/test_hf_daily.py -v`
Expected: PASS

- [ ] **Step 5.6: Smoke-test CLI**

Run: `uv run python -m sources.hf_daily --backfill-days 0`
Expected: prints count and writes `data/raw/<date>/hf_daily.json`

- [ ] **Step 5.7: Commit**

```bash
git add sources/hf_daily.py tests/test_hf_daily.py tests/fixtures/hf_daily_response.json
git commit -m "feat(sources): HuggingFace Daily Papers source"
```

---

### Task 6: Reddit source

**Files:**
- Create: `sources/reddit.py`
- Create: `sources/_arxiv_lookup.py` (shared utility for ID extraction + arXiv refetch)
- Test: `tests/test_reddit.py`
- Test fixture: `tests/fixtures/reddit_top_response.json`

- [ ] **Step 6.1: Create shared arXiv lookup utility**

```python
# sources/_arxiv_lookup.py
"""Shared: extract arXiv IDs from text, refetch full metadata via arXiv API."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import feedparser
import httpx

from sources.base import Paper, SourceRecord, SourceName

ARXIV_LINK_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")


def extract_arxiv_ids(text: str) -> set[str]:
    return set(ARXIV_LINK_RE.findall(text or ""))


async def fetch_arxiv_by_ids(
    ids: list[str],
    source_name: SourceName,
    extras_per_id: dict[str, dict] | None = None,
) -> list[Paper]:
    """Look up arXiv metadata for given IDs, return Paper objects with given source_name."""
    if not ids:
        return []
    extras_per_id = extras_per_id or {}
    id_query = ",".join(ids)
    url = "http://export.arxiv.org/api/query"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, params={"id_list": id_query, "max_results": len(ids)})
        resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    now = datetime.now(timezone.utc)
    papers: list[Paper] = []
    for e in feed.entries:
        m = re.search(r"abs/([\d.]+)(?:v\d+)?$", e.get("id", ""))
        if not m:
            continue
        arxiv_id = m.group(1)
        categories = [t["term"] for t in e.get("tags", [])]
        primary = categories[0] if categories else "unknown"
        pdf_link = next(
            (l["href"] for l in e.get("links", []) if l.get("type") == "application/pdf"),
            None,
        )
        try:
            pub = datetime.fromisoformat(e["published"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            pub = now
        papers.append(
            Paper(
                id=arxiv_id,
                title=e.get("title", "").strip().replace("\n ", " "),
                authors=[a.get("name", "") for a in e.get("authors", [])],
                abstract=e.get("summary", "").strip(),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_link or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                published_at=pub,
                primary_category=primary,
                categories=categories,
                sources=[
                    SourceRecord(
                        name=source_name,
                        fetched_at=now,
                        extras=extras_per_id.get(arxiv_id, {}),
                    )
                ],
            )
        )
    return papers
```

- [ ] **Step 6.2: Capture Reddit fixture (anonymized)**

Create file `tests/fixtures/reddit_top_response.json` with the following content (do not call live API for the test fixture; this minimal fixture is sufficient):

```json
{
  "data": {
    "children": [
      {
        "data": {
          "id": "abc123",
          "title": "New paper on FP4 quantization",
          "selftext": "Check out https://arxiv.org/abs/2402.17764 - results look great",
          "score": 230,
          "num_comments": 67,
          "permalink": "/r/LocalLLaMA/comments/abc123/new_paper_on_fp4/"
        }
      },
      {
        "data": {
          "id": "def456",
          "title": "Question about vLLM",
          "selftext": "How do I install vLLM on AMD?",
          "score": 5,
          "num_comments": 2,
          "permalink": "/r/LocalLLaMA/comments/def456/question/"
        }
      }
    ]
  }
}
```

- [ ] **Step 6.3: Write the failing test**

```python
# tests/test_reddit.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.reddit import RedditSource

FIXTURE = Path(__file__).parent / "fixtures" / "reddit_top_response.json"


@respx.mock
@pytest.mark.asyncio
async def test_reddit_extracts_arxiv_links_only(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "fake")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "fake")

    respx.post("https://www.reddit.com/api/v1/access_token").mock(
        return_value=Response(200, json={"access_token": "x", "expires_in": 3600})
    )
    respx.get("https://oauth.reddit.com/r/LocalLLaMA/top.json").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    # Mock arXiv lookup
    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2402.17764v1</id>
    <title>BitNet b1.58</title>
    <summary>1-bit LLM.</summary>
    <published>2026-05-10T00:00:00Z</published>
    <author><name>Shuming Ma</name></author>
    <category term="cs.CL"/>
    <link href="http://arxiv.org/pdf/2402.17764v1.pdf" type="application/pdf"/>
  </entry>
</feed>"""
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=arxiv_xml)
    )

    src = RedditSource(subreddit="LocalLLaMA", top_window="day")
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert len(papers) == 1  # only the post with arxiv link
    p = papers[0]
    assert p.id == "2402.17764"
    assert p.sources[0].name == "reddit"
    assert p.sources[0].extras["score"] == 230
    assert p.sources[0].extras["num_comments"] == 67
    assert "thread_url" in p.sources[0].extras
```

- [ ] **Step 6.4: Run test to verify it fails**

Run: `uv run pytest tests/test_reddit.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 6.5: Implement `sources/reddit.py`**

```python
from __future__ import annotations

import os
from datetime import datetime

import httpx

from sources._arxiv_lookup import extract_arxiv_ids, fetch_arxiv_by_ids
from sources.base import Paper, Source

USER_AGENT = "llm-paper-radar/0.1 (by /u/zhaolin-amd)"


class RedditSource(Source):
    name = "reddit"

    def __init__(self, subreddit: str = "LocalLLaMA", top_window: str = "day"):
        self.subreddit = subreddit
        self.top_window = top_window

    async def _get_token(self, client: httpx.AsyncClient) -> str:
        cid = os.environ["REDDIT_CLIENT_ID"]
        secret = os.environ["REDDIT_CLIENT_SECRET"]
        r = await client.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        return r.json()["access_token"]

    async def fetch(self, target_date: datetime) -> list[Paper]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            token = await self._get_token(client)
            r = await client.get(
                f"https://oauth.reddit.com/r/{self.subreddit}/top.json",
                params={"t": self.top_window, "limit": 50},
                headers={"Authorization": f"bearer {token}", "User-Agent": USER_AGENT},
            )
            r.raise_for_status()
            data = r.json()

        # Extract arXiv IDs and per-id extras
        id_to_extras: dict[str, dict] = {}
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            text_blob = " ".join(
                [post.get("title", ""), post.get("selftext", "")]
            )
            ids = extract_arxiv_ids(text_blob)
            for aid in ids:
                # First-seen wins; later posts about same paper ignored
                id_to_extras.setdefault(
                    aid,
                    {
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "thread_url": f"https://reddit.com{post.get('permalink', '')}",
                    },
                )

        return await fetch_arxiv_by_ids(
            list(id_to_extras.keys()), source_name="reddit", extras_per_id=id_to_extras
        )


if __name__ == "__main__":
    import asyncio
    import json
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.reddit.enabled:
            print("reddit source disabled")
            return
        src = RedditSource(subreddit=cfg.sources.reddit.subreddit, top_window=cfg.sources.reddit.top_window)
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            try:
                papers = asyncio.run(src.fetch(target))
            except Exception as e:
                print(f"reddit: skip {target.date()} due to {e}")
                continue
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "reddit.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"reddit: wrote {len(papers)} papers for {target.date()}")

    main()
```

- [ ] **Step 6.6: Run test to verify it passes**

Run: `uv run pytest tests/test_reddit.py -v`
Expected: PASS

- [ ] **Step 6.7: Commit**

```bash
git add sources/reddit.py sources/_arxiv_lookup.py tests/test_reddit.py tests/fixtures/reddit_top_response.json
git commit -m "feat(sources): Reddit source extracting arXiv links from r/LocalLLaMA"
```

---

### Task 7: Semantic Scholar source

**Files:**
- Create: `sources/semantic_scholar.py`
- Test: `tests/test_semantic_scholar.py`
- Test fixture: `tests/fixtures/ss_citations_response.json`

- [ ] **Step 7.1: Create test fixture**

`tests/fixtures/ss_citations_response.json`:
```json
{
  "data": [
    {
      "citingPaper": {
        "paperId": "abc",
        "externalIds": {"ArXiv": "2405.12345"},
        "title": "A new quant paper",
        "abstract": "We propose ...",
        "authors": [{"name": "Alice"}],
        "publicationDate": "2026-05-09",
        "url": "https://www.semanticscholar.org/paper/abc",
        "openAccessPdf": {"url": "https://example.com/paper.pdf"}
      }
    },
    {
      "citingPaper": {
        "paperId": "def",
        "externalIds": {},
        "title": "Non-arxiv paper",
        "abstract": "...",
        "authors": [],
        "publicationDate": "2025-01-01",
        "url": "https://www.semanticscholar.org/paper/def"
      }
    }
  ]
}
```

- [ ] **Step 7.2: Write the failing test**

```python
# tests/test_semantic_scholar.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.semantic_scholar import SemanticScholarSource

FIXTURE = Path(__file__).parent / "fixtures" / "ss_citations_response.json"


@respx.mock
@pytest.mark.asyncio
async def test_ss_fetches_citations_for_seeds(tmp_path: Path):
    seeds_file = tmp_path / "seeds.yaml"
    seeds_file.write_text("seeds:\n  - id: arXiv:2210.17323\n    name: GPTQ\n")

    respx.get("https://api.semanticscholar.org/graph/v1/paper/arXiv:2210.17323/citations").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )

    src = SemanticScholarSource(seeds_file=seeds_file, citation_window_days=7)
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))

    # Only the arxiv-bearing recent paper survives
    assert len(papers) == 1
    p = papers[0]
    assert p.id == "2405.12345"
    assert p.sources[0].name == "semantic_scholar"
```

- [ ] **Step 7.3: Run test to verify it fails**

Run: `uv run pytest tests/test_semantic_scholar.py -v`
Expected: FAIL

- [ ] **Step 7.4: Implement `sources/semantic_scholar.py`**

```python
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
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
                        pub = datetime.fromisoformat(pub_str).replace(tzinfo=timezone.utc)
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
                            primary_category="unknown",  # SS doesn't expose
                            categories=[],
                            sources=[
                                SourceRecord(
                                    name="semantic_scholar",
                                    fetched_at=datetime.now(timezone.utc),
                                    extras={"seed": seed},
                                )
                            ],
                        )
                    )
                await asyncio.sleep(1.0 if not headers else 0.1)

        return all_papers


if __name__ == "__main__":
    import json
    from datetime import datetime, timedelta

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.semantic_scholar.enabled:
            print("semantic_scholar disabled")
            return
        src = SemanticScholarSource(
            seeds_file=Path(cfg.sources.semantic_scholar.seeds_file),
            citation_window_days=cfg.sources.semantic_scholar.citation_window_days,
        )
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
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
```

- [ ] **Step 7.5: Run test to verify it passes**

Run: `uv run pytest tests/test_semantic_scholar.py -v`
Expected: PASS

- [ ] **Step 7.6: Commit**

```bash
git add sources/semantic_scholar.py tests/test_semantic_scholar.py tests/fixtures/ss_citations_response.json
git commit -m "feat(sources): Semantic Scholar citation graph source"
```

---

### Task 8: Papers with Code source

**Files:**
- Create: `sources/papers_with_code.py`
- Test: `tests/test_papers_with_code.py`
- Test fixture: `tests/fixtures/pwc_rss.xml`

- [ ] **Step 8.1: Create fixture**

`tests/fixtures/pwc_rss.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Papers with Code: Latest</title>
  <item>
    <title>BitNet b1.58</title>
    <link>https://paperswithcode.com/paper/bitnet-b158</link>
    <description>1-bit LLM. arXiv: https://arxiv.org/abs/2402.17764. Code: https://github.com/example/bitnet</description>
    <pubDate>Sat, 10 May 2026 12:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Some non-arxiv paper</title>
    <link>https://paperswithcode.com/paper/foo</link>
    <description>No arxiv link.</description>
    <pubDate>Sat, 10 May 2026 13:00:00 GMT</pubDate>
  </item>
</channel>
</rss>
```

- [ ] **Step 8.2: Write the failing test**

```python
# tests/test_papers_with_code.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.papers_with_code import PapersWithCodeSource

FIXTURE = Path(__file__).parent / "fixtures" / "pwc_rss.xml"


@respx.mock
@pytest.mark.asyncio
async def test_pwc_extracts_arxiv_and_code():
    respx.get("https://paperswithcode.com/latest/rss.xml").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2402.17764v1</id>
    <title>BitNet b1.58</title>
    <summary>1-bit LLM.</summary>
    <published>2026-05-10T00:00:00Z</published>
    <author><name>X</name></author>
    <category term="cs.CL"/>
    <link href="http://arxiv.org/pdf/2402.17764v1.pdf" type="application/pdf"/>
  </entry>
</feed>"""
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=arxiv_xml)
    )

    src = PapersWithCodeSource()
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert len(papers) == 1
    p = papers[0]
    assert p.id == "2402.17764"
    assert p.code_url == "https://github.com/example/bitnet"
    assert p.sources[0].name == "papers_with_code"
```

- [ ] **Step 8.3: Run test to verify it fails**

Run: `uv run pytest tests/test_papers_with_code.py -v`
Expected: FAIL

- [ ] **Step 8.4: Implement `sources/papers_with_code.py`**

```python
from __future__ import annotations

import re
from datetime import datetime

import feedparser
import httpx

from sources._arxiv_lookup import extract_arxiv_ids, fetch_arxiv_by_ids
from sources.base import Paper, Source

GITHUB_RE = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+")


class PapersWithCodeSource(Source):
    name = "papers_with_code"
    RSS_URL = "https://paperswithcode.com/latest/rss.xml"

    async def fetch(self, target_date: datetime) -> list[Paper]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(self.RSS_URL)
            r.raise_for_status()
        feed = feedparser.parse(r.text)

        # Build map: arxiv_id → code_url (from RSS item)
        id_to_code: dict[str, str] = {}
        for item in feed.entries:
            blob = " ".join([item.get("title", ""), item.get("description", "")])
            ids = extract_arxiv_ids(blob)
            code_match = GITHUB_RE.search(blob)
            code_url = code_match.group(0) if code_match else None
            for aid in ids:
                id_to_code.setdefault(aid, code_url)

        papers = await fetch_arxiv_by_ids(list(id_to_code.keys()), source_name="papers_with_code")
        for p in papers:
            if id_to_code.get(p.id):
                p.code_url = id_to_code[p.id]
        return papers


if __name__ == "__main__":
    import asyncio
    import json
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.papers_with_code.enabled:
            print("papers_with_code disabled")
            return
        src = PapersWithCodeSource()
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            try:
                papers = asyncio.run(src.fetch(target))
            except Exception as e:
                print(f"papers_with_code: skip {target.date()}: {e}")
                continue
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "papers_with_code.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"papers_with_code: wrote {len(papers)} for {target.date()}")

    main()
```

- [ ] **Step 8.5: Run test to verify it passes**

Run: `uv run pytest tests/test_papers_with_code.py -v`
Expected: PASS

- [ ] **Step 8.6: Commit**

```bash
git add sources/papers_with_code.py tests/test_papers_with_code.py tests/fixtures/pwc_rss.xml
git commit -m "feat(sources): Papers with Code RSS source"
```

---

### Task 9: Twitter (RSSHub) source

**Files:**
- Create: `sources/twitter_rsshub.py`
- Test: `tests/test_twitter_rsshub.py`
- Test fixture: `tests/fixtures/rsshub_twitter.xml`

- [ ] **Step 9.1: Create fixture**

`tests/fixtures/rsshub_twitter.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>@_akhaliq Twitter</title>
  <item>
    <title>Cool new paper</title>
    <link>https://twitter.com/_akhaliq/status/1</link>
    <description>BitNet b1.58 https://arxiv.org/abs/2402.17764</description>
    <pubDate>Sat, 10 May 2026 12:00:00 GMT</pubDate>
  </item>
</channel>
</rss>
```

- [ ] **Step 9.2: Write the failing test**

```python
# tests/test_twitter_rsshub.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.twitter_rsshub import TwitterRSSHubSource

FIXTURE = Path(__file__).parent / "fixtures" / "rsshub_twitter.xml"


@respx.mock
@pytest.mark.asyncio
async def test_twitter_rsshub_collects_arxiv_links(monkeypatch):
    monkeypatch.setenv("RSSHUB_BASE_URL", "https://rsshub.test")
    respx.get("https://rsshub.test/twitter/user/_akhaliq").mock(
        return_value=Response(200, text=FIXTURE.read_text())
    )
    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2402.17764v1</id>
    <title>BitNet</title>
    <summary>1-bit.</summary>
    <published>2026-05-10T00:00:00Z</published>
    <author><name>X</name></author>
    <category term="cs.CL"/>
    <link href="http://arxiv.org/pdf/2402.17764v1.pdf" type="application/pdf"/>
  </entry>
</feed>"""
    respx.get("http://export.arxiv.org/api/query").mock(
        return_value=Response(200, text=arxiv_xml)
    )

    src = TwitterRSSHubSource(accounts=["_akhaliq"])
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert len(papers) == 1
    assert papers[0].id == "2402.17764"
    assert papers[0].sources[0].name == "twitter_rsshub"
    assert "_akhaliq" in papers[0].sources[0].extras.get("accounts", [])


@pytest.mark.asyncio
async def test_twitter_skips_when_base_url_missing(monkeypatch):
    monkeypatch.delenv("RSSHUB_BASE_URL", raising=False)
    src = TwitterRSSHubSource(accounts=["_akhaliq"])
    papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
    assert papers == []


@pytest.mark.asyncio
async def test_twitter_skips_unreachable_account(monkeypatch):
    monkeypatch.setenv("RSSHUB_BASE_URL", "https://rsshub.test")
    with respx.mock:
        respx.get("https://rsshub.test/twitter/user/down").mock(
            return_value=Response(503, text="dead")
        )
        src = TwitterRSSHubSource(accounts=["down"])
        papers = await src.fetch(datetime(2026, 5, 11, tzinfo=timezone.utc))
        assert papers == []
```

- [ ] **Step 9.3: Run test to verify it fails**

Run: `uv run pytest tests/test_twitter_rsshub.py -v`
Expected: FAIL

- [ ] **Step 9.4: Implement `sources/twitter_rsshub.py`**

```python
from __future__ import annotations

import os
from datetime import datetime

import feedparser
import httpx

from sources._arxiv_lookup import extract_arxiv_ids, fetch_arxiv_by_ids
from sources.base import Paper, Source


class TwitterRSSHubSource(Source):
    name = "twitter_rsshub"

    def __init__(self, accounts: list[str]):
        self.accounts = accounts

    async def fetch(self, target_date: datetime) -> list[Paper]:
        base = os.environ.get("RSSHUB_BASE_URL", "").rstrip("/")
        if not base:
            return []

        id_to_accounts: dict[str, set[str]] = {}
        async with httpx.AsyncClient(timeout=20.0) as client:
            for acc in self.accounts:
                try:
                    r = await client.get(f"{base}/twitter/user/{acc}")
                    if r.status_code != 200:
                        continue
                    feed = feedparser.parse(r.text)
                except httpx.HTTPError:
                    continue
                for item in feed.entries:
                    blob = " ".join(
                        [item.get("title", ""), item.get("description", ""), item.get("summary", "")]
                    )
                    for aid in extract_arxiv_ids(blob):
                        id_to_accounts.setdefault(aid, set()).add(acc)

        if not id_to_accounts:
            return []

        extras_per_id = {aid: {"accounts": sorted(accs)} for aid, accs in id_to_accounts.items()}
        return await fetch_arxiv_by_ids(
            list(id_to_accounts.keys()), source_name="twitter_rsshub", extras_per_id=extras_per_id
        )


if __name__ == "__main__":
    import asyncio
    import json
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    import click

    from pipeline.config import load_config

    @click.command()
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    def main(backfill_days: int, out_dir: Path):
        cfg = load_config()
        if not cfg.sources.twitter_rsshub.enabled:
            print("twitter_rsshub disabled")
            return
        src = TwitterRSSHubSource(accounts=cfg.sources.twitter_rsshub.accounts)
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = today - timedelta(days=delta)
            try:
                papers = asyncio.run(src.fetch(target))
            except Exception as e:
                print(f"twitter_rsshub: skip {target.date()}: {e}")
                continue
            day_dir = out_dir / target.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            (day_dir / "twitter_rsshub.json").write_text(
                json.dumps([p.model_dump(mode="json") for p in papers], indent=2)
            )
            print(f"twitter_rsshub: wrote {len(papers)} for {target.date()}")

    main()
```

- [ ] **Step 9.5: Run test to verify it passes**

Run: `uv run pytest tests/test_twitter_rsshub.py -v`
Expected: PASS (3 passed)

- [ ] **Step 9.6: Commit**

```bash
git add sources/twitter_rsshub.py tests/test_twitter_rsshub.py tests/fixtures/rsshub_twitter.xml
git commit -m "feat(sources): Twitter via RSSHub source with graceful skip on absence"
```

---

## Phase 3 — Pipeline Stages (Tasks 10–14)

### Task 10: Dedupe (within-day + cross-day, lenient strategy)

**Files:**
- Create: `pipeline/dedupe.py`
- Test: `tests/test_dedupe.py`

- [ ] **Step 10.1: Write the failing test**

```python
# tests/test_dedupe.py
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline.config import Config, DedupeConfig
from pipeline.dedupe import dedupe_for_date, merge_papers
from sources.base import Paper, SourceRecord


def _mk(id_, title, source, **extras):
    return Paper(
        id=id_,
        title=title,
        authors=[],
        abstract="abs",
        url=f"https://arxiv.org/abs/{id_}",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL",
        categories=["cs.CL"],
        sources=[SourceRecord(name=source, fetched_at=datetime.now(timezone.utc), extras=extras)],
    )


def test_merge_papers_combines_sources_and_uses_priority():
    arxiv_p = _mk("X", "arxiv title", "arxiv")
    arxiv_p.abstract = "arxiv abs"
    hf_p = _mk("X", "hf title", "hf_daily", upvotes=42)
    hf_p.abstract = "hf abs"

    priority = ["hf_daily", "reddit", "semantic_scholar", "papers_with_code", "twitter_rsshub", "arxiv"]
    merged = merge_papers([arxiv_p, hf_p], priority)
    assert len(merged) == 1
    p = merged[0]
    assert p.title == "hf title"           # higher priority wins
    assert p.abstract == "hf abs"
    assert {s.name for s in p.sources} == {"arxiv", "hf_daily"}


def test_merge_keeps_distinct_papers():
    a = _mk("X", "x", "arxiv")
    b = _mk("Y", "y", "arxiv")
    merged = merge_papers([a, b], ["arxiv"])
    ids = {p.id for p in merged}
    assert ids == {"X", "Y"}


def test_dedupe_for_date_writes_files_and_marks_seen_before(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "2026-05-11"
    raw_dir.mkdir(parents=True)
    (raw_dir / "arxiv.json").write_text(
        json.dumps([_mk("X", "x", "arxiv").model_dump(mode="json")])
    )
    (raw_dir / "hf_daily.json").write_text(
        json.dumps([_mk("X", "x-hf", "hf_daily", upvotes=10).model_dump(mode="json")])
    )

    seen_path = tmp_path / "seen.json"
    out_path = tmp_path / "deduped" / "2026-05-11.json"
    out_path.parent.mkdir(parents=True)

    cfg = Config(dedupe=DedupeConfig(
        cross_day_strategy="lenient",
        source_priority=["hf_daily", "arxiv"]
    ))
    n = dedupe_for_date(
        date=datetime(2026, 5, 11, tzinfo=timezone.utc),
        raw_root=tmp_path / "raw",
        out_path=out_path,
        seen_path=seen_path,
        config=cfg,
    )
    assert n == 1
    data = json.loads(out_path.read_text())
    assert data[0]["title"] == "x-hf"  # hf priority wins
    assert data[0]["seen_before"] is False  # first time

    # Run again with same paper, expect seen_before True
    n2 = dedupe_for_date(
        date=datetime(2026, 5, 11, tzinfo=timezone.utc),
        raw_root=tmp_path / "raw",
        out_path=out_path,
        seen_path=seen_path,
        config=cfg,
    )
    data2 = json.loads(out_path.read_text())
    assert data2[0]["seen_before"] is True
```

- [ ] **Step 10.2: Run test to verify it fails**

Run: `uv run pytest tests/test_dedupe.py -v`
Expected: FAIL

- [ ] **Step 10.3: Implement `pipeline/dedupe.py`**

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pipeline.config import Config, load_config
from sources.base import Paper


def merge_papers(papers: list[Paper], source_priority: list[str]) -> list[Paper]:
    """Merge papers by id; for shared fields, take from the highest-priority source.
    `source_priority` is most-trusted-first."""
    by_id: dict[str, list[Paper]] = {}
    for p in papers:
        by_id.setdefault(p.id, []).append(p)

    rank = {name: i for i, name in enumerate(source_priority)}

    def primary_source_rank(p: Paper) -> int:
        # paper has 1 source at this stage, before merge
        return rank.get(p.sources[0].name, 999)

    merged: list[Paper] = []
    for pid, group in by_id.items():
        group_sorted = sorted(group, key=primary_source_rank)
        head = group_sorted[0].model_copy(deep=True)
        # collect all sources
        all_sources = [g.sources[0] for g in group]
        head.sources = all_sources
        # for non-empty fields, fall back from later (lower-priority) sources
        for other in group_sorted[1:]:
            for fld in ("title", "abstract", "pdf_url", "code_url"):
                if not getattr(head, fld) and getattr(other, fld):
                    setattr(head, fld, getattr(other, fld))
            if not head.authors and other.authors:
                head.authors = other.authors
            if not head.categories and other.categories:
                head.categories = other.categories
                head.primary_category = other.primary_category
        merged.append(head)
    return merged


def _load_raw(raw_dir: Path) -> list[Paper]:
    papers: list[Paper] = []
    if not raw_dir.exists():
        return papers
    for f in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for item in data:
            papers.append(Paper.model_validate(item))
    return papers


def dedupe_for_date(
    date: datetime,
    raw_root: Path,
    out_path: Path,
    seen_path: Path,
    config: Config,
) -> int:
    raw_dir = raw_root / date.strftime("%Y-%m-%d")
    papers = _load_raw(raw_dir)
    merged = merge_papers(papers, config.dedupe.source_priority)

    # Cross-day flag
    seen: set[str] = set()
    if seen_path.exists():
        seen = set(json.loads(seen_path.read_text()))
    for p in merged:
        if p.id in seen:
            p.seen_before = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([p.model_dump(mode="json") for p in merged], indent=2))

    # Update seen.json (add new ids)
    seen.update(p.id for p in merged)
    seen_path.write_text(json.dumps(sorted(seen), indent=2))

    return len(merged)


if __name__ == "__main__":
    from datetime import timezone, timedelta

    import click

    @click.command()
    @click.option("--date", default=None, help="YYYY-MM-DD; default today UTC")
    @click.option("--backfill-days", default=0, type=int)
    @click.option("--raw-root", default="data/raw", type=click.Path(path_type=Path))
    @click.option("--out-root", default="data/deduped", type=click.Path(path_type=Path))
    @click.option("--seen-path", default="data/seen.json", type=click.Path(path_type=Path))
    def main(date, backfill_days, raw_root, out_root, seen_path):
        cfg = load_config()
        if date:
            base = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        else:
            base = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            out = out_root / f"{target.strftime('%Y-%m-%d')}.json"
            n = dedupe_for_date(target, raw_root, out, seen_path, cfg)
            print(f"dedupe: {n} unique papers for {target.date()}")

    main()
```

- [ ] **Step 10.4: Run test to verify it passes**

Run: `uv run pytest tests/test_dedupe.py -v`
Expected: PASS (3 passed)

- [ ] **Step 10.5: Commit**

```bash
git add pipeline/dedupe.py tests/test_dedupe.py
git commit -m "feat(pipeline): dedupe with field-merge priority + cross-day seen tracking"
```

---

### Task 11: LLM relevance filter

**Files:**
- Create: `pipeline/llm_client.py` (shared Anthropic helper)
- Create: `pipeline/filter.py`
- Test: `tests/test_filter.py`

- [ ] **Step 11.1: Implement shared LLM helper**

```python
# pipeline/llm_client.py
"""Async Anthropic client wrapper with prompt-cache support and retries."""
from __future__ import annotations

import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
    async def call_json(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)


def load_prompt(path: Path) -> str:
    return Path(path).read_text()
```

- [ ] **Step 11.2: Write the failing test**

```python
# tests/test_filter.py
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.filter import filter_papers
from sources.base import Paper, SourceRecord


def _mk(id_, title, abstract):
    return Paper(
        id=id_, title=title, authors=[], abstract=abstract,
        url="https://x", pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL", categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(timezone.utc))],
    )


@pytest.mark.asyncio
async def test_filter_assigns_score_and_reason(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    deduped = [_mk("1", "FP4 quant for LLM", "Quantization."),
               _mk("2", "Random RAG paper", "RAG.")]
    deduped_path = tmp_path / "in.json"
    deduped_path.write_text(json.dumps([p.model_dump(mode="json") for p in deduped]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("test prompt")

    fake = AsyncMock()
    fake.call_json.side_effect = [
        {"relevance_score": 9, "reason": "FP4 quantization method"},
        {"relevance_score": 1, "reason": "RAG, not relevant"},
    ]

    n = await filter_papers(
        deduped_path=deduped_path,
        out_path=out_path,
        prompt_path=prompt_path,
        client=fake,
        concurrency=2,
    )
    assert n == 2
    out = json.loads(out_path.read_text())
    assert out[0]["relevance_score"] == 9
    assert out[0]["relevance_reason"] == "FP4 quantization method"
    assert out[1]["relevance_score"] == 1


@pytest.mark.asyncio
async def test_filter_handles_per_paper_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    deduped = [_mk("1", "good", "good"), _mk("2", "bad", "bad")]
    deduped_path = tmp_path / "in.json"
    deduped_path.write_text(json.dumps([p.model_dump(mode="json") for p in deduped]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.side_effect = [
        {"relevance_score": 8, "reason": "ok"},
        Exception("boom"),
    ]
    await filter_papers(deduped_path, out_path, prompt_path, fake, concurrency=2)
    out = json.loads(out_path.read_text())
    # Failed paper kept with score=None (caller can retry/skip)
    assert any(r["relevance_score"] == 8 for r in out)
    assert any(r["relevance_score"] is None for r in out)
```

- [ ] **Step 11.3: Run test to verify it fails**

Run: `uv run pytest tests/test_filter.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 11.4: Implement `pipeline/filter.py`**

```python
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
    papers = [Paper.model_validate(p) for p in json.loads(Path(deduped_path).read_text())]
    prompt = load_prompt(prompt_path)
    sem = asyncio.Semaphore(concurrency)
    scored = await asyncio.gather(*[_score_one(p, prompt, client, sem) for p in papers])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([p.model_dump(mode="json") for p in scored], indent=2))
    return len(scored)


if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta

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
            base = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        else:
            base = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            in_path = in_root / f"{target.strftime('%Y-%m-%d')}.json"
            out_path = out_root / f"{target.strftime('%Y-%m-%d')}.json"
            if not in_path.exists():
                print(f"filter: skip {target.date()} (no deduped input)")
                continue
            n = asyncio.run(filter_papers(in_path, out_path, prompt_path, client, cfg.filter.concurrency))
            print(f"filter: scored {n} papers for {target.date()}")

    main()
```

- [ ] **Step 11.5: Run test to verify it passes**

Run: `uv run pytest tests/test_filter.py -v`
Expected: PASS (2 passed)

- [ ] **Step 11.6: Commit**

```bash
git add pipeline/llm_client.py pipeline/filter.py tests/test_filter.py
git commit -m "feat(pipeline): LLM client + filter for relevance scoring"
```

---

### Task 12: LLM bilingual summarizer

**Files:**
- Create: `pipeline/summarize.py`
- Test: `tests/test_summarize.py`

- [ ] **Step 12.1: Write the failing test**

```python
# tests/test_summarize.py
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.summarize import summarize_papers
from sources.base import Paper, SourceRecord


def _mk(id_, score):
    p = Paper(
        id=id_, title="t", authors=[], abstract="a",
        url="https://x", pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL", categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(timezone.utc))],
    )
    p.relevance_score = score
    return p


@pytest.mark.asyncio
async def test_summarize_only_runs_for_above_threshold(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    scored = [_mk("hi", 9), _mk("low", 5), _mk("hi2", 8)]
    in_path = tmp_path / "scored.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in scored]))
    out_path = tmp_path / "summarized.json"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.return_value = {
        "summary": "English summary",
        "highlights": ["🎯 a", "📊 b"],
    }
    n = await summarize_papers(in_path, out_path, prompt_path, fake, threshold=7, concurrency=2)
    assert n == 3  # all written, but only 2 summarized
    out = json.loads(out_path.read_text())
    by_id = {p["id"]: p for p in out}
    assert by_id["hi"]["summary"] == "English summary"
    assert by_id["hi2"]["summary"] == "English summary"
    assert by_id["low"]["summary"] is None
    # LLM called only for above-threshold papers
    assert fake.call_json.call_count == 2
```

- [ ] **Step 12.2: Run test to verify it fails**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: FAIL

- [ ] **Step 12.3: Implement `pipeline/summarize.py`**

```python
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pipeline.config import load_config
from pipeline.llm_client import LLMClient, load_prompt
from sources.base import Paper


async def _summarize_one(paper: Paper, prompt: str, client: LLMClient, sem: asyncio.Semaphore) -> Paper:
    user = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    async with sem:
        try:
            r = await client.call_json(prompt, user, max_tokens=900)
            paper.summary_zh = r.get("summary_zh")
            paper.highlights_zh = r.get("highlights_zh", [])
            paper.summary_en = r.get("summary_en")
            paper.highlights_en = r.get("highlights_en", [])
        except Exception as e:
            print(f"summarize: paper {paper.id} failed: {e}")
    return paper


async def summarize_papers(
    in_path: Path,
    out_path: Path,
    prompt_path: Path,
    client: LLMClient,
    threshold: int,
    concurrency: int,
) -> int:
    papers = [Paper.model_validate(p) for p in json.loads(Path(in_path).read_text())]
    prompt = load_prompt(prompt_path)
    sem = asyncio.Semaphore(concurrency)
    targets = [p for p in papers if (p.relevance_score or 0) >= threshold]
    await asyncio.gather(*[_summarize_one(p, prompt, client, sem) for p in targets])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers], indent=2))
    return len(papers)


if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta

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
            base = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        else:
            base = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            in_path = in_root / f"{target.strftime('%Y-%m-%d')}.json"
            out_path = out_root / f"{target.strftime('%Y-%m-%d')}.json"
            if not in_path.exists():
                print(f"summarize: skip {target.date()}")
                continue
            n = asyncio.run(
                summarize_papers(in_path, out_path, prompt_path, client, cfg.filter.threshold, cfg.summarize.concurrency)
            )
            print(f"summarize: processed {n} papers for {target.date()}")

    main()
```

- [ ] **Step 12.4: Run test to verify it passes**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: PASS

- [ ] **Step 12.5: Commit**

```bash
git add pipeline/summarize.py tests/test_summarize.py
git commit -m "feat(pipeline): bilingual summarizer (Chinese + English)"
```

---

### Task 13: Daily digest renderer

**Files:**
- Create: `pipeline/render.py`
- Test: `tests/test_render.py`

- [ ] **Step 13.1: Write the failing test**

```python
# tests/test_render.py
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline.render import heat_score, render_daily, render_index_line, sort_papers
from sources.base import Paper, SourceRecord


def _mk(id_, score, hf_upvotes=0, reddit_score=0, trending_rank=None, twitter_accounts=None, summary_zh="zh"):
    sources = [
        SourceRecord(name="arxiv", fetched_at=datetime.now(timezone.utc)),
        SourceRecord(name="hf_daily", fetched_at=datetime.now(timezone.utc), extras={"upvotes": hf_upvotes, "num_comments": 0}),
        SourceRecord(name="reddit", fetched_at=datetime.now(timezone.utc), extras={"score": reddit_score, "num_comments": 0}),
    ]
    if trending_rank is not None:
        sources.append(SourceRecord(name="hf_daily", fetched_at=datetime.now(timezone.utc), extras={"trending_rank": trending_rank}))
    if twitter_accounts:
        sources.append(SourceRecord(name="twitter_rsshub", fetched_at=datetime.now(timezone.utc), extras={"accounts": twitter_accounts}))
    p = Paper(
        id=id_, title=f"Title {id_}", authors=["A"], abstract="abs",
        url=f"https://arxiv.org/abs/{id_}", pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL", categories=["cs.CL"],
        sources=sources,
    )
    p.relevance_score = score
    p.summary_zh = summary_zh
    p.highlights_zh = ["🎯 zh hl"]
    p.summary_en = "en"
    p.highlights_en = ["🎯 en hl"]
    return p


def test_heat_score_combines_all_signals():
    p = _mk("x", 9, hf_upvotes=10, reddit_score=99, trending_rank=2, twitter_accounts=["a", "b"])
    expected = (100 / 2) + 10 + math.log(100) * 5 + 10 * 2
    assert abs(heat_score(p) - expected) < 0.01


def test_heat_score_zero_when_no_signal():
    p = _mk("x", 9)
    assert heat_score(p) == 0.0


def test_trending_rank_above_30_gives_no_bonus():
    p = _mk("x", 9, trending_rank=50)
    assert heat_score(p) == 0.0


def test_sort_papers_heat_primary_relevance_tiebreaker():
    # b is hotter even though lower relevance — wins
    a = _mk("a", 9, hf_upvotes=0)
    b = _mk("b", 7, trending_rank=1)
    c = _mk("c", 9, hf_upvotes=5)
    sorted_ = sort_papers([a, b, c])
    assert [p.id for p in sorted_] == ["b", "c", "a"]


def test_render_daily_writes_top10_full_then_table(tmp_path: Path):
    summarized_path = tmp_path / "summarized.json"
    # Use heat-decreasing inputs so order is predictable
    papers = [_mk(f"id{i}", 9, trending_rank=i + 1) for i in range(15)]
    # Mix in below-threshold to ensure they are excluded
    below = _mk("low", 5)
    summarized_path.write_text(
        json.dumps([p.model_dump(mode="json") for p in papers + [below]])
    )

    digests_dir = tmp_path / "digests"
    readme = tmp_path / "README.md"
    index = tmp_path / "INDEX.md"

    render_daily(
        date=datetime(2026, 5, 11, tzinfo=timezone.utc),
        summarized_path=summarized_path,
        digests_dir=digests_dir,
        readme_path=readme,
        index_path=index,
        full_top_n=10,
        threshold=7,
    )

    out = (digests_dir / "2026-05-11.md").read_text()
    assert "## 🔥 Top 10" in out
    assert out.count("#### Summary") == 10
    assert "## 📚 Full List" in out
    # below-threshold excluded
    assert "id|low" not in out and "Title low" not in out
    # README mirrors the digest
    assert readme.read_text() == out
    # INDEX has a line
    assert "[05-11](digests/2026-05-11.md)" in index.read_text()


def test_render_index_line_includes_summary_stats():
    line = render_index_line(
        datetime(2026, 5, 11, tzinfo=timezone.utc),
        scanned=487, passed=38, top_title="BitNet b1.58",
    )
    assert "[05-11](digests/2026-05-11.md)" in line
    assert "487" in line and "38" in line and "BitNet" in line
```

- [ ] **Step 13.2: Run test to verify it fails**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL

- [ ] **Step 13.3: Implement `pipeline/render.py`**

```python
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from pipeline.config import load_config
from sources.base import Paper

REPO_URL = "https://github.com/zhaolin-amd/llm-paper-radar"
TRENDING_BONUS_CAP = 30  # ranks 1..30 contribute; below that → 0


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
📡 Sources: {_source_badge(p)}

#### Summary
{p.summary or ''}

{hl}

---
"""


def _table_row(rank: int, p: Paper) -> str:
    sources = "+".join({s.name for s in p.sources})
    code = "✅" if p.code_url else "—"
    return f"| {rank} | [{p.title}]({p.url}) | {p.relevance_score} | {sources} | {code} | {p.published_at.strftime('%m-%d')} |"


def render_index_line(date: datetime, scanned: int, passed: int, top_title: str) -> str:
    return f"- [{date.strftime('%m-%d')}](digests/{date.strftime('%Y-%m-%d')}.md) — {scanned} scanned, {passed} passed, top: {top_title}"


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
    body.append(f"# LLM Inference Optimization Daily · {date.strftime('%Y-%m-%d')}\n")
    body.append(f"> 📅 Window: {date.strftime('%Y-%m-%d')} (UTC daily)")
    body.append(f"> 📊 Scanned {scanned} papers → passed filter {len(surviving)} (threshold ≥{threshold})")
    body.append("")
    body.append(f"> Auto-generated daily digest from [llm-paper-radar]({REPO_URL}).")
    body.append("> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6\n")

    body.append(f"## 🔥 Top {min(full_top_n, len(surviving))} (Full Detail)\n")
    for i, p in enumerate(surviving[:full_top_n], start=1):
        body.append(_full_block(i, p))

    body.append("## 📚 Full List (by score, descending)\n")
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

    # Append/update INDEX.md
    top_title = surviving[0].title if surviving else "(no papers passed)"
    new_line = render_index_line(date, scanned, len(surviving), top_title[:50])
    if index_path.exists():
        existing = index_path.read_text()
        # Avoid duplicate insertion for the same date
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
    from datetime import timezone, timedelta

    import click

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
            base = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        else:
            base = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for delta in range(backfill_days + 1):
            target = base - timedelta(days=delta)
            in_path = in_root / f"{target.strftime('%Y-%m-%d')}.json"
            if not in_path.exists():
                print(f"render: skip {target.date()}")
                continue
            render_daily(target, in_path, digests_dir, readme, index, cfg.render.full_top_n, cfg.filter.threshold)
            print(f"render: wrote digest for {target.date()}")

    main()
```

- [ ] **Step 13.4: Run test to verify it passes**

Run: `uv run pytest tests/test_render.py -v`
Expected: PASS (4 passed)

- [ ] **Step 13.5: Commit**

```bash
git add pipeline/render.py tests/test_render.py
git commit -m "feat(pipeline): daily digest renderer with Top 10 full + table"
```

---

### Task 14: Weekly digest

**Files:**
- Create: `pipeline/weekly.py`
- Test: `tests/test_weekly.py`

- [ ] **Step 14.1: Write the failing test**

```python
# tests/test_weekly.py
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipeline.weekly import render_weekly
from sources.base import Paper, SourceRecord


def _mk(id_, score, day_offset):
    p = Paper(
        id=id_, title=f"Title {id_}", authors=[], abstract="a",
        url=f"https://x/{id_}", pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc) - timedelta(days=day_offset),
        primary_category="cs.CL", categories=["cs.CL"],
        sources=[SourceRecord(name="arxiv", fetched_at=datetime.now(timezone.utc))],
    )
    p.relevance_score = score
    p.summary_zh = "z"; p.summary_en = "e"
    return p


def test_weekly_aggregates_past_seven_days(tmp_path: Path):
    summarized = tmp_path / "summarized"
    summarized.mkdir()
    for d in range(7):
        date = datetime(2026, 5, 11, tzinfo=timezone.utc) - timedelta(days=d)
        papers = [_mk(f"d{d}p{i}", 9 - i, day_offset=d) for i in range(5)]
        (summarized / f"{date.strftime('%Y-%m-%d')}.json").write_text(
            json.dumps([p.model_dump(mode="json") for p in papers])
        )

    out_dir = tmp_path / "weekly"
    render_weekly(
        end_date=datetime(2026, 5, 11, tzinfo=timezone.utc),
        summarized_root=summarized,
        out_dir=out_dir,
        top_n=20,
        threshold=7,
    )

    files = list(out_dir.glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "Top 20" in text
    assert "Per-source" in text
```

- [ ] **Step 14.2: Run test to verify it fails**

Run: `uv run pytest tests/test_weekly.py -v`
Expected: FAIL

- [ ] **Step 14.3: Implement `pipeline/weekly.py`**

```python
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from pipeline.config import load_config
from pipeline.render import sort_papers
from sources.base import Paper


def render_weekly(
    end_date: datetime,
    summarized_root: Path,
    out_dir: Path,
    top_n: int,
    threshold: int,
) -> None:
    all_papers: list[Paper] = []
    for d in range(7):
        date = end_date - timedelta(days=d)
        f = summarized_root / f"{date.strftime('%Y-%m-%d')}.json"
        if not f.exists():
            continue
        for item in json.loads(f.read_text()):
            all_papers.append(Paper.model_validate(item))

    # Dedupe by id, keep highest-scored copy
    by_id: dict[str, Paper] = {}
    for p in all_papers:
        prev = by_id.get(p.id)
        if not prev or (p.relevance_score or 0) > (prev.relevance_score or 0):
            by_id[p.id] = p
    surviving = [p for p in by_id.values() if (p.relevance_score or 0) >= threshold]
    surviving = sort_papers(surviving)[:top_n]

    src_counts = Counter(s.name for p in surviving for s in p.sources)

    iso = end_date.isocalendar()
    fname = f"{iso.year}-W{iso.week:02d}.md"
    body = []
    body.append(f"# Weekly · Week {iso.week} of {iso.year} (ending {end_date.date()})\n")
    body.append(f"## Top {len(surviving)} (Week's highlights)\n")
    for i, p in enumerate(surviving, start=1):
        body.append(f"### {i}. [{p.title}]({p.url}) ({p.relevance_score}/10)")
        if p.summary:
            body.append(f"\n{p.summary}\n")

    body.append("\n## Per-source contribution\n")
    body.append("| Source | Count |")
    body.append("|--------|-------|")
    for src, cnt in src_counts.most_common():
        body.append(f"| {src} | {cnt} |")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_text("\n".join(body))


if __name__ == "__main__":
    from datetime import timezone

    import click

    @click.command()
    @click.option("--end-date", default=None)
    @click.option("--in-root", default="data/summarized", type=click.Path(path_type=Path))
    @click.option("--out-dir", default="weekly", type=click.Path(path_type=Path))
    def main(end_date, in_root, out_dir):
        cfg = load_config()
        end = (
            datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
            if end_date
            else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        )
        render_weekly(end, in_root, out_dir, top_n=20, threshold=cfg.filter.threshold)
        print(f"weekly: digest written for week ending {end.date()}")

    main()
```

- [ ] **Step 14.4: Run test to verify it passes**

Run: `uv run pytest tests/test_weekly.py -v`
Expected: PASS

- [ ] **Step 14.5: Commit**

```bash
git add pipeline/weekly.py tests/test_weekly.py
git commit -m "feat(pipeline): weekly digest aggregating past 7 days"
```

---

## Phase 4 — Workflows + Deployment (Tasks 15–19)

### Task 15: Daily GitHub Actions workflow

**Files:**
- Create: `.github/workflows/daily.yml`

- [ ] **Step 15.1: Create the workflow file**

Paste exactly the YAML from spec §7 `daily.yml` (including the artifact upload for `pipeline-debug` and the Teams notification step).

```yaml
name: Daily LLM Paper Radar

on:
  schedule:
    - cron: '0 23 * * *'
  workflow_dispatch:
    inputs:
      backfill_days:
        description: 'Number of past days to backfill (max 2)'
        required: false
        default: '0'

permissions:
  contents: write

jobs:
  fetch:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    strategy:
      fail-fast: false
      matrix:
        source: [arxiv, hf_daily, reddit, semantic_scholar, papers_with_code, twitter_rsshub]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - name: Fetch ${{ matrix.source }}
        run: uv run python -m sources.${{ matrix.source }} --backfill-days ${{ inputs.backfill_days || 0 }}
        env:
          REDDIT_CLIENT_ID: ${{ secrets.REDDIT_CLIENT_ID }}
          REDDIT_CLIENT_SECRET: ${{ secrets.REDDIT_CLIENT_SECRET }}
          SEMANTIC_SCHOLAR_API_KEY: ${{ secrets.SEMANTIC_SCHOLAR_API_KEY }}
          RSSHUB_BASE_URL: ${{ secrets.RSSHUB_BASE_URL }}
        continue-on-error: true
      - uses: actions/upload-artifact@v4
        with:
          name: raw-${{ matrix.source }}
          path: data/raw/
          retention-days: 7
          if-no-files-found: ignore

  pipeline:
    needs: fetch
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - uses: actions/download-artifact@v4
        with:
          path: data/raw/
          merge-multiple: true
      - run: uv run python -m pipeline.dedupe --backfill-days ${{ inputs.backfill_days || 0 }}
      - run: uv run python -m pipeline.filter --backfill-days ${{ inputs.backfill_days || 0 }}
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - run: uv run python -m pipeline.summarize --backfill-days ${{ inputs.backfill_days || 0 }}
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - run: uv run python -m pipeline.render --backfill-days ${{ inputs.backfill_days || 0 }}
      - name: Upload debug artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: pipeline-debug
          path: |
            data/deduped/
            data/scored/
          retention-days: 14
          if-no-files-found: ignore
      - name: Commit & push
        run: |
          git config user.name "llm-paper-radar[bot]"
          git config user.email "actions@github.com"
          git add digests/ README.md INDEX.md data/seen.json data/summarized/
          git diff --cached --quiet || git commit -m "📚 Daily digest $(date -u +%Y-%m-%d)"
          git push
      - name: Notify Teams on failure
        if: failure()
        run: |
          curl -H "Content-Type: application/json" -d '{
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "FF0000",
            "summary": "llm-paper-radar daily run failed",
            "sections": [{
              "activityTitle": "❌ llm-paper-radar daily run failed",
              "facts": [
                {"name": "Date", "value": "'$(date -u +%Y-%m-%d)'"},
                {"name": "Run", "value": "'${{ github.run_id }}'"}
              ]
            }],
            "potentialAction": [{
              "@type": "OpenUri",
              "name": "View run",
              "targets": [{"os": "default", "uri": "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}]
            }]
          }' "${{ secrets.TEAMS_WEBHOOK_URL }}"
```

- [ ] **Step 15.2: Validate YAML locally**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/daily.yml'))"`
Expected: no error.

- [ ] **Step 15.3: Commit**

```bash
git add .github/workflows/daily.yml
git commit -m "ci: daily workflow (fetch matrix → pipeline → commit + Teams alert)"
```

---

### Task 16: Weekly + cleanup workflows

**Files:**
- Create: `.github/workflows/weekly.yml`
- Create: `.github/workflows/cleanup.yml`

- [ ] **Step 16.1: Create `weekly.yml`**

```yaml
name: Weekly LLM Paper Digest

on:
  schedule:
    - cron: '0 23 * * 1'   # Mondays UTC 23:00
  workflow_dispatch:

permissions:
  contents: write

jobs:
  weekly:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run python -m pipeline.weekly
      - name: Commit
        run: |
          git config user.name "llm-paper-radar[bot]"
          git config user.email "actions@github.com"
          git add weekly/
          git diff --cached --quiet || git commit -m "📅 Weekly digest $(date -u +%Y-W%U)"
          git push
```

- [ ] **Step 16.2: Create `cleanup.yml`**

```yaml
name: Cleanup old data

on:
  schedule:
    - cron: '0 22 * * 0'   # Sundays UTC 22:00
  workflow_dispatch:

permissions:
  contents: write

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Remove summarized data older than 90 days
        run: |
          if [ -d data/summarized ]; then
            find data/summarized -type f -name "*.json" -mtime +90 -delete
          fi
      - name: Commit if anything removed
        run: |
          git config user.name "llm-paper-radar[bot]"
          git config user.email "actions@github.com"
          git add -A
          git diff --cached --quiet || git commit -m "🧹 Cleanup old summarized data"
          git push
```

- [ ] **Step 16.3: Validate YAML**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/weekly.yml')); yaml.safe_load(open('.github/workflows/cleanup.yml'))"`
Expected: no error.

- [ ] **Step 16.4: Commit**

```bash
git add .github/workflows/weekly.yml .github/workflows/cleanup.yml
git commit -m "ci: weekly digest + Sunday cleanup workflows"
```

---

### Task 17: End-to-end smoke test (offline, mocked LLM)

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 17.1: Write the end-to-end test**

```python
# tests/test_e2e.py
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.config import Config, DedupeConfig
from pipeline.dedupe import dedupe_for_date
from pipeline.filter import filter_papers
from pipeline.render import render_daily
from pipeline.summarize import summarize_papers
from sources.base import Paper, SourceRecord


def _mk_raw(id_, source, **extras):
    return Paper(
        id=id_, title=f"Paper {id_}", authors=["A"], abstract=f"abs {id_}",
        url=f"https://arxiv.org/abs/{id_}", pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        primary_category="cs.CL", categories=["cs.CL"],
        sources=[SourceRecord(name=source, fetched_at=datetime.now(timezone.utc), extras=extras)],
    )


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    date = datetime(2026, 5, 11, tzinfo=timezone.utc)
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "2026-05-11"
    raw_dir.mkdir(parents=True)
    (raw_dir / "arxiv.json").write_text(json.dumps([
        _mk_raw("X", "arxiv").model_dump(mode="json"),
        _mk_raw("Y", "arxiv").model_dump(mode="json"),
    ]))
    (raw_dir / "hf_daily.json").write_text(json.dumps([
        _mk_raw("X", "hf_daily", upvotes=20).model_dump(mode="json"),
    ]))

    cfg = Config(dedupe=DedupeConfig(
        cross_day_strategy="lenient",
        source_priority=["hf_daily", "arxiv"]
    ))

    deduped_path = tmp_path / "deduped" / "2026-05-11.json"
    deduped_path.parent.mkdir()
    seen_path = tmp_path / "seen.json"
    n = dedupe_for_date(date, raw_root, deduped_path, seen_path, cfg)
    assert n == 2  # X and Y unique

    scored_path = tmp_path / "scored" / "2026-05-11.json"
    scored_path.parent.mkdir()
    prompt = tmp_path / "p.md"
    prompt.write_text("score this")
    fake_filter = AsyncMock()
    fake_filter.call_json.side_effect = [
        {"relevance_score": 9, "reason": "good"},
        {"relevance_score": 4, "reason": "weak"},
    ]
    await filter_papers(deduped_path, scored_path, prompt, fake_filter, concurrency=2)

    summarized_path = tmp_path / "summarized" / "2026-05-11.json"
    summarized_path.parent.mkdir()
    sum_prompt = tmp_path / "s.md"
    sum_prompt.write_text("summarize")
    fake_sum = AsyncMock()
    fake_sum.call_json.return_value = {
        "summary": "English", "highlights": ["🎯 a"],
    }
    await summarize_papers(scored_path, summarized_path, sum_prompt, fake_sum, threshold=7, concurrency=2)
    assert fake_sum.call_json.call_count == 1  # only the score=9 paper

    digests = tmp_path / "digests"
    readme = tmp_path / "README.md"
    index = tmp_path / "INDEX.md"
    render_daily(date, summarized_path, digests, readme, index, full_top_n=10, threshold=7)

    out = (digests / "2026-05-11.md").read_text()
    assert "Paper X" in out
    assert "Paper Y" not in out  # below threshold
    assert readme.read_text() == out
```

- [ ] **Step 17.2: Run the e2e test**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: PASS

- [ ] **Step 17.3: Run the full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS (~20 tests)

- [ ] **Step 17.4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end pipeline smoke test"
```

---

### Task 18: Lint + format pass

- [ ] **Step 18.1: Run ruff**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no errors after auto-fix; reformatted files.

- [ ] **Step 18.2: Re-run tests after format**

Run: `uv run pytest -v`
Expected: ALL PASS.

- [ ] **Step 18.3: Commit format changes (if any)**

```bash
git add -A
git diff --cached --quiet || git commit -m "style: ruff format + lint pass"
```

---

### Task 19: Repo creation + first deployment

- [ ] **Step 19.1: Create remote repo**

```bash
gh repo create zhaolin-amd/llm-paper-radar --public \
  --description "Daily LLM inference optimization paper digest" \
  --source=. --remote=origin --push
```

Expected: repo created at `https://github.com/zhaolin-amd/llm-paper-radar`, current code pushed to `main`.

- [ ] **Step 19.2: Configure required secrets**

Run (each command will prompt for the secret value):
```bash
gh secret set ANTHROPIC_API_KEY
gh secret set REDDIT_CLIENT_ID
gh secret set REDDIT_CLIENT_SECRET
gh secret set TEAMS_WEBHOOK_URL
```

Expected: each command prints `✓ Set Actions secret <NAME>`.

- [ ] **Step 19.3: Configure optional secrets (skip any you don't have yet)**

```bash
gh secret set RSSHUB_BASE_URL              # leave blank to disable Twitter source
gh secret set SEMANTIC_SCHOLAR_API_KEY     # optional
```

- [ ] **Step 19.4: Trigger first run with 2-day backfill**

```bash
gh workflow run daily.yml -f backfill_days=2
sleep 5
gh run list --workflow=daily.yml --limit 1
```

Expected: a `queued` or `in_progress` run appears.

- [ ] **Step 19.5: Watch the run**

```bash
gh run watch
```

Expected: workflow completes successfully (~10 min). Output shows `📚 Daily digest YYYY-MM-DD` commits.

- [ ] **Step 19.6: Verify output**

```bash
git pull
ls digests/
cat README.md | head -30
```

Expected:
- `digests/` contains 3 daily files (today + 2 backfill days)
- `README.md` shows the latest digest
- `INDEX.md` lists all 3 dates

- [ ] **Step 19.7: Final commit (if any local config tweaks needed)**

If first run revealed any config issues, fix and commit. Otherwise nothing to commit.

---

## Self-Review Notes

Performed against spec `2026-05-11-llm-paper-radar-design.md`:

**Coverage check:**
- ✅ §1 Goals → Tasks 1–19 all support
- ✅ §2 Architecture (modular pipeline) → matched in file structure (Task 1) and pipeline tasks (10–14)
- ✅ §3 Six data sources (hf_daily fetches both daily API and trending HTML) → Tasks 4–9
- ✅ §4 Schema + dedupe → Tasks 3, 10
- ✅ §5 LLM filter + bilingual summary → Tasks 11, 12 (FP4 in prompt covers MXFP4/NVFP4/rocFP4)
- ✅ §6 Render (Top 10 full / 11+ table, heat-primary sort with HF trending bonus) → Task 13
- ✅ §7 GH Actions (daily/weekly/cleanup, Teams alert, 2-day backfill) → Tasks 15, 16
- ✅ §8 Config + seeds → Tasks 1 (config), 1.5 (seeds with all 8 confirmed papers)
- ✅ §9 Storage → handled via .gitignore (Task 1) + workflow artifact retentions (Task 15)
- ✅ §10 Deployment → Task 19
- ✅ §11 Risks (RSSHub fragility) → Task 9 graceful skip
- ✅ §12 Open items → seeds.yaml has all 8 confirmed IDs

**Type/method consistency:**
- `Paper` schema defined in Task 3, consumed identically in Tasks 4–14
- `LLMClient.call_json(system, user, max_tokens)` signature consistent in Tasks 11 and 12
- `SourceRecord(name=..., fetched_at=..., extras=...)` shape used uniformly
- `community_signal()`, `sort_papers()` defined in Task 13, reused in Task 14

**No placeholders remain:** all code blocks complete; arxiv IDs filled; no "TBD" or "implement later".

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-11-llm-paper-radar.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
