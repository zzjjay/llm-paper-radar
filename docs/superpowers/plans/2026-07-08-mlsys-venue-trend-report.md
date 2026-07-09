# MLSys Venue Trend Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-shot fetch of all MLSys 2026 accepted papers from OpenReview, score them for "LLM inference deployment optimization" relevance + subfield, group by subfield, and produce a Markdown trend report via a parallel Workflow.

**Architecture:** New, self-contained files added to `llm-paper-radar` (fetch → score → group are plain Python CLI stages matching the repo's existing `sources/`/`pipeline/` pattern; trend analysis + synthesis is a Workflow script since those steps are independent-per-subfield and parallelizable). Nothing in the existing daily/weekly pipeline is touched.

**Tech Stack:** Python 3.11+, httpx, pydantic, click, pytest + respx + pytest-asyncio (all already in `pyproject.toml`), Claude Sonnet via the existing `pipeline/llm_client.py`, the Workflow tool for trend synthesis.

---

Reference spec: `docs/superpowers/specs/2026-07-08-mlsys-venue-trend-report-design.md`

## Task 1: Full-venue fetch with accept filtering, retry, and resumable page cache

**Files:**
- Create: `sources/openreview_venue.py`
- Test: `tests/test_openreview_venue.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_openreview_venue.py
from pathlib import Path

import pytest
import respx
from httpx import Response

from sources.openreview_venue import (
    VenueFetchIncomplete,
    _is_accepted,
    fetch_venue_accepted,
)


def _note(note_id: str, title: str, venueid: str) -> dict:
    return {
        "id": note_id,
        "cdate": 1750000000000,
        "content": {
            "title": {"value": title},
            "abstract": {"value": f"Abstract for {title}."},
            "authors": {"value": ["Alice"]},
            "venueid": {"value": venueid},
        },
    }


def test_is_accepted_matches_exact_venue_string():
    venue = "MLSys.org/2026/Conference"
    assert _is_accepted(_note("1", "X", venue), venue) is True
    assert _is_accepted(_note("1", "X", f"{venue}/Rejected_Submission"), venue) is False
    assert _is_accepted(_note("1", "X", f"{venue}/-/Submission"), venue) is False


@respx.mock
@pytest.mark.asyncio
async def test_fetch_venue_accepted_keeps_only_accepted_notes():
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    notes = [
        _note("a1", "Accepted paper", venue),
        _note("r1", "Rejected paper", f"{venue}/Rejected_Submission"),
    ]
    respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}"
        f"&offset=0&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(200, json={"notes": notes}))

    papers = await fetch_venue_accepted(venue)
    assert [p.title for p in papers] == ["Accepted paper"]


@respx.mock
@pytest.mark.asyncio
async def test_fetch_venue_accepted_retries_403_then_succeeds():
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    route = respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}"
        f"&offset=0&limit=1000&sort=cdate:desc"
    )
    route.side_effect = [
        Response(403, json={"message": "Challenge verification required"}),
        Response(200, json={"notes": [_note("a1", "Accepted paper", venue)]}),
    ]

    papers = await fetch_venue_accepted(venue)
    assert [p.title for p in papers] == ["Accepted paper"]
    assert route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_fetch_venue_accepted_raises_after_retries_exhausted():
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}"
        f"&offset=0&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(403, json={"message": "Challenge verification required"}))

    with pytest.raises(VenueFetchIncomplete):
        await fetch_venue_accepted(venue)


@respx.mock
@pytest.mark.asyncio
async def test_fetch_venue_accepted_resumes_from_page_cache(tmp_path: Path):
    """A cached page-0000.json is reused instead of re-fetched; only the
    (not-yet-cached) second page hits the network."""
    venue = "TEST.cc/2026/Conference"
    invitation = f"{venue}/-/Submission"
    cache_dir = tmp_path / "pages"
    cache_dir.mkdir()
    cached_page = [_note(f"cached-{i}", f"Cached {i}", venue) for i in range(1000)]
    (cache_dir / "page-0000.json").write_text(__import__("json").dumps(cached_page))

    route = respx.get(
        f"https://api2.openreview.net/notes?invitation={invitation}"
        f"&offset=1000&limit=1000&sort=cdate:desc"
    ).mock(return_value=Response(200, json={"notes": [_note("a2", "Second page paper", venue)]}))

    papers = await fetch_venue_accepted(venue, page_cache_dir=cache_dir)
    titles = {p.title for p in papers}
    assert "Cached 0" in titles
    assert "Second page paper" in titles
    assert route.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd llm-paper-radar && uv run pytest tests/test_openreview_venue.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sources.openreview_venue'`

- [ ] **Step 3: Implement `sources/openreview_venue.py`**

```python
"""One-shot fetch of a conference's full accepted-paper set from OpenReview.

Unlike `sources.openreview.OpenReviewSource` (which pulls a rolling window
of recently-created submissions for the daily/weekly incremental pipeline),
this fetches *every* submission for a venue, keeps only the ones whose
decision routed them to the venue itself, and returns them as `Paper`
objects. Meant for one-off conference batch analysis (e.g.
`scripts/venue_report.sh`), not the daily cron.

Decision detection follows the common OpenReview v2 convention: once
decisions are released, an accepted submission's `content.venueid.value` is
rewritten from the `/-/Submission` invitation venue to the venue string
itself (e.g. "MLSys.org/2026/Conference"); rejected/withdrawn submissions
get a `.../Rejected_Submission` or `.../Withdrawn_Submission` suffix
instead. This has not been verified against MLSys 2026 specifically — the
OpenReview API returned intermittent 403s throughout development (see
docs/superpowers/specs/2026-07-08-mlsys-venue-trend-report-design.md).
Run this module's CLI once the API is reachable, inspect a sample of
`content.venueid.value` values in the raw page cache, and adjust
`_is_accepted` if MLSys's actual field differs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from sources.base import Paper
from sources.openreview import _content_value, _note_pub, _note_to_paper

BASE_URL = "https://api2.openreview.net/notes"
PAGE_SIZE = 1000
RETRYABLE_STATUSES = {403, 429}


class VenueFetchIncomplete(RuntimeError):
    """Raised when a page could not be fetched after retries. Callers must
    not treat a partial result as final — see spec Section 3 (Error Handling)."""


def _is_accepted(note: dict, venue: str) -> bool:
    venueid = _content_value(note.get("content", {}) or {}, "venueid", "")
    return venueid == venue


async def _fetch_page(
    client: httpx.AsyncClient,
    invitation: str,
    offset: int,
    max_attempts: int = 6,
) -> list[dict]:
    url = (
        f"{BASE_URL}?invitation={invitation}"
        f"&offset={offset}&limit={PAGE_SIZE}&sort=cdate:desc"
    )
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = await client.get(url)
            if resp.status_code in RETRYABLE_STATUSES:
                wait = 5 * (2**attempt)
                print(
                    f"openreview_venue: {resp.status_code} at offset={offset}, "
                    f"attempt {attempt + 1}/{max_attempts}, sleep {wait}s"
                )
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("notes", [])
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            wait = 5 * (2**attempt)
            print(
                f"openreview_venue: {type(e).__name__} at offset={offset}, sleep {wait}s"
            )
            await asyncio.sleep(wait)
    raise VenueFetchIncomplete(
        f"gave up on offset={offset} after {max_attempts} attempts: {last_exc}"
    )


async def fetch_venue_accepted(
    venue: str,
    max_pages: int = 20,
    page_cache_dir: Path | None = None,
) -> list[Paper]:
    """Fetch every accepted paper for `venue`, e.g. "MLSys.org/2026/Conference".

    Paginates `/-/Submission` notes to completion (no time window), keeps
    only notes whose decision routed them to `venue` itself, and returns
    them as `Paper` objects. Raises `VenueFetchIncomplete` if any page fails
    after retries. If `page_cache_dir` is given, each raw page is persisted
    there and reused on a subsequent call, so a re-run after a mid-run
    failure resumes instead of restarting from offset 0.
    """
    invitation = f"{venue}/-/Submission"
    accepted: list[Paper] = []
    offset = 0
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for page_num in range(max_pages):
            cache_file = (
                page_cache_dir / f"page-{page_num:04d}.json" if page_cache_dir else None
            )
            if cache_file is not None and cache_file.exists():
                notes = json.loads(cache_file.read_text())
            else:
                notes = await _fetch_page(client, invitation, offset)
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(notes))
            if not notes:
                break
            for note in notes:
                if not _is_accepted(note, venue):
                    continue
                pub = _note_pub(note)
                if pub is None:
                    continue
                paper = _note_to_paper(note, venue, pub)
                if paper is not None:
                    accepted.append(paper)
            if len(notes) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            await asyncio.sleep(1.0)
    return accepted


if __name__ == "__main__":
    import click

    MIN_ACCEPTED_PAPERS = 20

    def _slug(venue: str) -> str:
        conf = venue.split(".", 1)[0].split("/", 1)[0].lower()
        year = venue.split("/")[1]
        return f"{conf}-{year}"

    @click.command()
    @click.option("--venue", required=True, help='e.g. "MLSys.org/2026/Conference"')
    @click.option("--out-dir", default="data/raw", type=click.Path(path_type=Path))
    @click.option("--min-papers", default=MIN_ACCEPTED_PAPERS, type=int)
    def main(venue: str, out_dir: Path, min_papers: int):
        venue_dir = out_dir / _slug(venue)
        page_cache_dir = venue_dir / "openreview_pages"
        try:
            papers = asyncio.run(fetch_venue_accepted(venue, page_cache_dir=page_cache_dir))
        except VenueFetchIncomplete as e:
            raise SystemExit(f"venue fetch incomplete, re-run later: {e}")
        if len(papers) < min_papers:
            raise SystemExit(
                f"only {len(papers)} accepted papers found for {venue} "
                f"(expected >= {min_papers}) — treating as incomplete, not writing output"
            )
        venue_dir.mkdir(parents=True, exist_ok=True)
        out_path = venue_dir / "accepted.json"
        out_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers], indent=2))
        print(f"openreview_venue: wrote {len(papers)} accepted papers to {out_path}")

    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-paper-radar && uv run pytest tests/test_openreview_venue.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd llm-paper-radar
git add sources/openreview_venue.py tests/test_openreview_venue.py
git commit -m "feat: add full-venue OpenReview fetch with accept filtering and resumable page cache"
```

---

## Task 2: Relevance/subfield scoring prompt and stage

**Files:**
- Create: `prompts/inference_relevance.md`
- Create: `pipeline/venue_filter.py`
- Test: `tests/test_venue_filter.py`

- [ ] **Step 1: Write `prompts/inference_relevance.md`**

```markdown
You are curating papers for an LLM inference deployment optimization team. The goal is to identify papers whose primary contribution improves how large language models are served/run in production — latency, throughput, memory, or cost of inference — as opposed to training, alignment, or generic ML-systems work with no LLM-inference angle.

# In scope

A paper is in scope if its primary contribution falls under one of these subfields (use these names verbatim when they fit; propose a short new name under "other" only when none fit):

- `kv_cache`: KV cache compression, eviction, quantization, or layout (StreamingLLM, H2O, KIVI, paged KV).
- `quantization`: PTQ/QAT for weights/activations, any bit-width, on a fixed pretrained LLM.
- `speculative_decoding`: draft-and-verify / parallel decoding schemes (EAGLE, Medusa, lookahead-style), with or without a compression angle.
- `scheduling_batching`: request scheduling, continuous batching, admission control, or serving-engine throughput work (vLLM/SGLang-style contributions).
- `moe_inference`: Mixture-of-Experts serving — expert placement, routing at inference time, expert offloading/caching.
- `long_context_pd_disaggregation`: long-context serving, prefill/decode disaggregation, context caching across requests.
- `multi_gpu_heterogeneous`: tensor/pipeline/expert parallelism for serving, heterogeneous-hardware deployment (mixed GPU generations, CPU offload for serving).
- `compiler_kernel_fusion`: inference compilers, kernel fusion, custom CUDA/Triton kernels targeting a fixed pretrained model's inference path.
- `other`: primary contribution is clearly LLM-inference-deployment-optimization but does not fit the above — name the actual subfield in 2-4 words.

# Out of scope — `hard_gate=true`

- Training-time-only work (pretraining recipes, RLHF/alignment, SFT data curation) with no inference-serving angle.
- Model releases / technical reports whose primary artifact is a new model, not an inference technique.
- Multimodal / vision-only / non-LLM applications without an LLM-inference angle.
- Pure hardware/ASIC/FPGA accelerator design papers whose primary contribution is a chip, not a technique applicable on commodity GPUs via a real inference stack (vLLM/SGLang/TensorRT-LLM/etc.).
- Evaluation/benchmark papers with no new inference technique.
- Anything whose primary contribution is not one of the in-scope subfields above.

# Output

Return JSON only, no prose, no markdown fences:

{
  "hard_gate": bool,
  "subfield": str,
  "reason": str
}

`subfield` must be one of the eight names above, or (if `other`) a short 2-4 word label prefixed with "other: ", e.g. "other: prompt caching". `reason` is 1-2 sentences on why the paper was gated or how it was classified.

# Language

`reason` MUST be written in **中文**. Technical terms (model names, method names, benchmark names, units) stay in English. `subfield` stays in English exactly as documented above.

# Few-shot anchors

- "PagedAttention: Efficient Memory Management for LLM Serving" → hard_gate=false, subfield=kv_cache
- "AWQ: Activation-aware Weight Quantization for LLM Compression" → hard_gate=false, subfield=quantization
- "EAGLE-3: Scaling up Inference Acceleration via Speculative Sampling" → hard_gate=false, subfield=speculative_decoding
- "Orca: A Distributed Serving System for Transformer-Based Generative Models" (continuous batching) → hard_gate=false, subfield=scheduling_batching
- "DeepSpeed-MoE: Advancing MoE Inference and Training" → hard_gate=false, subfield=moe_inference
- "Mooncake: A KV-Cache-centric Disaggregated Architecture for LLM Serving" → hard_gate=false, subfield=long_context_pd_disaggregation
- "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (heterogeneous GPU allocation across prefill/decode) → hard_gate=false, subfield=multi_gpu_heterogeneous
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (kernel-level fusion) → hard_gate=false, subfield=compiler_kernel_fusion
- "LoRA: Low-Rank Adaptation of Large Language Models" → hard_gate=true (training-time fine-tuning technique, no inference-serving angle)
- "Direct Preference Optimization" → hard_gate=true (alignment/training, no inference angle)
- "Qwen3 Technical Report" → hard_gate=true (model release, not an inference technique)
- "A Systolic-Array Accelerator for Transformer Inference on 7nm ASIC" → hard_gate=true (custom silicon, not applicable on commodity GPUs via a real inference stack)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_venue_filter.py
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pipeline.venue_filter import score_papers
from sources.base import Paper, SourceRecord


def _mk(id_, title, abstract):
    return Paper(
        id=id_,
        title=title,
        authors=[],
        abstract=abstract,
        url="https://x",
        pdf_url=None,
        published_at=datetime(2026, 5, 10, tzinfo=UTC),
        primary_category="mlsys",
        categories=["mlsys"],
        sources=[SourceRecord(name="openreview", fetched_at=datetime.now(UTC))],
    )


@pytest.mark.asyncio
async def test_score_papers_assigns_subfield_and_reason(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    papers = [
        _mk("1", "PagedAttention for LLM Serving", "KV cache paging for vLLM."),
        _mk("2", "LoRA fine-tuning", "Parameter-efficient fine-tuning."),
    ]
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "p.md"
    prompt_path.write_text("score this")

    fake = AsyncMock()
    fake.call_json.side_effect = [
        {"hard_gate": False, "subfield": "kv_cache", "reason": "KV cache 分页管理"},
        {"hard_gate": True, "subfield": "other", "reason": "训练期微调，无推理角度"},
    ]

    n = await score_papers(in_path, out_path, prompt_path, fake, concurrency=2)
    assert n == 2
    out = {p["id"]: p for p in json.loads(out_path.read_text())}
    assert out["1"]["relevance_breakdown"]["hard_gate"] is False
    assert out["1"]["relevance_breakdown"]["subfield"] == "kv_cache"
    assert out["1"]["relevance_score"] == 1
    assert out["2"]["relevance_breakdown"]["hard_gate"] is True
    assert out["2"]["relevance_score"] == 0


@pytest.mark.asyncio
async def test_score_papers_records_judge_unavailable_on_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    papers = [_mk("1", "good", "good")]
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in papers]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "p.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.side_effect = Exception("boom")

    await score_papers(in_path, out_path, prompt_path, fake, concurrency=1)
    out = json.loads(out_path.read_text())[0]
    assert out["relevance_breakdown"]["hard_gate"] is True
    assert "judge unavailable" in out["relevance_reason"]


@pytest.mark.asyncio
async def test_score_papers_skips_empty_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    stub = _mk("stub", "", "")
    real = _mk("real", "AWQ quantization", "Weight quantization for LLMs.")
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps([p.model_dump(mode="json") for p in [stub, real]]))
    out_path = tmp_path / "scored.json"
    prompt_path = tmp_path / "p.md"
    prompt_path.write_text("p")

    fake = AsyncMock()
    fake.call_json.return_value = {"hard_gate": False, "subfield": "quantization", "reason": "ok"}

    await score_papers(in_path, out_path, prompt_path, fake, concurrency=2)
    out = {p["id"]: p for p in json.loads(out_path.read_text())}
    assert out["stub"]["relevance_score"] is None
    assert out["real"]["relevance_score"] == 1
    assert fake.call_json.call_count == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd llm-paper-radar && uv run pytest tests/test_venue_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.venue_filter'`

- [ ] **Step 4: Implement `pipeline/venue_filter.py`**

```python
"""LLM relevance/subfield scoring for a one-shot venue batch (not the daily
incremental pipeline — see pipeline/filter.py for that). No prefilter and no
milestone override: this is a single-pass classification over a venue's full
accepted-paper set, not a recurring cron stage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pipeline.llm_client import LLMClient, load_prompt
from sources.base import Paper

_BREAKDOWN_FIELDS = ("hard_gate", "subfield", "reason")


def _hard_gate_result(reason: str) -> dict:
    return {"hard_gate": True, "subfield": "unknown", "reason": reason}


def _composite(hard_gate: bool) -> int:
    return 0 if hard_gate else 1


async def _score_one(paper: Paper, prompt: str, client: LLMClient, sem: asyncio.Semaphore) -> Paper:
    if not paper.title.strip() or not paper.abstract.strip():
        paper.relevance_score = None
        paper.relevance_reason = "skipped: missing title/abstract"
        return paper

    user_msg = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    async with sem:
        try:
            result = await client.call_json(prompt, user_msg, max_tokens=512)
        except Exception as e:
            cause = e
            inner = getattr(e, "last_attempt", None)
            if inner is not None and inner.failed:
                try:
                    cause = inner.exception() or e
                except Exception:
                    pass
            print(f"venue_filter: paper {paper.id} failed: {type(cause).__name__}: {cause}")
            result = _hard_gate_result(f"judge unavailable: {type(cause).__name__}: {cause}"[:200])

    hard_gate = bool(result.get("hard_gate"))
    paper.relevance_score = _composite(hard_gate)
    paper.relevance_reason = str(result.get("reason", ""))
    paper.relevance_breakdown = {k: result.get(k) for k in _BREAKDOWN_FIELDS}
    return paper


async def score_papers(
    in_path: Path,
    out_path: Path,
    prompt_path: Path,
    client: LLMClient,
    concurrency: int = 20,
) -> int:
    raw = await asyncio.to_thread(Path(in_path).read_text)
    papers = [Paper.model_validate(p) for p in json.loads(raw)]
    prompt = load_prompt(prompt_path)
    sem = asyncio.Semaphore(concurrency)
    scored = await asyncio.gather(*[_score_one(p, prompt, client, sem) for p in papers])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps([p.model_dump(mode="json") for p in scored], indent=2)
    await asyncio.to_thread(out_path.write_text, output)
    return len(scored)


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--in-path", required=True, type=click.Path(path_type=Path))
    @click.option("--out-path", required=True, type=click.Path(path_type=Path))
    @click.option("--prompt-path", default="prompts/inference_relevance.md", type=click.Path(path_type=Path))
    @click.option("--model", default="claude-sonnet-4-6")
    @click.option("--concurrency", default=20, type=int)
    def main(in_path: Path, out_path: Path, prompt_path: Path, model: str, concurrency: int):
        client = LLMClient(model=model)
        n = asyncio.run(score_papers(in_path, out_path, prompt_path, client, concurrency))
        print(f"venue_filter: scored {n} papers -> {out_path}")

    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd llm-paper-radar && uv run pytest tests/test_venue_filter.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
cd llm-paper-radar
git add prompts/inference_relevance.md pipeline/venue_filter.py tests/test_venue_filter.py
git commit -m "feat: add LLM inference-deployment relevance/subfield scoring stage"
```

---

## Task 3: Group scored papers by subfield

**Files:**
- Create: `pipeline/venue_group.py`
- Test: `tests/test_venue_group.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_venue_group.py
import json
from pathlib import Path

from pipeline.venue_group import group_by_subfield


def _scored(id_, subfield, hard_gate, title="T", abstract="A", url="https://x"):
    return {
        "id": id_,
        "title": title,
        "abstract": abstract,
        "url": url,
        "relevance_breakdown": {"hard_gate": hard_gate, "subfield": subfield, "reason": "r"},
    }


def test_group_by_subfield_excludes_hard_gated(tmp_path: Path):
    scored = [
        _scored("1", "kv_cache", False),
        _scored("2", "kv_cache", False),
        _scored("3", "quantization", False),
        _scored("4", "unknown", True),
    ]
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(scored))

    groups = group_by_subfield(path)
    assert set(groups) == {"kv_cache", "quantization"}
    assert len(groups["kv_cache"]) == 2
    assert len(groups["quantization"]) == 1


def test_group_by_subfield_defaults_missing_subfield_to_unknown(tmp_path: Path):
    scored = [{"id": "1", "title": "T", "abstract": "A", "url": "https://x",
               "relevance_breakdown": {"hard_gate": False, "reason": "r"}}]
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(scored))

    groups = group_by_subfield(path)
    assert set(groups) == {"unknown"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd llm-paper-radar && uv run pytest tests/test_venue_group.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.venue_group'`

- [ ] **Step 3: Implement `pipeline/venue_group.py`**

```python
"""Pure grouping stage: partitions a venue's scored papers by subfield.
Hard-gated papers are dropped — they never reach the trend-analysis step."""

from __future__ import annotations

import json
from pathlib import Path


def group_by_subfield(scored_path: Path) -> dict[str, list[dict]]:
    papers = json.loads(Path(scored_path).read_text())
    groups: dict[str, list[dict]] = {}
    for p in papers:
        breakdown = p.get("relevance_breakdown") or {}
        if breakdown.get("hard_gate"):
            continue
        subfield = breakdown.get("subfield") or "unknown"
        groups.setdefault(subfield, []).append(p)
    return groups


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--scored-path", required=True, type=click.Path(path_type=Path))
    @click.option("--out-path", required=True, type=click.Path(path_type=Path))
    def main(scored_path: Path, out_path: Path):
        groups = group_by_subfield(scored_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(groups, indent=2))
        summary = ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])))
        print(f"venue_group: {summary} -> {out_path}")

    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-paper-radar && uv run pytest tests/test_venue_group.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd llm-paper-radar
git add pipeline/venue_group.py tests/test_venue_group.py
git commit -m "feat: add subfield grouping stage for venue trend report"
```

---

## Task 4: CLI wrapper chaining fetch → score → group

**Files:**
- Create: `scripts/venue_report.sh`

- [ ] **Step 1: Write `scripts/venue_report.sh`**

```bash
#!/usr/bin/env bash
# Fetch, score, and group a conference's accepted papers for the venue
# trend report. Does NOT run the trend-analysis Workflow — that step needs
# the Workflow tool and is run separately (see docs/superpowers/plans/
# 2026-07-08-mlsys-venue-trend-report.md, Task 5).
set -euo pipefail

VENUE="${1:?usage: venue_report.sh <venue> e.g. MLSys.org/2026/Conference}"
CONF_SLUG="$(echo "$VENUE" | cut -d'.' -f1 | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')-$(echo "$VENUE" | cut -d'/' -f2)"

echo "[1/3] fetching accepted papers for $VENUE"
uv run python -m sources.openreview_venue --venue "$VENUE"

echo "[2/3] scoring papers for LLM inference deployment relevance"
uv run python -m pipeline.venue_filter \
  --in-path "data/raw/${CONF_SLUG}/accepted.json" \
  --out-path "data/scored/${CONF_SLUG}.json"

echo "[3/3] grouping by subfield"
uv run python -m pipeline.venue_group \
  --scored-path "data/scored/${CONF_SLUG}.json" \
  --out-path "data/scored/${CONF_SLUG}-grouped.json"

echo "done: data/scored/${CONF_SLUG}-grouped.json"
```

- [ ] **Step 2: Make it executable**

Run: `cd llm-paper-radar && chmod +x scripts/venue_report.sh`

- [ ] **Step 3: Commit**

```bash
cd llm-paper-radar
git add scripts/venue_report.sh
git commit -m "feat: add CLI wrapper chaining venue fetch/score/group stages"
```

---

## Task 5: Trend-analysis Workflow and report synthesis

**Files:**
- Create: `workflows/venue_trend_report.js`

- [ ] **Step 1: Write `workflows/venue_trend_report.js`**

```javascript
export const meta = {
  name: 'venue-trend-report',
  description: 'Summarize per-subfield research trends for a conference\'s accepted LLM-inference papers and produce a Markdown report',
  phases: [
    { title: 'Trend analysis' },
  ],
}

const TREND_SCHEMA = {
  type: 'object',
  properties: {
    core_problems: { type: 'string' },
    representative_papers: { type: 'array', items: { type: 'string' } },
    method_commonalities: { type: 'string' },
  },
  required: ['core_problems', 'representative_papers', 'method_commonalities'],
}

// args.groups: [{ subfield: string, papers: [{title, abstract, url}] }]
const groups = args.groups

const summaries = await pipeline(groups, (group) => {
  const paperList = group.papers
    .map((p) => `- ${p.title}\n  ${p.abstract}\n  (${p.url})`)
    .join('\n')
  return agent(
    `You are analyzing this conference's accepted papers in the "${group.subfield}" ` +
    `subfield of LLM inference deployment optimization. Papers:\n\n${paperList}\n\n` +
    `Summarize: (1) the core problems this subfield's papers are tackling, ` +
    `(2) 2-4 representative paper titles, (3) commonalities/divergences across their methods.`,
    { label: `trend:${group.subfield}`, phase: 'Trend analysis', schema: TREND_SCHEMA }
  ).then((trend) => ({ subfield: group.subfield, papers: group.papers, trend }))
})

const results = summaries.filter(Boolean).sort((a, b) => b.papers.length - a.papers.length)
log(`Summarized ${results.length}/${groups.length} subfields`)

const distributionTable = results
  .map((r) => `| ${r.subfield} | ${r.papers.length} |`)
  .join('\n')

const sections = results
  .map((r) => {
    const paperLinks = r.papers.map((p) => `- [${p.title}](${p.url})`).join('\n')
    const coreProblems = r.trend?.core_problems ?? '(analysis failed)'
    const reps = r.trend?.representative_papers?.map((t) => `- ${t}`).join('\n') ?? '(analysis failed)'
    const commonalities = r.trend?.method_commonalities ?? '(analysis failed)'
    return (
      `## ${r.subfield} (${r.papers.length} papers)\n\n` +
      `**Core problems:** ${coreProblems}\n\n` +
      `**Representative papers:**\n${reps}\n\n` +
      `**Method commonalities:** ${commonalities}\n\n` +
      `**All papers in this subfield:**\n${paperLinks}\n`
    )
  })
  .join('\n')

const report =
  `# ${args.title ?? 'Conference'} — LLM Inference Deployment Optimization Trend Report\n\n` +
  `## Subfield distribution\n\n| Subfield | # Papers |\n|---|---|\n${distributionTable}\n\n` +
  sections

return { report }
```

- [ ] **Step 2: Commit**

```bash
cd llm-paper-radar
git add workflows/venue_trend_report.js
git commit -m "feat: add trend-analysis Workflow script for venue reports"
```

---

## Task 6: End-to-end manual run and report write-out

This task has no automated test — it is the manual verification pass, run once OpenReview is reachable (not currently, given the intermittent 403s described in the spec's Known Risks).

- [ ] **Step 1: Run the fetch/score/group pipeline**

Run: `cd llm-paper-radar && ./scripts/venue_report.sh "MLSys.org/2026/Conference"`

Expected: `data/scored/mlsys-2026-grouped.json` is written; the script exits non-zero if fewer than 20 accepted papers are found (see Task 1) or if the OpenReview fetch fails after retries — in that case, wait and re-run (the resumable page cache under `data/raw/mlsys-2026/openreview_pages/` means the re-run skips already-fetched pages).

- [ ] **Step 2: Inspect the decision-field assumption**

Run: `python3 -c "import json; d=json.load(open('data/raw/mlsys-2026/openreview_pages/page-0000.json')); print(set(n['content'].get('venueid', {}).get('value') for n in d[:20]))"`

Expected: a small set of distinct `venueid` values. Confirm one of them equals `"MLSys.org/2026/Conference"` exactly (the accepted case) and that rejected/withdrawn notes carry a different suffixed value. If MLSys uses a different convention (e.g. a separate `venue` display string, or a `decision` note under a different invitation instead of `venueid` rewriting), adjust `_is_accepted` in `sources/openreview_venue.py` accordingly, re-run Task 1's tests, and re-run the fetch.

- [ ] **Step 3: Build the `args.groups` payload for the Workflow**

Run:

```bash
python3 -c "
import json
grouped = json.load(open('data/scored/mlsys-2026-grouped.json'))
groups = [
    {
        'subfield': k,
        'papers': [{'title': p['title'], 'abstract': p['abstract'], 'url': p['url']} for p in v],
    }
    for k, v in grouped.items()
]
json.dump(groups, open('data/scored/mlsys-2026-groups-arg.json', 'w'), indent=2)
"
```

- [ ] **Step 4: Invoke the Workflow tool**

Call the `Workflow` tool with `scriptPath: "llm-paper-radar/workflows/venue_trend_report.js"` and `args` set to `{"title": "MLSys 2026", "groups": <contents of data/scored/mlsys-2026-groups-arg.json>}`.

Expected: the workflow returns `{ report: "<markdown string>" }`.

- [ ] **Step 5: Write the report to disk**

Use the `Write` tool to save the returned `report` string to `llm-paper-radar/venue-reports/mlsys-2026.md` (per-conference reports live in `venue-reports/`, not `digests/`, since they're not per-day artifacts).

- [ ] **Step 6: Read the report and sanity-check it**

Confirm: the subfield distribution table lists all groups from `mlsys-2026-grouped.json`, no subfield section is silently missing, and any subfield whose agent failed shows "(analysis failed)" rather than being dropped.

- [ ] **Step 7: Commit the report**

```bash
cd llm-paper-radar
git add venue-reports/mlsys-2026.md
git commit -m "docs: add MLSys 2026 LLM inference deployment trend report"
```

---

## Self-Review Notes

- **Spec coverage:** Fetch/retry/completeness-threshold (Task 1) ↔ spec §2 step 1 + §3; scoring with judge-unavailable fallback (Task 2) ↔ spec §2 step 2 + §3; grouping (Task 3) ↔ spec §2 step 3; parallel trend analysis + synthesis (Task 5) ↔ spec §2 steps 4-5 + §3 (per-subfield failure isolation); manual verification (Task 6) ↔ spec §4 Testing and §5 Known Risks (decision-field assumption check).
- **Type consistency:** `relevance_breakdown` keys (`hard_gate`, `subfield`, `reason`) are identical across `pipeline/venue_filter.py`, `pipeline/venue_group.py`, and the Workflow's expected `args.groups` shape (`subfield`, `papers[].{title,abstract,url}`).
- **No placeholders:** the one open question (MLSys's exact `venueid` convention) is implemented with a concrete, well-known default and an explicit verification+adjustment step (Task 6, Step 2) rather than left as TBD.
