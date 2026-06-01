# LLM Paper Radar — Design Spec

**Date**: 2026-05-11
**Author**: zhaolin
**Status**: Draft (awaiting review)

## 1. Goal & Non-Goals

### Goal

Build an automated daily pipeline that tracks new papers in **LLM inference optimization** (quantization, compression, KV cache, speculative decoding, inference engines, MoE compression, distillation, edge deployment), filters them by personal relevance using an LLM, generates Chinese + English summaries with highlights, **sorts the surviving papers primarily by community heat (so the genuinely hot ones surface first)**, and publishes a daily digest as Markdown to a public GitHub repo.

The HuggingFace signal explicitly leans on the trending board (`https://huggingface.co/papers/trending`) in addition to the official daily list — trending position is the strongest heat signal HF exposes.

### Non-Goals (v1)

- No real-time push (notifications, Slack/Discord) — daily digest only.
- No web dashboard — Markdown + git is the UI.
- No PDF parsing — abstracts only.
- No coverage of: RLHF / alignment / safety, agents / RAG, multimodal (unless related to compression), training algorithms (unless compression-related).

### Success Criteria

- After 2 weeks of daily runs: at least 80% of papers user actually reads were surfaced by the radar.
- Total monthly cost ≤ $25 (LLM API + optional services).
- Pipeline succeeds on >95% of days; partial-source failure does not block digest generation.

---

## 2. Architecture

Modular pipeline (Approach B from brainstorming):

```
GitHub Actions (cron daily 23:00 UTC)
        │
        ├── Fetch (parallel matrix, continue-on-error per source)
        │     ├─ sources/arxiv.py
        │     ├─ sources/arxiv_authors.py
        │     ├─ sources/hf_daily.py
        │     └─ sources/openreview.py
        │           ↓
        │     data/raw/YYYY-MM-DD/{source}.json
        │
        ├── pipeline/dedupe.py       → data/deduped/YYYY-MM-DD.json
        ├── pipeline/filter.py       → data/scored/YYYY-MM-DD.json   (Claude Sonnet 4.6)
        ├── pipeline/summarize.py    → data/summarized/YYYY-MM-DD.json (Claude Opus 4.7)
        └── pipeline/render.py       → digests/YYYY-MM-DD.md + README.md + INDEX.md
              ↓
        git commit + push
              ↓
        On failure → Teams webhook
```

### Repo Layout

```
llm-paper-radar/
├── .github/workflows/
│   ├── daily.yml
│   ├── weekly.yml
│   └── cleanup.yml
├── sources/
│   ├── __init__.py
│   ├── base.py              # Source ABC + unified Paper schema
│   ├── arxiv.py
│   ├── arxiv_authors.py
│   ├── hf_daily.py
│   └── openreview.py
├── pipeline/
│   ├── dedupe.py
│   ├── filter.py
│   ├── summarize.py
│   └── render.py
├── prompts/
│   ├── relevance.md
│   └── summary.md
├── config.yaml
├── seeds.yaml
├── data/
│   ├── raw/YYYY-MM-DD/      # 7 days, artifact-only
│   ├── deduped/             # in-flight, not committed
│   ├── scored/              # 30 days, artifact-only
│   ├── summarized/YYYY-MM-DD.json   # 90 days, committed
│   └── seen.json            # cross-day dedup set, committed
├── digests/YYYY-MM-DD.md           # permanent, committed
├── weekly/YYYYMMDD-YYYYMMDD.md     # permanent, committed (full table, every gated paper)
├── snapshots/YYYYMMDD.md                 # single-day per-run paper-list snapshot
├── snapshots/YYYYMMDD-YYYYMMDD-Ndays.md  # multi-day rollup snapshot, committed
├── README.md                       # always = latest digest
├── INDEX.md
├── pyproject.toml           # uv-managed
└── .env.example
```

### Design Principles

1. **Source decoupling** — each source implements `fetch() → List[Paper]`. Single-source failure does not block the pipeline (`continue-on-error: true` in matrix).
2. **Traceable data** — every pipeline step persists JSON to `data/`, enabling post-hoc debugging of why a paper was/wasn't surfaced.
3. **Prompt-as-config** — relevance and summary prompts live in `prompts/*.md` and can be evolved independently of code via commits.

---

## 3. Data Sources

Organized into 3 tiers reflecting reliability and information density:

### Tier 1 — Primary (must run daily)

#### `sources/arxiv.py`
- API: `http://export.arxiv.org/api/query` (no key required)
- Categories: `cs.CL`, `cs.LG`, `cs.AR`
- Window: papers with `submittedDate` in last 24 hours
- Rate limit: 3 sec/request, paginate at 200/page
- Retry: shared `arxiv_get_with_retry` helper, 7 attempts with exponential backoff + jitter (~10 min total budget), honors `Retry-After`. Retries 429 / 503 / Timeout / TransportError. Also retries HTTP 200 with an empty Atom feed on page 0 of a **weekday** (Mon–Fri UTC) — arxiv is observed to serve empty feeds in lieu of 429 when throttled, and a real weekday cs.LG/cs.CL/cs.AR batch always has hundreds of submissions. Weekends (Sat/Sun UTC) skip this check: arxiv barely processes new submissions on weekends, so an empty page 0 is the legitimate "nothing posted today" answer. Empty feeds on later pages always remain the normal end-of-pagination signal.
- Expected: ~300–500 papers/day

#### `sources/hf_daily.py`
Fetches **two HF surfaces** and merges them:

1. **Daily papers** — API: `https://huggingface.co/api/daily_papers?date=YYYY-MM-DD`
   - Window: current date's curated list (~10–30 papers/day)
   - Extras: `upvotes`, `num_comments`
2. **Trending board** — HTML scrape of `https://huggingface.co/papers/trending`
   - Extracts arXiv IDs and their rank (1, 2, 3, ...) on the page
   - Top ~30 papers, multi-day rolling window (HF's own algorithm)
   - Extras: `trending_rank` (lower = hotter)

Both lists are normalized to `Paper` objects with `name="hf_daily"` but distinct `extras` fields. A paper appearing in both surfaces gets merged in dedupe (sources list keeps both records). Trending rank is a primary input to the heat score in §6.

### Volume Summary

| Source | Daily Volume | Primary Value |
|--------|-------------|---------------|
| arXiv | 400 | Complete coverage |
| HF Daily | 20 | Community heat signal |
| arxiv_authors (watched groups) | 5–10 | Never-miss for curated authors |
| OpenReview | 5–20 | Pre-arXiv ICLR / ICML / MLSys / AAAI / ACL / EMNLP / NeurIPS submissions (venue templates with `{year}` auto-expand to current + next year; `/-/Submission` only — no reviews / comments / rebuttals) |
| **After dedupe** | **~450–500** | → LLM filter |

---

## 4. Unified Paper Schema & Deduplication

### Paper Schema (Pydantic model in `sources/base.py`)

```python
class SourceRecord(BaseModel):
    name: Literal["arxiv", "arxiv_authors", "hf_daily", "openreview"]
    fetched_at: datetime
    extras: dict = {}            # source-specific (upvotes, score, thread_url, ...)

class Paper(BaseModel):
    id: str                       # arxiv ID primary; DOI or url-hash fallback
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str | None
    published_at: datetime
    primary_category: str
    categories: list[str]
    code_url: str | None = None

    sources: list[SourceRecord]   # accumulated across sources

    # Filled by pipeline stages
    relevance_score: int | None = None        # 0–10
    relevance_reason: str | None = None       # ≤30 chars Chinese
    summary_zh: str | None = None
    highlights_zh: list[str] = []
    summary_en: str | None = None
    highlights_en: list[str] = []
    seen_before: bool = False                 # cross-day dedup flag
```

### Dedupe Logic (`pipeline/dedupe.py`)

**Within-day matching** (priority order):
1. arXiv ID exact match (~95% of cases)
2. DOI exact match (non-arXiv sources)
3. Normalized title match (lowercase + strip punctuation + strip whitespace)

**Field-merge priority** when same paper appears across sources:
```
hf_daily > openreview > arxiv_authors > arxiv (fallback)
```
For each field (title, abstract, authors, etc.), take the value from the highest-priority source that has a non-empty value. arXiv serves as the safety net.

`sources` field is always accumulated (not overwritten) — a paper appearing in 3 sources has 3 records.

`code_url` takes the first non-null value from any source.

### Cross-Day Dedup (Lenient strategy)

- Maintain `data/seen.json`: `Set[paper_id]` of all papers ever surfaced in a digest.
- After within-day dedupe, mark `seen_before = true` for any paper already in the set.
- **Lenient render rule**: papers with `seen_before = true` are re-shown only if `relevance_score` increased by ≥2 since last appearance, OR `heat_score` (defined in §6) increased by ≥50%. Marked with 🔁 in render.
- After successful render, add new IDs to `seen.json` and commit.

---

## 5. LLM Filter & Summarize

### `pipeline/filter.py` — Relevance Scoring

- **Model**: Claude Sonnet 4.6 (`claude-sonnet-4-6`)
- **Input**: title + abstract (~350 input tokens)
- **Output**: JSON `{relevance_score: int, reason: str}`
- **Concurrency**: `asyncio.gather` 50 parallel
- **Prompt caching**: system prompt uses `cache_control: ephemeral` (~30% cost reduction)
- **Threshold**: no numeric gate; every paper with `hard_gate=false` surfaces (per-bucket caps control digest length)

### Relevance Prompt (`prompts/relevance.md`)

```
You are a researcher focused on LLM inference optimization. Your primary interests:
- Model quantization (post-training and QAT, including FP4/FP8/INT4/INT8/binary/ternary, and concrete formats like MXFP4/NVFP4/rocFP4)
- Model pruning and sparsification (including N:M sparsity)
- KV cache compression and management (PagedAttention, Quest, H2O, etc.)
- Speculative decoding and parallel sampling (EAGLE, Medusa, Lookahead)
- LLM inference engines and systems (vLLM, SGLang, llama.cpp, MLX)
- MoE compression and efficient inference
- Knowledge distillation (for compression)
- Edge / mobile LLM deployment
- LoRA / QLoRA / quantization + fine-tuning combinations

Not of interest:
- RLHF, alignment, safety
- Agents / tool use / RAG
- Multimodal (unless it involves compression / quantization)
- Pure training algorithms (unless strongly related to compression)

Given a paper's title and abstract, output:
- relevance_score: integer 0-10
  - 9-10: core new method in quantization / compression / inference optimization
  - 7-8: strongly related (new systems work / hardware adaptation / applied research / relevant benchmark)
  - 4-6: weakly related (related but incremental)
  - 0-3: unrelated
- reason: one short English sentence (≤20 words)

Return JSON only: {"relevance_score": int, "reason": str}
```

### `pipeline/summarize.py` — Summary + Highlights

- **Model**: Claude Opus 4.7 (`claude-opus-4-7`)
- **Input**: title + full abstract + (if present) HF comment excerpts
- **Output**: JSON with bilingual (zh + en) `summary` and `highlights`
- **Concurrency**: 20 parallel
- **Triggered**: every paper with `hard_gate=false` (~50–300/day depending on volume)

### Summary Prompt (`prompts/summary.md`)

```
You are writing a paper summary for an LLM inference optimization researcher (who is familiar with quantization / compression / systems optimization).

Requirements:
1. summary: 3-5 sentence English summary covering (a) what problem it solves (b) core method (c) main results.
   No filler phrases like "this paper proposes a novel method...". Get directly to the technical content.
2. highlights: 2-4 bullet points, each ≤25 words. Must include concrete numbers (compression ratio / speedup / accuracy delta, etc.) when present in the abstract.
   Use emoji prefixes: 🎯 Method / 📊 Result / 💡 Innovation / ⚠️ Limitation / 🔧 Engineering
3. Preserve technical English terms verbatim (e.g. GPTQ, FP8, KV cache). Do not paraphrase well-known names.
4. If the abstract lacks information to support a point, omit it. Do not fabricate.

Return JSON only: {"summary": str, "highlights": list[str]}
```

### Cost Estimate

Filter (Sonnet, ~500 papers/day, with prompt cache) + Summarize (Opus, ~50–300 papers/day, bilingual zh+en). The original Haiku/Sonnet design targeted ~$16/month; Sonnet+Opus is materially more expensive. Re-measure under current pricing before quoting.

---

## 6. Render & Output

### `digests/YYYY-MM-DD.md` Structure

```markdown
# LLM Inference Optimization Daily · 2026-05-11

> 📅 Window: 2026-05-10 00:00 UTC ~ 2026-05-11 00:00 UTC
> 📊 Scanned 487 papers → passed filter 38 (threshold ≥7)
> 💰 LLM cost: $0.52

## 🔥 Top 10 (Full Detail)

### 1. <Title> (10/10)
**<primary_source>** · `<arxiv_id>` · <published_at>
👥 <authors> · 🏷 <categories>
🔗 [arXiv](url) · [PDF](pdf_url) · [GitHub](code_url)
📡 Sources: arxiv, hf_daily (👍 142, 💬 28)

#### Summary
<summary>
- <highlights items>

---
... (entries 2–10) ...

## 📚 Full List (by score, descending)

| # | Title | Score | Sources | Code | Date |
|---|-------|-------|---------|------|------|
| 1 | ... | 10 | arxiv, hf | ✅ | 05-10 |
| 2 | ... | 9 | hf, ss | ✅ | 05-10 |
| ... |
| 38 | ... | 7 | arxiv | — | 05-10 |

## 🔁 Revisited
- [<title> (8/10)](#) — score +2 since last appearance
```

### Sort Order — Heat-Primary

Papers below the relevance threshold (§5) are excluded. Among the survivors, **heat is the primary sort key** so genuinely trending papers surface first; relevance breaks ties:

```
sort_key = (heat_score desc, relevance_score desc)

heat_score = (
    trending_bonus       # max 100, from HF trending rank
    + hf_upvotes
    + star_bonus         # min(log(github_stars+1) * 3, 25)
)

trending_bonus =  100 / trending_rank   if rank ≤ 30 else 0
                  # rank 1 → 100, rank 2 → 50, rank 10 → 10, rank 30 → 3.3
```

This means a paper that is rank-1 on HF trending starts with +100 heat and almost always wins the top slot regardless of how many upvotes the daily-papers endpoint reports. A paper that is only on arXiv with no community signal gets `heat_score = 0` and ranks below any paper with any heat — but is still included in the digest if its `relevance_score >= threshold`.

### `README.md`

Always overwritten with the latest day's digest. Top banner:

```markdown
> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6
```

### `INDEX.md`

Grouped by month, one line per day:

```markdown
# Digest History Index

## 2026 May
- [05-11](digests/2026-05-11.md) — 487 scanned, 38 passed, top: BitNet b1.58
- [05-10](digests/2026-05-10.md) — ...

## 2026 April
- ...
```

### Render Rules

- **Top 10**: full detail block (Chinese + English summary + highlights)
- **11+**: table-only
- No "abstract keywords" column (kept clean)

### Weekly Digest (`weekly/YYYYMMDD-YYYYMMDD.md`)

Produced manually via `uv run python -m pipeline.weekly --end-date YYYY-MM-DD`:
- All papers from the past 7 days that pass the hard gate (no Top-N cap)
- Same compact `# | Bucket | Paper | Authors | Date | Why` table as README
- Why-column links resolve to `../digests/<date>.md#p-<id>`
- Filename = `<start>-<end>.md` (7 days inclusive, no week-number cadence)

---

## 7. GitHub Actions Workflows

### `.github/workflows/daily.yml`

```yaml
name: Daily LLM Paper Radar

on:
  schedule:
    - cron: '0 23 * * *'    # UTC 23:00 = Beijing 07:00 next day
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
        source: [arxiv, arxiv_authors, hf_daily, openreview]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - name: Fetch ${{ matrix.source }}
        run: uv run python -m sources.${{ matrix.source }} --backfill-days ${{ inputs.backfill_days || 0 }}
        continue-on-error: true
      - uses: actions/upload-artifact@v4
        with:
          name: raw-${{ matrix.source }}
          path: data/raw/
          retention-days: 7

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
      - run: uv run python -m pipeline.dedupe
      - run: uv run python -m pipeline.filter
        env: { ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }} }
      - run: uv run python -m pipeline.summarize
        env: { ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }} }
      - run: uv run python -m pipeline.render
      - name: Upload debug artifacts (deduped + scored)
        uses: actions/upload-artifact@v4
        with:
          name: pipeline-debug
          path: |
            data/deduped/
            data/scored/
          retention-days: 14
      - name: Commit & push
        run: |
          git config user.name "llm-paper-radar[bot]"
          git config user.email "actions@github.com"
          git add digests/ weekly/ README.md INDEX.md data/seen.json data/summarized/
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
              ],
              "markdown": true
            }],
            "potentialAction": [{
              "@type": "OpenUri",
              "name": "View run",
              "targets": [{"os": "default", "uri": "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}]
            }]
          }' "${{ secrets.TEAMS_WEBHOOK_URL }}"
```

### `.github/workflows/weekly.yml`

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
          git add weekly/ INDEX.md
          git diff --cached --quiet || git commit -m "📅 Weekly digest $(date -u +%Y-W%U)"
          git push
```

### `.github/workflows/cleanup.yml`

```yaml
name: Cleanup old data

on:
  schedule:
    - cron: '0 22 * * 0'   # Sundays UTC 22:00

permissions:
  contents: write

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Remove summarized data older than 90 days
        run: |
          find data/summarized/ -type f -mtime +90 -delete
      - name: Commit
        run: |
          git config user.name "llm-paper-radar[bot]"
          git config user.email "actions@github.com"
          git add -A
          git diff --cached --quiet || git commit -m "🧹 Cleanup old summarized data"
          git push
```

### Secrets

| Secret | Required | Purpose |
|--------|----------|---------|
| `ANTHROPIC_API_KEY` | yes | Claude API |
| `ANTHROPIC_BASE_URL` | optional | proxy / gateway URL |
| `ANTHROPIC_CUSTOM_HEADERS` | optional | extra headers for proxy |
| `TEAMS_WEBHOOK_URL` | optional | Failure alerts |

### Pipeline Timing

| Stage | Duration |
|-------|----------|
| Fetch (parallel matrix) | 2–3 min |
| Dedupe | <30 sec |
| Filter (Sonnet, 50 concurrency) | 1–2 min |
| Summarize (Opus, 20 concurrency) | 2–3 min |
| Render + commit | <30 sec |
| **Total** | **~6–9 min/day** |

Public repo on GitHub Actions: free, unlimited minutes.

---

## 8. Configuration

### `config.yaml`

```yaml
sources:
  arxiv:
    enabled: true
    categories: [cs.CL, cs.LG, cs.AR]
  hf_daily:
    enabled: true

filter:
  model: claude-sonnet-4-6
  concurrency: 50

summarize:
  model: claude-opus-4-7
  concurrency: 20

render:
  full_top_n: 10
  truncate_after: 10

dedupe:
  cross_day_strategy: lenient
  source_priority:
    - hf_daily
    - openreview
    - arxiv_authors
    - arxiv
```

### `seeds.yaml` (curated index of important papers per bucket; consumed by the paper-triage skill)

```yaml
seeds:
  - id: arXiv:2210.17323     # GPTQ
    name: GPTQ
  - id: arXiv:2306.00978     # AWQ
    name: AWQ
  - id: arXiv:2211.10438     # SmoothQuant
    name: SmoothQuant
  - id: arXiv:2404.00456     # QuaRot
    name: QuaRot
  - id: arXiv:2504.19874     # TurboQuant
    name: TurboQuant
  - id: arXiv:2512.02010     # FourOverSix (4over6)
    name: FourOverSix
  - id: arXiv:2407.11062     # EfficientQAT
    name: EfficientQAT
  - id: arXiv:2305.14314     # QLoRA
    name: QLoRA
```

---

## 9. Storage & Retention

| Path | Retention | Committed? |
|------|-----------|-----------|
| `digests/YYYY-MM-DD.md` | permanent | yes |
| `weekly/YYYYMMDD-YYYYMMDD.md` | permanent | yes |
| `snapshots/YYYYMMDD.md` (single-day) / `snapshots/YYYYMMDD-YYYYMMDD-Ndays.md` (multi-day) | permanent | yes |
| `README.md` | always overwritten with latest | yes |
| `INDEX.md` | permanent (grows over time) | yes |
| `data/seen.json` | permanent (~50KB/year) | yes |
| `data/summarized/YYYY-MM-DD.json` | 90 days (sliding via cleanup workflow) | yes |
| `data/scored/YYYY-MM-DD.json` | 14 days | artifact only (`pipeline-debug`) |
| `data/deduped/YYYY-MM-DD.json` | 14 days | artifact only (`pipeline-debug`) |
| `data/raw/YYYY-MM-DD/*.json` | 7 days | artifact only (per-source) |

**Repo size projection**: <50MB after 3–5 years.

---

## 10. First-Time Deployment

```bash
# 1. Create repo
gh repo create zhaolin-amd/llm-paper-radar --public \
  --description "Daily LLM inference optimization paper digest"
cd llm-paper-radar

# 2. (After implementation) push code

# 3. Configure secrets
gh secret set ANTHROPIC_API_KEY
gh secret set TEAMS_WEBHOOK_URL        # optional, for failure alerts

# 4. Edit config.yaml + seeds.yaml + prompts/*.md as needed

# 5. Trigger first run with backfill
gh workflow run daily.yml -f backfill_days=2

# 6. Watch
gh run watch
```

### Required External Setup (User-side)

1. **Anthropic API key** — at console.anthropic.com
2. **Microsoft Teams incoming webhook** (optional) — channel settings → Connectors → Incoming Webhook

---

## 11. Known Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Anthropic API rate limit | Low | 50 concurrency well within tier limits; backoff in client |
| arXiv API outage | Low | Other Tier 1 source (HF Daily) still works |
| arXiv throttle returning empty feeds instead of 429 | Medium | `sources/arxiv.py` page-0 empty-feed validator promotes "200 + 0 entries" to a retryable soft-failure inside the same 10-min backoff budget — only on weekdays (Sat/Sun are skipped because arxiv legitimately publishes ~0 papers then); persistent empties raise so the day is skipped instead of writing 0 papers |
| LLM hallucinates relevance | Medium | Threshold + reason field allows audit; can adjust prompt |
| Repo grows unbounded | Low | Sliding retention + cleanup workflow |
| Cost overrun (>$25/mo) | Low | Hard concurrency limits; can tighten threshold to ≥8 |

---

## 12. Open Items for User

Resolved:
- ✅ TurboQuant arXiv ID: `2504.19874`
- ✅ FourOverSix (4over6) arXiv ID: `2512.02010`
- ✅ GitHub account: `zhaolin-amd`

Outstanding (deployment-time, not blocking design):
- Microsoft Teams incoming webhook URL — optional, for failure alerts

---

## 13. Out of Scope (v2 candidates)

- Personal feedback loop: thumbs-up/down per paper → fine-tune relevance prompt
- PDF parsing for deeper summaries (when abstract is uninformative)
- Web dashboard or HF Space frontend
- Slack/Discord push for top-tier papers
- Multi-user support / personalized digests
