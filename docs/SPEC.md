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
- No X/Twitter for v1 unless RSSHub instance is available (kept as conditional source).

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
        │     ├─ sources/hf_daily.py
        │     ├─ sources/reddit.py
        │     ├─ sources/semantic_scholar.py
        │     ├─ sources/papers_with_code.py
        │     └─ sources/twitter_rsshub.py
        │           ↓
        │     data/raw/YYYY-MM-DD/{source}.json
        │
        ├── pipeline/dedupe.py       → data/deduped/YYYY-MM-DD.json
        ├── pipeline/filter.py       → data/scored/YYYY-MM-DD.json   (Claude Haiku 4.5)
        ├── pipeline/summarize.py    → data/summarized/YYYY-MM-DD.json (Claude Sonnet 4.6)
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
│   ├── hf_daily.py
│   ├── reddit.py
│   ├── semantic_scholar.py
│   ├── papers_with_code.py
│   └── twitter_rsshub.py
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
├── digests/YYYY-MM-DD.md    # permanent, committed
├── weekly/YYYY-Www.md       # permanent, committed
├── README.md                # always = latest digest
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

### Tier 2 — Secondary (also daily)

#### `sources/reddit.py`
- API: `https://oauth.reddit.com/r/LocalLLaMA/top.json?t=day&limit=50`
- Auth: OAuth via `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`
- Logic: filter posts containing `arxiv.org/abs/XXXX.XXXXX`; extract arXiv IDs via regex; refetch full metadata from arXiv API; record `reddit_score`, `reddit_comments_count`, `thread_url` in extras
- Expected: 5–15 papers/day with arXiv links

#### `sources/semantic_scholar.py`
- API: `https://api.semanticscholar.org/graph/v1/paper/{seed_id}/citations`
- Config: `seeds.yaml` with ~20 foundational papers
- Window: last 7 days of new citations per seed
- Rate limit: 1 req/sec without key, higher with `SEMANTIC_SCHOLAR_API_KEY`
- Expected: 20–50 papers/day

### Tier 3 — Supplementary

#### `sources/papers_with_code.py`
- Source: RSS feed `https://paperswithcode.com/latest/rss.xml` (PwC search API is unstable)
- Extras retained: `code_url` (highlighted in render)
- Expected: ~50–100 papers/day (high overlap with arXiv, dedupe handles)

#### `sources/twitter_rsshub.py` (conditional, depends on self-hosted RSSHub)
- URL: `${RSSHUB_BASE_URL}/twitter/user/<account>` per account
- Accounts (from config.yaml): `_akhaliq`, `Tim_Dettmers`, `HanLab_MIT`, `vllm_project`, `danielhanchen`, `tri_dao`, `omarsar0`
- Logic: same as Reddit — filter for arXiv links, extract IDs, merge
- **Known fragility**: RSSHub Twitter route is brittle (X frequently bans IPs). Source skips silently when unreachable. Manual monthly health check recommended.

### Volume Summary

| Source | Daily Volume | Primary Value |
|--------|-------------|---------------|
| arXiv | 400 | Complete coverage |
| HF Daily | 20 | Community heat signal |
| r/LocalLLaMA | 5–15 | Industry adoption signal |
| Semantic Scholar | 30 | Citation graph (covers arXiv blind spots) |
| PwC | 80 | Code availability tag |
| Twitter (RSSHub) | 5–20 | Author commentary signal |
| **After dedupe** | **~450–500** | → LLM filter |

---

## 4. Unified Paper Schema & Deduplication

### Paper Schema (Pydantic model in `sources/base.py`)

```python
class SourceRecord(BaseModel):
    name: Literal["arxiv", "hf_daily", "reddit", "semantic_scholar", "papers_with_code", "twitter_rsshub"]
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
hf_daily > reddit > semantic_scholar > papers_with_code > twitter_rsshub > arxiv (fallback)
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

- **Model**: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
- **Input**: title + abstract (~350 input tokens)
- **Output**: JSON `{relevance_score: int, reason: str}`
- **Concurrency**: `asyncio.gather` 50 parallel
- **Prompt caching**: system prompt uses `cache_control: ephemeral` (~30% cost reduction)
- **Threshold**: `relevance_score >= 7` proceeds to summarize

### Relevance Prompt (`prompts/relevance.md`)

```
You are a researcher focused on LLM inference optimization. Your primary interests:
- Model quantization (post-training and QAT, including FP4/FP8/INT4/INT8/binary/ternary, and concrete formats like MXFP4/NVFP4/rocFP4)
- Model pruning and sparsification (including N:M sparsity)
- KV cache compression and management (PagedAttention, Quest, H2O, etc.)
- Speculative decoding and parallel sampling (EAGLE, Medusa, Lookahead)
- LLM inference engines and systems (vLLM, SGLang, TensorRT-LLM, llama.cpp, MLX)
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

- **Model**: Claude Sonnet 4.6 (`claude-sonnet-4-6`)
- **Input**: title + full abstract + (if present) HF/Reddit comment excerpts
- **Output**: JSON with `summary` and `highlights` (English only)
- **Concurrency**: 20 parallel
- **Triggered**: only on papers with `relevance_score >= 7` (~30–50/day)

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

| Stage | Daily Cost | Monthly |
|-------|-----------|---------|
| Filter (Haiku, ~500 papers, with cache) | $0.25 | ~$7.5 |
| Summarize (Sonnet, ~50 papers, English only) | $0.30 | ~$9 |
| **Total** | **~$0.55** | **~$16** |

(Range $12–20 depending on daily volume.)

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
📡 Sources: arxiv, hf_daily (👍 142, 💬 28), reddit (🔥 score 580)

#### Summary
<summary>
- <highlights items>

---
... (entries 2–10) ...

## 📚 Full List (by score, descending)

| # | Title | Score | Sources | Code | Date |
|---|-------|-------|---------|------|------|
| 1 | ... | 10 | arxiv, hf, reddit | ✅ | 05-10 |
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
    + log(reddit_score + 1) * 5
    + twitter_account_bonus  # 10 per distinct account that linked it
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

### Weekly Digest (`weekly/YYYY-Www.md`)

Generated by `weekly.yml` every Monday 23:00 UTC:
- Top 20 papers from past 7 days (re-ranked by aggregate score + community signal)
- Trending keywords (extracted from week's titles via TF-IDF over a baseline corpus)
- Per-source contribution counts
- Same render style as daily but condensed

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
| `REDDIT_CLIENT_ID` | yes | Reddit OAuth |
| `REDDIT_CLIENT_SECRET` | yes | Reddit OAuth |
| `TEAMS_WEBHOOK_URL` | yes | Failure alerts |
| `RSSHUB_BASE_URL` | optional | If absent, twitter source disabled |
| `SEMANTIC_SCHOLAR_API_KEY` | optional | Higher rate limit |

### Pipeline Timing

| Stage | Duration |
|-------|----------|
| Fetch (parallel matrix) | 2–3 min |
| Dedupe | <30 sec |
| Filter (Haiku, 50 concurrency) | 1–2 min |
| Summarize (Sonnet, 20 concurrency) | 2–3 min |
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
  reddit:
    enabled: true
    subreddit: LocalLLaMA
    top_window: day
  semantic_scholar:
    enabled: true
    seeds_file: seeds.yaml
    citation_window_days: 7
  papers_with_code:
    enabled: true
  twitter_rsshub:
    enabled: true
    accounts:
      - _akhaliq
      - Tim_Dettmers
      - HanLab_MIT
      - vllm_project
      - danielhanchen
      - tri_dao
      - omarsar0

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
  source_priority:
    - hf_daily
    - reddit
    - semantic_scholar
    - papers_with_code
    - twitter_rsshub
    - arxiv
```

### `seeds.yaml` (Semantic Scholar seed papers)

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
| `weekly/YYYY-Www.md` | permanent | yes |
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
gh secret set REDDIT_CLIENT_ID
gh secret set REDDIT_CLIENT_SECRET
gh secret set TEAMS_WEBHOOK_URL
gh secret set RSSHUB_BASE_URL          # optional
gh secret set SEMANTIC_SCHOLAR_API_KEY # optional

# 4. Edit config.yaml + seeds.yaml + prompts/*.md as needed

# 5. Trigger first run with backfill
gh workflow run daily.yml -f backfill_days=2

# 6. Watch
gh run watch
```

### Required External Setup (User-side)

1. **Reddit OAuth app** — create at https://www.reddit.com/prefs/apps (script type, free)
2. **Anthropic API key** — at console.anthropic.com
3. **Microsoft Teams incoming webhook** — channel settings → Connectors → Incoming Webhook
4. **RSSHub instance** (optional, for Twitter source) — Docker on a VPS or your server

---

## 11. Known Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| RSSHub Twitter route gets banned | High | Source skips silently; monthly health check reminder |
| Anthropic API rate limit | Low | 50 concurrency well within tier limits; backoff in client |
| arXiv API outage | Low | Other Tier 1 source (HF Daily) still works |
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
- Microsoft Teams incoming webhook URL — needed before first run
- RSSHub base URL — if absent, twitter source auto-disables; can be added later

---

## 13. Out of Scope (v2 candidates)

- Personal feedback loop: thumbs-up/down per paper → fine-tune relevance prompt
- Switch to SocialData.tools / TwitterAPI.io if RSSHub proves too unstable
- PDF parsing for deeper summaries (when abstract is uninformative)
- Web dashboard or HF Space frontend
- Slack/Discord push for top-tier papers
- Multi-user support / personalized digests
