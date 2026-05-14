# 📡 LLM Paper Radar

> Daily, automated digest of LLM compression and inference-optimization papers.

A small pipeline that fetches papers from arXiv + HF Daily + Reddit + Semantic Scholar + Twitter (RSSHub), scores each one with Claude Haiku 4.5 against a two-axis rubric (topic relevance × practicality), groups the survivors by topic bucket (PTQ / QAT / KV cache / Speculative decoding / Distillation / Pruning / …), and renders a daily Markdown digest. A single cron job keeps it running.

[Today's digest](#-todays-digest) · [How papers are scored](#-how-papers-are-scored) · [Pipeline](#-pipeline) · [Setup your own radar](#-setup-your-own-radar) · [Repo layout](#-repo-layout)

---

## 📰 Today's digest

> Auto-updated daily at 06:00 local time. Older digests live under [`digests/`](digests/) — see [INDEX.md](INDEX.md).

<!-- LATEST_START -->

# LLM Inference Optimization Daily · 2026-05-14

> 📅 Window: 2026-05-14 (UTC daily)
> 📊 Scanned 358 papers → passed filter 0 → highlighted 0 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Highlights by topic

## 📚 Full List (by score, descending)

| # | Title | Score | Topic | Pract | Bucket | Sources | Code | Date |
|---|-------|-------|-------|-------|--------|---------|------|------|


<!-- LATEST_END -->

---

## 🧮 How papers are scored

Every paper goes through Claude Haiku 4.5 with [`prompts/relevance.md`](prompts/relevance.md). The model returns a structured JSON breakdown; the orchestrator combines it into a 0–10 composite score.

### Two axes

| axis | range | what it captures |
|---|---|---|
| `topic_relevance` | 0–5 | How squarely the paper sits in LLM compression. 5 = core PTQ/QAT/pruning/etc. with proper accuracy benchmarks. 3 = compression-adjacent (sparse attention coupled with quant, comprehensive survey). 0 = unrelated. |
| `practicality` | 0–5 | Algorithm simplicity + clear inference perf impact + low calibration cost + small GPU memory footprint. 5 = AWQ-style "few lines, big speedup, data-free." 0 = complex, no perf benefit, intractable calibration. |

`relevance_score = topic_relevance + practicality` (so 0–10). A paper passes the highlight gate at **score ≥ 7**.

### Hard gate

`hard_gate = true` zeros both axes. Triggered when the paper is unambiguously off-topic — RAG, agents, alignment, multimodal-without-compression, pure training algorithms, or models smaller than 1B parameters tested on toys (BERT-base, GPT-2-small).

### Topic buckets and per-bucket caps

Surviving papers are grouped by `topic_bucket`. The digest shows at most:

| bucket | cap |
|---|---|
| **PTQ** | 3 |
| **QAT / low-bit pretraining** | 2 |
| **KV cache compression** | 2 |
| **Speculative decoding** | 2 |
| **Knowledge distillation** | 2 |
| **Pruning / sparsity** | 2 |
| Diffusion compression | 2 |
| Survey | 2 |
| Other | 2 |

PTQ gets the highest cap because the team's focus is on quantization recipes and PTQ tends to dominate the daily volume. Caps are set in [`config.yaml`](config.yaml) under `render.topic_caps` and can be overridden without touching code.

The full ranked table (no caps) lives below the highlights so nothing gets lost — the caps only control what bubbles up to the "🔥 Highlights by topic" section.

---

## 🛠 Pipeline

```
   ┌────────────┐     fetchers (one per source)
   │  sources   │ ─── arxiv + hf_daily + reddit + semantic_scholar + twitter_rsshub
   └─────┬──────┘            ↓
         │            data/raw/YYYY-MM-DD/{source}.json
         ↓
   ┌────────────┐     pipeline/dedupe.py
   │   dedupe   │ ─── merge by arXiv id, keep all source attributions
   └─────┬──────┘            ↓
         │            data/deduped/YYYY-MM-DD.json
         ↓
   ┌────────────┐     pipeline/filter.py     (Claude Haiku 4.5)
   │   filter   │ ─── two-axis rubric → composite 0-10 + relevance_breakdown
   └─────┬──────┘            ↓
         │            data/scored/YYYY-MM-DD.json
         ↓
   ┌────────────┐     pipeline/summarize.py  (Claude Sonnet 4.6)
   │ summarize  │ ─── only papers ≥ threshold get an English summary + highlights
   └─────┬──────┘            ↓
         │            data/summarized/YYYY-MM-DD.json
         ↓
   ┌────────────┐     pipeline/render.py
   │   render   │ ─── group by bucket, apply caps, splice into README + digest
   └────────────┘            ↓
                      digests/YYYY-MM-DD.md  +  README.md  +  INDEX.md
```

Each stage is independently runnable from the CLI:

```bash
uv run python -m sources.hf_daily   --backfill-days 0
uv run python -m sources.arxiv      --backfill-days 0
uv run python -m pipeline.dedupe    --backfill-days 0
uv run python -m pipeline.filter    --backfill-days 0
uv run python -m pipeline.summarize --backfill-days 0
uv run python -m pipeline.render    --backfill-days 0
```

`scripts/daily.sh` chains all of these, then `git commit && git push` if anything changed.

---

## 🚀 Setup your own radar

### 1. Fork & clone

```bash
gh repo fork zhaolin-amd/llm-paper-radar --clone
cd llm-paper-radar
uv sync                       # installs deps from pyproject.toml + uv.lock
```

### 2. Configure access to Claude

The pipeline calls Anthropic via the official SDK. You can use either path:

- **Anthropic API directly:** set `ANTHROPIC_API_KEY` to your key and unset `ANTHROPIC_BASE_URL`.
- **Custom proxy / gateway:** set `ANTHROPIC_BASE_URL` and any required `ANTHROPIC_CUSTOM_HEADERS` (e.g. enterprise subscription header). `ANTHROPIC_API_KEY` can stay as a placeholder.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or, for a proxy:
export ANTHROPIC_BASE_URL="https://your-proxy.example.com/Anthropic"
export ANTHROPIC_CUSTOM_HEADERS="Subscription-Key: ..."
```

### 3. (Optional) Add other source credentials

Sources without credentials will silently produce 0 papers. To enable:

| source | env vars / config |
|---|---|
| Reddit (LocalLLaMA) | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` |
| Semantic Scholar | `SEMANTIC_SCHOLAR_API_KEY` + seed papers in [`seeds.yaml`](seeds.yaml) |
| Twitter (via RSSHub) | `RSSHUB_BASE_URL` pointing at a self-hosted RSSHub |

`hf_daily` and `arxiv` work without credentials.

### 4. Tune the prompt and topic caps

- Edit [`prompts/relevance.md`](prompts/relevance.md) to bias the rubric toward your team's focus areas (e.g. swap "compression" for "RL" or "robotics").
- Adjust [`config.yaml`](config.yaml) — categories, threshold, per-bucket caps, source priority for dedup tie-breaks.

### 5. Smoke-test the chain

```bash
./scripts/daily.sh             # logs to scripts/log/YYYY-MM-DD.log
```

If everything is wired up, you'll see `data/raw/`, `data/deduped/`, `data/scored/`, `data/summarized/` populate, then a fresh `digests/YYYY-MM-DD.md` plus an updated `README.md`.

### 6. Schedule it

This repo ships a GitHub Actions workflow at `.github/workflows/daily.yml` that runs the pipeline daily at **12:00 UTC** and pushes the digest. To use it, set repo secrets:

| secret | required | what for |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | sk-ant-… key, OR a placeholder like `dummy` if you're routing through a proxy |
| `ANTHROPIC_BASE_URL` | optional | proxy / gateway URL (e.g. `https://your-proxy/Anthropic`); leave unset for default api.anthropic.com |
| `ANTHROPIC_CUSTOM_HEADERS` | optional | extra headers required by your proxy (e.g. `Subscription-Key: …`) |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | optional | enable Reddit source |
| `SEMANTIC_SCHOLAR_API_KEY` | optional | enable Semantic Scholar source |
| `RSSHUB_BASE_URL` | optional | enable Twitter source |
| `TEAMS_WEBHOOK_URL` | optional | failure notifications |

```
Settings → Secrets and variables → Actions → New repository secret
```

The workflow re-fetches today's papers, runs the full pipeline, and commits to `main` as `llm-paper-radar[bot]`.

If you'd rather run it on your own machine (e.g. behind a corporate Anthropic proxy that can't be exposed to GitHub):

```bash
crontab -e
# add (replace path):
0 6 * * * /absolute/path/to/llm-paper-radar/scripts/daily.sh
```

`scripts/daily.sh` is idempotent: it `git pull --rebase`s first, runs the pipeline, and only commits if there are real changes.

---

## 🗂 Repo layout

```
llm-paper-radar/
├── README.md                    # this file (LATEST_START/END auto-updated)
├── INDEX.md                     # one-line per past digest, newest first
├── config.yaml                  # source toggles, models, threshold, topic_caps
├── seeds.yaml                   # Semantic Scholar seed papers
├── prompts/
│   ├── relevance.md             # filter rubric (two-axis + buckets + anchors)
│   └── summarize.md             # summary format prompt
├── sources/                     # one fetcher per upstream
│   ├── arxiv.py
│   ├── hf_daily.py
│   ├── reddit.py
│   ├── semantic_scholar.py
│   └── twitter_rsshub.py
├── pipeline/
│   ├── config.py                # Pydantic config model
│   ├── llm_client.py            # async Anthropic wrapper with prompt cache
│   ├── dedupe.py
│   ├── filter.py                # two-axis scoring
│   ├── summarize.py
│   ├── render.py                # bucket grouping + README splicing
│   ├── readme_template.md       # static doc template (this file's source)
│   └── weekly.py                # Monday weekly roll-up
├── scripts/
│   └── daily.sh                 # cron entrypoint: fetch → ... → push
├── digests/
│   └── YYYY-MM-DD.md            # daily digest archive
├── weekly/
│   └── YYYY-Www.md              # weekly digest archive
├── data/                        # mostly gitignored; seen.json + summarized/ kept
│   ├── raw/                     # gitignored
│   ├── deduped/                 # gitignored
│   ├── scored/                  # gitignored
│   ├── summarized/              # tracked
│   └── seen.json                # tracked: papers seen across days for 🔁 marker
├── tests/
└── pyproject.toml + uv.lock
```

---

## 📜 License

MIT.
