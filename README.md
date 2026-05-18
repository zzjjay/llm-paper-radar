# 📡 LLM Paper Radar

> Daily, automated digest of LLM compression and inference-optimization papers.

A small pipeline that fetches papers from arXiv + HF Daily + Reddit + Semantic Scholar + Twitter (RSSHub), scores each one with Claude Haiku 4.5 against a two-axis rubric (topic relevance × practicality), groups the survivors by topic bucket (PTQ / QAT / KV cache / Speculative decoding / Distillation / Pruning / …), and renders a daily Markdown digest. A single cron job keeps it running.

[Today's digest](#-todays-digest) · [How papers are scored](#-how-papers-are-scored) · [Pipeline](#-pipeline) · [Setup your own radar](#-setup-your-own-radar) · [Repo layout](#-repo-layout)

---

## 📰 Today's digest

> Auto-updated daily at 06:00 local time. Older digests live under [`digests/`](digests/) — see [INDEX.md](INDEX.md).

<!-- LATEST_START -->

# LLM Inference Optimization Daily · 2026-05-18

> 📅 Window: 2026-05-18 (UTC daily)
> 📊 Scanned 30 papers → passed filter 2 → highlighted 2 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Highlights by topic

_Up to 3 PTQ, 2 others per topic — change in [`config.yaml`](config.yaml) under `render.topic_caps`._

### KV cache compression

### 1. GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding (8/10)
**hf_daily** · `2605.15250` · 2026-05-14
👥 Fanxu Meng · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.15250) · [PDF](https://arxiv.org/pdf/2605.15250.pdf)
📡 Sources: hf_daily (💬 1)
🧪 kv_cache · Group-Query Latent Attention (GQLA), low-rank KV compression with dual decoding paths · Llama-3-8B · cal: No retraining needed; TransMLA→TransGQLA conversion from pretrained GQA checkpoint. · perf: Hardware-adaptive: MQA-absorb on H100, GQA+MTP on H20; supports 8-way tensor parallelism.

#### Summary
MLA in DeepSeek-V2/V3 compresses KV cache into low-rank latents but exposes only one decoding path (absorbed MQA), which is optimized for H100 compute-bandwidth ratios and breaks tensor parallelism and Multi-Token Prediction on bandwidth-limited GPUs like H20. GQLA adds a second algebraically equivalent GQA decoding path over the same weights, allowing runtime selection of either MQA-absorb (H100) or GQA+MTP (H20) without retraining or custom kernels, plus up to 8-way tensor parallelism on the GQA path. TransGQLA converts pretrained GQA checkpoints (e.g., LLaMA-3-8B) into GQLA models, achieving 28.125% of the GQA baseline KV cache footprint on the MQA-absorb path while preserving GQA-level memory traffic on the per-group path.

- 🎯 Method: Two algebraically equivalent decoding paths (MQA-absorb / GQA) over one weight set, runtime-selected to match H100 vs H20 rooflines
- 📊 Result: KV cache compressed to 28.125% of GQA baseline on MQA-absorb path for LLaMA-3-8B via TransGQLA conversion
- 💡 Innovation: TransGQLA converts pretrained GQA checkpoints to GQLA without pretraining from scratch, inheriting MLA's low-rank latent compression
- 🔧 Engineering: Supports up to 8-way zero-redundancy tensor parallelism on GQA path; enables Multi-Token Prediction gains on H20 with no custom kernels

---

### Knowledge distillation

### 2. Distilling Long-CoT Reasoning through Collaborative Step-wise Multi-Teacher Decoding (7/10)
**hf_daily** · `2605.02290` · 2026-05-04
👥 Taewon Yun, Jisu Shin, Jeonghwan Choi... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.02290) · [PDF](https://arxiv.org/pdf/2605.02290.pdf)
📡 Sources: hf_daily (👍 22, 💬 1)
🧪 distillation · collaborative multi-teacher step-wise reasoning distillation · cal: multi-teacher decoding with beam search; cost not detailed · perf: no substantial efficiency overhead reported

#### Summary
Distilling Long-CoT reasoning from large reasoning models (LRMs) to smaller students is hampered by existing curation-based methods that select complete traces post-hoc, missing inter-model collaboration and dynamic exploration. CoRD addresses this with a step-wise multi-teacher decoding framework where heterogeneous LRMs collaboratively construct reasoning trajectories via predictive perplexity-based scoring and beam search, maintaining diverse high-potential hypotheses at each step. The resulting distillation data yields near teacher-level student performance with fewer, more structured supervision signals, and generalizes to out-of-domain and open-ended settings.

- 🎯 Method: Step-wise beam search over heterogeneous LRMs scored by predictive perplexity, enabling joint Long-CoT trajectory construction rather than post-hoc trace selection.
- 📊 Result: Achieves near teacher-level student performance with fewer structured supervision signals and no substantial efficiency overhead.
- 💡 Innovation: Replaces single-teacher or post-hoc curation with dynamic multi-teacher collaborative decoding, capturing complementary reasoning across heterogeneous LRMs.
- ⚠️ Limitation: Abstract lacks concrete compression ratios, accuracy deltas, or speedup numbers to quantify improvements precisely.

---

## 📚 Full List (by score, descending)

| # | Title | Score | Topic | Pract | Bucket | Sources | Code | Date |
|---|-------|-------|-------|-------|--------|---------|------|------|
| 1 | [Distilling Long-CoT Reasoning through Collaborative Step-wise Multi-Teacher Decoding](https://arxiv.org/abs/2605.02290) | 7 | 4 | 3 | Knowledge distillation | hf_daily | — | 05-04 |
| 2 | [GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding](https://arxiv.org/abs/2605.15250) | 8 | 4 | 4 | KV cache compression | hf_daily | — | 05-14 |


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

**Option A — host crontab (works behind a corporate Anthropic proxy).** This is what this fork actually uses, because the LLM endpoint sits inside the AMD network and isn't reachable from public CI runners:

```bash
crontab -e
# add (replace path):
0 6 * * * /absolute/path/to/llm-paper-radar/scripts/daily.sh
```

`scripts/daily.sh` sources `~/.bashrc` for the Anthropic env vars, runs the full pipeline (`fetch → dedupe → filter → summarize → render`), and only commits + pushes when something actually changed. Logs land in `scripts/log/YYYY-MM-DD.log`.

**Option B — GitHub Actions (forks with a public sk-ant key).** The `.github/workflows/daily.yml` workflow has the schedule commented out, but everything else is wired up. To use it, set repo secrets and re-enable the schedule line:

| secret | required | what for |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | sk-ant-… key |
| `ANTHROPIC_BASE_URL` | optional | proxy / gateway URL; leave unset for default api.anthropic.com |
| `ANTHROPIC_CUSTOM_HEADERS` | optional | extra headers required by your proxy |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | optional | enable Reddit source |
| `SEMANTIC_SCHOLAR_API_KEY` | optional | enable Semantic Scholar source |
| `RSSHUB_BASE_URL` | optional | enable Twitter source |
| `TEAMS_WEBHOOK_URL` | optional | failure notifications |

```
Settings → Secrets and variables → Actions → New repository secret
```

Then uncomment the `schedule:` block in `.github/workflows/daily.yml`. The workflow runs on the `main` branch and commits as `llm-paper-radar[bot]`.

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
