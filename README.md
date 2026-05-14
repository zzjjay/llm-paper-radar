# 📡 LLM Paper Radar

> Daily, automated digest of LLM compression and inference-optimization papers.

A small pipeline that fetches papers from arXiv + HF Daily + Reddit + Semantic Scholar + Twitter (RSSHub), scores each one with Claude Haiku 4.5 against a two-axis rubric (topic relevance × practicality), groups the survivors by topic bucket (PTQ / QAT / KV cache / Speculative decoding / Distillation / Pruning / …), and renders a daily Markdown digest. A single cron job keeps it running.

[Today's digest](#-todays-digest) · [How papers are scored](#-how-papers-are-scored) · [Pipeline](#-pipeline) · [Setup your own radar](#-setup-your-own-radar) · [Repo layout](#-repo-layout)

---

## 📰 Today's digest

> Auto-updated daily at 06:00 local time. Older digests live under [`digests/`](digests/) — see [INDEX.md](INDEX.md).

<!-- LATEST_START -->

# LLM Inference Optimization Daily · 2026-05-13

> 📅 Window: 2026-05-13 (UTC daily)
> 📊 Scanned 394 papers → passed filter 8 → highlighted 6 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Highlights by topic

### PTQ (post-training quantization) (top 3 of cap 3)

### 1. Search Your Block Floating Point Scales! (9/10) 🔁
**arxiv** · `2605.12464` · 2026-05-12
👥 Tanmaey Gupta, Hayden Prairie, Xiaoxia Wu... · 🏷 cs.LG, cs.AR, cs.PF
🔗 [arXiv](https://arxiv.org/abs/2605.12464) · [PDF](https://arxiv.org/pdf/2605.12464v1)
📡 Sources: arxiv
🧪 ptq · NVFP4 BFP with fine-grained scale search · Llama-3.1-70B · cal: fine-grained search on calibration data; cost not explicitly stated · perf: near-0 perf loss for causal language modeling with ScaleSearchAttention

#### Summary
Standard Block Floating Point (BFP) quantization uses the block maximum as the scale factor, which is suboptimal for minimizing quantization error. ScaleSearch replaces this fixed rule with a fine-grained search over the mantissa bits of microscaling formats (e.g., NVFP4) to find the scale that minimizes quantization error for the actual data distribution. Applied to PTQ and attention kernels, ScaleSearch reduces NVFP4 quantization error by 27%, improves MATH500 accuracy by up to 15 points on Qwen3-8B, and improves Wikitext-2 PPL by up to 0.77 points on Llama 3.1 70B via ScaleSearchAttention.

- 🎯 Method: Fine-grained scale search over microscaling BFP mantissa bits to minimize quantization error, replacing fixed max-magnitude scale selection.
- 📊 Result: 27% reduction in NVFP4 quantization error; up to 15-point MATH500 improvement on Qwen3-8B PTQ.
- 📊 Result: ScaleSearchAttention improves Wikitext-2 PPL by up to 0.77 points on Llama 3.1 70B with near-zero performance loss.
- 💡 Innovation: NVFP4-based attention algorithm integrating ScaleSearch, compatible with existing PTQ pipelines and low-precision attention methods.

---

### 2. Grid Games: The Power of Multiple Grids for Quantizing Large Language Models (9/10) 🔁
**arxiv** · `2605.12327` · 2026-05-12
👥 Vage Egiazarian, Erik Schultheis, Andrei Panferov... · 🏷 cs.LG
🔗 [arXiv](https://arxiv.org/abs/2605.12327) · [PDF](https://arxiv.org/pdf/2605.12327v1)
📡 Sources: arxiv
🧪 ptq · MXFP4/NVFP4 with multiple adaptive grids (PO2) · Llama-like models (size unknown from abstract)

#### Summary
The paper addresses accuracy loss in microscaled 4-bit quantization (MXFP4, NVFP4) by extending the single fixed floating-point grid paradigm to multiple selectable grids per group, where one or more bits in the scale value encode grid selection. The authors formalize the power-of-two-grids (PO2) problem, prove theoretically that small-group formats benefit significantly from multiple grids while large groups see diminishing returns, and instantiate four concrete grid families: PO2(NF4) pairing NF4 with a learned grid, MPO2 fully learned from weights/activations, PO2(Split87) an asymmetric explicit-zero grid, and SFP4 a TensorCore-compatible triple pairing NVFP4 with two shifted variants. Post-training quantization experiments on standard open models and pre-training of Llama-like models demonstrate consistent accuracy improvements over single-grid FP4 in both weight-only and weight+activation quantization settings.

- 🎯 Method: Multiple 4-bit grids selected per group via scale bits, formalizing the PO2 problem to extend MXFP4/NVFP4 microscaling formats.
- 💡 Innovation: SFP4 is TensorCore-implementable, pairing NVFP4 with two shifted variants; MPO2 grids are fully learned from real weight/activation distributions.
- 📊 Result: Adaptive multi-grid FP4 consistently outperforms single-grid FP4 in both weight-only and weight+activation quantization across PTQ and pre-training.
- ⚠️ Limitation: Theoretical analysis shows the multi-grid advantage vanishes for very large group sizes.

---

### 3. SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization (9/10) 🔁
**arxiv** · `2605.12245` · 2026-05-12
👥 Chengzhu Bao, Xianglong Yan, Zhiteng Li... · 🏷 cs.LG
🔗 [arXiv](https://arxiv.org/abs/2605.12245) · [PDF](https://arxiv.org/pdf/2605.12245v1)
📡 Sources: arxiv
🧪 ptq · NVFP4 PTQ with joint scale optimization · cal: Post-training only; calibration cost unknown · perf: Native hardware support; no additional overhead reported

#### Summary
SOAR addresses suboptimal accuracy in NVFP4 quantization of LLMs caused by inflexible scale selection and coupled treatment of quantization/dequantization scales. The method introduces Closed-form Joint Scale Optimization (CJSO), which analytically co-optimizes global and block-wise scales by minimizing reconstruction error, and Decoupled Scale Search (DSS), which separates the high-precision quantization scale from its constrained dequantization counterpart and applies discrete search to mitigate scale quantization precision loss. SOAR is a post-training quantization framework that consistently outperforms existing NVFP4 baselines across multiple LLMs with no additional hardware overhead or memory cost.

- 🎯 Method: CJSO derives closed-form analytical solutions for jointly optimizing global and block-wise NVFP4 scales via reconstruction error minimization.
- 💡 Innovation: DSS decouples quantization scale (high-precision) from dequantization scale (constrained), then performs discrete search to recover precision lost to scale quantization.
- 📊 Result: Consistently outperforms existing NVFP4 quantization baselines across multiple LLMs at identical memory footprint with no added hardware overhead.
- 🔧 Engineering: Drop-in PTQ framework with native NVFP4 hardware support; no additional inference-time cost beyond standard NVFP4 deployment.

---

### KV cache compression (top 1 of cap 2)

### 4. KV-Fold: One-Step KV-Cache Recurrence for Long-Context Inference (8/10) 🔁
**arxiv** · `2605.12471` · 2026-05-12
👥 Alireza Nadali, Patrick Cooper, Ashutosh Trivedi... · 🏷 cs.LG, cs.AI, cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.12471) · [PDF](https://arxiv.org/pdf/2605.12471v1)
📡 Sources: arxiv
🧪 kv_cache · KV cache chunked recurrence, training-free · Llama-3.1-8B · cal: training-free, one-step protocol · perf: single 40GB GPU, tractable forward passes; absolute speedup not reported

#### Summary
KV-Fold addresses long-context inference by treating the KV cache as an accumulator in a left fold (foldl) over sequence chunks—each chunk is processed conditioned on the carried KV cache prefix, which is then enlarged and passed forward, requiring no model modification or retraining. The method repurposes KV cache concatenation (originally developed for latent multi-agent communication) as a chunk-to-chunk recurrence mechanism. On needle-in-a-haystack benchmarks, KV-Fold achieves 100% exact-match retrieval across 152 trials with contexts from 16K to 128K tokens and chain depths up to 511 on Llama-3.1-8B, all within a single 40GB GPU, while per-step drift saturates into a stable plateau robust to 10,000x changes in numerical precision.

- 🎯 Method: Training-free KV cache recurrence via chunk-wise foldl; model attends to accumulated KV prefix without architectural changes or fine-tuning.
- 📊 Result: 100% exact-match retrieval on needle-in-a-haystack across 16K–128K token contexts, chain depth up to 511, on single 40GB GPU.
- 💡 Innovation: Per-step drift saturates to a stable plateau insensitive to 10,000x precision changes and consistent across chunk sizes and model families.
- ⚠️ Limitation: Memory grows with accumulated KV cache depth; compared to streaming methods, bounded memory is not guaranteed.

---

### Knowledge distillation (top 2 of cap 2)

### 5. OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning (7/10) 🔁
**arxiv** · `2605.12400` · 2026-05-12
👥 Yuxiao Yang, Xiaoyun Wang, Weitong Zhang · 🏷 cs.LG, cs.AI
🔗 [arXiv](https://arxiv.org/abs/2605.12400) · [PDF](https://arxiv.org/pdf/2605.12400v1)
📡 Sources: arxiv
🧪 distillation · on-policy self-distillation with outcome-guided logit steering · cal: on-policy trajectory collection + outcome verification; cost unknown · perf: not reported

#### Summary
On-policy self-distillation (OPSD) for LLM reasoning suffers from a mismatch between teacher and student distributions caused by reflection-induced bias and response templates, leading to miscalibrated token-level supervision. OGLS-SD addresses this by using verifiable outcome rewards to contrast successful and failed on-policy trajectories, then steering teacher logits to calibrate token-level supervision signals. This combines outcome-level correctness (sparse reward) with dense token-level guidance via logit steering, stabilizing self-distillation and outperforming standard OPSD and its variants across diverse reasoning benchmarks.

- 🎯 Method: Outcome-guided logit steering calibrates teacher logits by contrasting correct vs. failed on-policy trajectories using verifiable rewards.
- 💡 Innovation: Identifies reflection-induced bias and template artifacts as a systematic miscalibration source in on-policy self-distillation.
- 📊 Result: Outperforms standard OPSD and multiple variants across diverse reasoning benchmarks.
- 🔧 Engineering: Combines sparse outcome-level rewards with dense token-level logit steering without requiring an external teacher model.

---

### 6. OmniThoughtVis: A Scalable Distillation Pipeline for Deployable Multimodal Reasoning Models (7/10) 🔁
**arxiv** · `2605.11629` · 2026-05-12
👥 Yuanhao Yue, Chengyu Wang, Yuanjie Lyu... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.11629) · [PDF](https://arxiv.org/pdf/2605.11629v1)
📡 Sources: arxiv
🧪 distillation · CoT reasoning distillation with difficulty-aware curation · Qwen3-VL-8B · cal: 1.8M curated samples; full training on distilled corpus required

#### Summary
OmniThoughtVis addresses the gap between high-capacity teacher MLLMs and smaller deployment-oriented models by providing a scalable CoT distillation pipeline. Starting from a diverse open-source seed pool, it generates structured CoT traces with joint annotation of reasoning difficulty, answer quality, and semantic task tags, then applies rule-based filtering, difficulty-aware selection, and tag-based diversity sampling to produce a 1.8M-sample curated corpus. Distilling Qwen3-VL models (2B–8B parameters) using this pipeline yields up to +16.8 points on MathVerse and +5.6 points on MMMU-Pro for the 4B model, with the distilled 4B model matching or surpassing the undistilled 8B baseline on several benchmarks.

- 🎯 Method: Scalable CoT distillation pipeline with difficulty-aware selection and tag-based diversity sampling producing 1.8M curated multimodal reasoning samples.
- 📊 Result: Distilled Qwen3-VL 4B gains +16.8 points on MathVerse and +5.6 points on MMMU-Pro over baseline.
- 📊 Result: Distilled 4B model matches or surpasses undistilled 8B baseline, demonstrating cross-scale efficiency gains.
- 💡 Innovation: Joint annotation of reasoning difficulty, answer quality, and semantic task tags enables controllable subset construction for downstream training.

---

## 📚 Full List (by score, descending)

| # | Title | Score | Topic | Pract | Bucket | Sources | Code | Date |
|---|-------|-------|-------|-------|--------|---------|------|------|
| 1 | [Search Your Block Floating Point Scales!](https://arxiv.org/abs/2605.12464) | 9 | 5 | 4 | PTQ (post-training quantization) | arxiv | — | 05-12 |
| 2 | [Grid Games: The Power of Multiple Grids for Quantizing Large Language Models](https://arxiv.org/abs/2605.12327) | 9 | 5 | 4 | PTQ (post-training quantization) | arxiv | — | 05-12 |
| 3 | [SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization](https://arxiv.org/abs/2605.12245) | 9 | 5 | 4 | PTQ (post-training quantization) | arxiv | — | 05-12 |
| 4 | [KV-Fold: One-Step KV-Cache Recurrence for Long-Context Inference](https://arxiv.org/abs/2605.12471) | 8 | 4 | 4 | KV cache compression | arxiv | — | 05-12 |
| 5 | [Efficient LLM-based Advertising via Model Compression and Parallel Verification](https://arxiv.org/abs/2605.11582) | 8 | 4 | 4 | PTQ (post-training quantization) | arxiv | — | 05-12 |
| 6 | [OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning](https://arxiv.org/abs/2605.12400) | 7 | 4 | 3 | Knowledge distillation | arxiv | — | 05-12 |
| 7 | [ROMER: Expert Replacement and Router Calibration for Robust MoE LLMs on Analog Compute-in-Memory Systems](https://arxiv.org/abs/2605.11800) | 7 | 4 | 3 | PTQ (post-training quantization) | arxiv | — | 05-12 |
| 8 | [OmniThoughtVis: A Scalable Distillation Pipeline for Deployable Multimodal Reasoning Models](https://arxiv.org/abs/2605.11629) | 7 | 4 | 3 | Knowledge distillation | arxiv | — | 05-12 |


## 🔁 Revisited

- [Search Your Block Floating Point Scales!](https://arxiv.org/abs/2605.12464) — score 9
- [Grid Games: The Power of Multiple Grids for Quantizing Large Language Models](https://arxiv.org/abs/2605.12327) — score 9
- [SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization](https://arxiv.org/abs/2605.12245) — score 9
- [KV-Fold: One-Step KV-Cache Recurrence for Long-Context Inference](https://arxiv.org/abs/2605.12471) — score 8
- [OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning](https://arxiv.org/abs/2605.12400) — score 7

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
