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
> 📊 Scanned 394 papers → passed filter 9 → highlighted 6 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Highlights by topic

### PTQ (post-training quantization) (top 3 of cap 3)

### 1. Search Your Block Floating Point Scales! (9/10) 🔁
**arxiv** · `2605.12464` · 2026-05-12
👥 Tanmaey Gupta, Hayden Prairie, Xiaoxia Wu... · 🏷 cs.LG, cs.AR, cs.PF
🔗 [arXiv](https://arxiv.org/abs/2605.12464) · [PDF](https://arxiv.org/pdf/2605.12464v1)
📡 Sources: arxiv
🧪 ptq · NVFP4 BFP with learned scale search · Llama-3.1-70B · cal: Unknown; appears to be post-training scale search, likely minimal · perf: Near-zero performance loss; 27% quantization error reduction NVFP4

#### Summary
Standard Block Floating Point (BFP) quantization uses the block maximum as the scale factor, which is suboptimal for minimizing quantization error. ScaleSearch replaces this with a fine-grained search over the mantissa bits in microscaling formats (e.g., NVFP4) to find scales that minimize quantization error for a given distribution. Applied to PTQ and attention kernels, ScaleSearch reduces NVFP4 quantization error by 27%, improves MATH500 accuracy by up to 15 points for Qwen3-8B, and improves Wikitext-2 perplexity by up to 0.77 points for Llama 3.1 70B via ScaleSearchAttention.

- 🎯 Method: Fine-grained search over mantissa bits in microscaling BFP formats to minimize block quantization error, replacing the standard max-magnitude scale.
- 📊 Result: 27% reduction in NVFP4 quantization error; up to 15-point MATH500 improvement on Qwen3-8B PTQ.
- 📊 Result: ScaleSearchAttention improves Wikitext-2 PPL by up to 0.77 points for Llama 3.1 70B with near-zero performance loss.
- 💡 Innovation: NVFP4-based attention algorithm integrating ScaleSearch, compatible with existing PTQ pipelines and low-precision attention methods.

---

### 2. Grid Games: The Power of Multiple Grids for Quantizing Large Language Models (9/10) 🔁
**arxiv** · `2605.12327` · 2026-05-12
👥 Vage Egiazarian, Erik Schultheis, Andrei Panferov... · 🏷 cs.LG
🔗 [arXiv](https://arxiv.org/abs/2605.12327) · [PDF](https://arxiv.org/pdf/2605.12327v1)
📡 Sources: arxiv
🧪 ptq · Multi-grid FP4 (MXFP4, NVFP4 variants with learned grids)

#### Summary
Microscaled 4-bit formats like MXFP4 and NVFP4 use a single fixed floating-point grid per quantization group, leaving accuracy on the table. This paper formalizes the power-of-two-grids (PO2) problem, where one or more extra bits in the scale value select among multiple 4-bit grids per group, and proves theoretically that small-group formats benefit significantly from this while large groups do not. Four grid families are instantiated—PO2(NF4), MPO2, PO2(Split87), and SFP4 (TensorCore-compatible NVFP4 with two shifted variants)—and evaluated on PTQ of open LLMs and pre-training of Llama-like models, consistently showing accuracy improvements over single-grid FP4 in both weight-only and weight+activation quantization settings.

- 🎯 Method: Per-group 4-bit grid selection using 1+ scale bits (PO2), enabling adaptive quantization grids without changing bit-width
- 💡 Innovation: SFP4 pairs NVFP4 with two shifted variants for TensorCore compatibility; MPO2 learns grid pairs directly from real weights and activations
- 📊 Result: Adaptive PO2 grids consistently improve accuracy over single-grid FP4 in both PTQ and pre-training of Llama-like models
- ⚠️ Limitation: Theoretical analysis shows PO2 advantage vanishes for very large quantization groups

---

### 3. SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization (9/10) 🔁
**arxiv** · `2605.12245` · 2026-05-12
👥 Chengzhu Bao, Xianglong Yan, Zhiteng Li... · 🏷 cs.LG
🔗 [arXiv](https://arxiv.org/abs/2605.12245) · [PDF](https://arxiv.org/pdf/2605.12245v1)
📡 Sources: arxiv
🧪 ptq · NVFP4 PTQ with joint scale optimization · cal: Post-training only; cost details unknown · perf: Native hardware support; end-to-end speedup not reported

#### Summary
SOAR addresses suboptimal accuracy in NVFP4 post-training quantization of LLMs caused by inflexible scale selection and coupled treatment of quantization/dequantization scales. The framework introduces Closed-form Joint Scale Optimization (CJSO), which jointly optimizes global and block-wise scales via analytical solutions minimizing reconstruction error, and Decoupled Scale Search (DSS), which separates the high-precision quantization scale from its constrained dequantization counterpart and applies discrete search to reduce precision loss from scale quantization. SOAR consistently outperforms existing NVFP4 quantization baselines across multiple LLMs with no additional hardware overhead under the same memory footprint.

- 🎯 Method: CJSO derives closed-form analytical solutions for joint global and block-wise scale optimization via reconstruction error minimization in NVFP4.
- 💡 Innovation: DSS decouples quantization scale from dequantization scale, then uses discrete search to mitigate precision loss from scale quantization.
- 📊 Result: Consistently outperforms existing NVFP4 quantization baselines across multiple LLMs at identical memory footprint with zero hardware overhead.
- 🔧 Engineering: Targets native hardware support of NVFP4 microscaling format, making optimizations directly deployable without extra cost.

---

### KV cache compression (top 1 of cap 2)

### 4. KV-Fold: One-Step KV-Cache Recurrence for Long-Context Inference (8/10) 🔁
**arxiv** · `2605.12471` · 2026-05-12
👥 Alireza Nadali, Patrick Cooper, Ashutosh Trivedi... · 🏷 cs.LG, cs.AI, cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.12471) · [PDF](https://arxiv.org/pdf/2605.12471v1)
📡 Sources: arxiv
🧪 kv_cache · KV cache recurrence via left-fold chunk processing · Llama-3.1-8B · cal: training-free, no calibration required · perf: no end-to-end speedup reported; memory-bounded single GPU (40GB)

#### Summary
KV-Fold addresses long-context inference beyond a model's native context window without retraining by treating the KV cache as a left-fold accumulator over sequence chunks. Each chunk is processed conditioned on the concatenated KV cache from all prior chunks, creating a recurrence that reuses internal state across segments purely through the KV cache concatenation primitive. On needle-in-a-haystack benchmarks, it achieves 100% exact-match retrieval across 152 trials for contexts from 16K to 128K tokens with chain depths up to 511 on Llama-3.1-8B, all within a single 40GB GPU memory budget, outperforming streaming methods that sacrifice retrieval fidelity for bounded memory.

- 🎯 Method: Training-free KV cache recurrence via chunk-to-chunk left fold (foldl), conditioning each chunk on accumulated KV cache prefix without model modification.
- 📊 Result: 100% exact-match needle-in-a-haystack retrieval across 152 trials, 16K–128K token contexts, chain depths up to 511, on a single 40GB GPU.
- 💡 Innovation: Per-step drift saturates into a stable plateau insensitive to 10,000x changes in numerical precision and robust across chunk sizes and model families.
- ⚠️ Limitation: KV cache grows linearly with chain depth, constraining maximum context by GPU memory rather than achieving truly bounded memory like streaming methods.

---

### Knowledge distillation (top 2 of cap 2)

### 5. OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning (7/10) 🔁
**arxiv** · `2605.12400` · 2026-05-12
👥 Yuxiao Yang, Xiaoyun Wang, Weitong Zhang · 🏷 cs.LG, cs.AI
🔗 [arXiv](https://arxiv.org/abs/2605.12400) · [PDF](https://arxiv.org/pdf/2605.12400v1)
📡 Sources: arxiv
🧪 distillation · on-policy self-distillation with outcome-guided logit steering

#### Summary
On-policy self-distillation (OPSD) improves LLM reasoning by distilling teacher distributions along the model's own trajectories, but teacher responses suffer from reflection-induced bias and template artifacts that miscalibrate token-level supervision. OGLS-SD addresses this by using verifiable outcome rewards to contrast successful vs. failed on-policy trajectories and steer teacher logits, combining outcome-level correctness signals with dense token-level guidance. The method stabilizes self-distillation and outperforms standard OPSD and its variants across diverse reasoning benchmarks.

- 🎯 Method: Outcome-guided logit steering contrasts successful/failed on-policy trajectories to calibrate teacher logits and reduce reflection-induced bias in self-distillation.
- 💡 Innovation: Combines verifiable outcome rewards (sparse) with token-level logit steering (dense) to fix teacher-student distribution mismatch in OPSD.
- 📊 Result: Outperforms standard OPSD and multiple variants across diverse reasoning benchmarks.
- ⚠️ Limitation: Requires verifiable outcome rewards, limiting applicability to tasks with checkable correctness signals.

---

### 6. OmniThoughtVis: A Scalable Distillation Pipeline for Deployable Multimodal Reasoning Models (7/10) 🔁
**arxiv** · `2605.11629` · 2026-05-12
👥 Yuanhao Yue, Chengyu Wang, Yuanjie Lyu... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.11629) · [PDF](https://arxiv.org/pdf/2605.11629v1)
📡 Sources: arxiv
🧪 distillation · reasoning distillation with structured CoT curation · Qwen3-VL-8B · cal: 1.8M synthetic CoT samples; full training required (cost not itemized) · perf: not reported

#### Summary
OmniThoughtVis addresses the gap between large MLLM reasoning capability and deployment constraints by building a scalable distillation pipeline that transfers chain-of-thought reasoning from high-capacity teachers to smaller models (2B–8B parameters). The pipeline curates 1.8M multimodal CoT samples from open-source seeds with joint annotation of reasoning difficulty, answer quality, and semantic task tags, followed by rule-based filtering, difficulty-aware selection, and tag-based diversity sampling. Distilled Qwen3-VL models (2B–8B) show consistent benchmark improvements, including +16.8 points on MathVerse and +5.6 points on MMMU-Pro for the 4B model, with the distilled 4B matching or surpassing the undistilled 8B baseline on several tasks.

- 🎯 Method: Scalable CoT distillation pipeline with difficulty-aware selection and tag-based diversity sampling over 1.8M curated multimodal samples.
- 📊 Result: Distilled 4B Qwen3-VL gains +16.8 pts on MathVerse and +5.6 pts on MMMU-Pro versus undistilled baseline.
- 📊 Result: Distilled 4B model matches or surpasses undistilled 8B baseline on several benchmarks, enabling parameter-efficient deployment.
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
| 9 | [Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning](https://arxiv.org/abs/2605.11458) | 7 | 4 | 3 | Knowledge distillation | arxiv | — | 05-12 |


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

```bash
crontab -e
# add (replace path):
0 6 * * * /absolute/path/to/llm-paper-radar/scripts/daily.sh
```

The script is idempotent: it `git pull --rebase`s first, runs the pipeline, and only commits if there are real changes.

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
