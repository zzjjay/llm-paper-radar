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
> 📊 Scanned 358 papers → passed filter 4 → highlighted 4 (threshold ≥7)

> Auto-generated daily digest from [llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar).
> History: [INDEX.md](INDEX.md) · Config: [config.yaml](config.yaml) · Powered by Claude Sonnet 4.6

## 🔥 Highlights by topic

_Up to 3 PTQ, 2 others per topic — change in [`config.yaml`](config.yaml) under `render.topic_caps`._

### PTQ (post-training quantization)

### 1. Provable Quantization with Randomized Hadamard Transform (7/10) 🔁
**arxiv** · `2605.13810` · 2026-05-13
👥 Ying Feng, Piotr Indyk, Michael Kapralov... · 🏷 cs.LG, cs.DS
🔗 [arXiv](https://arxiv.org/abs/2605.13810) · [PDF](https://arxiv.org/pdf/2605.13810v1)
📡 Sources: arxiv
🧪 ptq · Hadamard rotation + dithered scalar quantization · cal: Data-free (random projection + offset). · perf: O(d log d) vs O(d²) for dense rotation; no end-to-end speedup measured.

#### Summary
Vector quantization using randomized Hadamard transform (HD) lacks strong theoretical guarantees due to HD's discrete structure, despite its O(d log d) efficiency advantage over Θ(d²) dense rotations. This work analyzes dithered quantization (random scalar offset subtracted before quantizing) combined with a single randomized Hadamard transform, proving it is unbiased and achieves MSE bounds asymptotically matching those of truly random rotation matrices. Specifically, a dithered TurboQuant achieves MSE (π√3/2 + o(1))·4^{-b} at b bits per coordinate, with the o(1) term vanishing uniformly over all unit vectors as quantization levels grow.

- 🎯 Method: Dithered quantization with single randomized Hadamard transform (HD) adds random scalar offset at negligible cost, enabling provable guarantees matching dense random rotations.
- 📊 Result: MSE bound (π√3/2 + o(1))·4^{-b} at b bits/coordinate, asymptotically matching optimal dense-rotation quantization.
- 💡 Innovation: Reduces computational cost from Θ(d²) (dense rotation) to O(d log d) (HD) while preserving asymptotically tight theoretical MSE guarantees.
- 📊 Result: o(1) error term vanishes uniformly over all unit vectors and all dimensions as quantization levels grow, giving strong dimension-free guarantees.

---

### Knowledge distillation

### 2. Prefix Teach, Suffix Fade: Local Teachability Collapse in Strong-to-Weak On-Policy Distillation (7/10) 🔁
**arxiv** · `2605.13643` · 2026-05-13
👥 Kaiyuan Liu, Ziyuan Zhuang, Yang Bai... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.13643) · [PDF](https://arxiv.org/pdf/2605.13643v1)
📡 Sources: arxiv
🧪 distillation · on-policy distillation with trajectory-specific supervision truncation · Qwen3 (family; scale unknown) · cal: on-policy rollouts required; change-point detection on trajectories (cost not quantified) · perf: not reported

#### Summary
On-policy distillation (OPD) trains student models on self-generated rollouts with dense teacher feedback, but full-sequence supervision degrades when later trajectory segments lack discriminative teacher-student contrast—a failure mode termed 'local teachability collapse'. The proposed fix is a trajectory-specific release rule that measures the teacher's margin over the student's top-K candidate set at NLTK-sentence granularity, then truncates dense OPD supervision at a BIC-style downward change point. Evaluated on the Qwen3 model family, the method outperforms standard full-trajectory OPD on five in-domain benchmarks across multiple student scales while better preserving out-of-domain capabilities.

- 🎯 Method: Truncate dense OPD supervision when teacher-student margin drops below a BIC-detected change point across sentence-level segments, rather than supervising full trajectories.
- 📊 Result: Outperforms full-trajectory OPD on 5 in-domain benchmarks at various student scales using Qwen3 model family.
- 💡 Innovation: Introduces 'local teachability collapse'—the insight that non-zero teacher-student advantage does not guarantee discriminative, learnable feedback at trajectory suffix regions.
- 📊 Result: Better out-of-domain capability preservation compared to baseline distillation methods, suggesting reduced over-supervision harm.

---

### 3. GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning (7/10) 🔁
**arxiv** · `2605.13136` · 2026-05-13
👥 Kasidit Sermsri, Teerapong Panboonyuen · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.13136) · [PDF](https://arxiv.org/pdf/2605.13136v1)
📡 Sources: arxiv
🧪 distillation · confidence-gated reasoning distillation · Flan-T5

#### Summary
Reasoning distillation from LLMs to compact models suffers from noisy rationales and hallucinated supervision due to open-loop teacher-student interactions that assume uniform teacher reliability. GateKD addresses this with a closed-loop framework using three confidence-gated mechanisms: soft supervision filtering, hidden-state alignment conditioned on teacher confidence, and reliability-filtered attention distillation. Evaluated on commonsense, logical, and symbolic reasoning benchmarks with T5/Flan-T5 backbones, GateKD consistently outperforms open-loop distillation baselines, with particularly substantial gains on logical and symbolic reasoning tasks.

- 🎯 Method: Three-component confidence gating (soft supervision, hidden-state evolution, attention distillation) forms a closed feedback loop suppressing hallucination transfer.
- 📊 Result: Consistent improvements over strong open-loop baselines on commonsense, logical, and symbolic reasoning benchmarks using T5 and Flan-T5 backbones.
- 💡 Innovation: Teacher treated as dynamic gatekeeper—confidence score continuously modulates distillation signal rather than assuming uniform teacher reliability.
- ⚠️ Limitation: Performance degrades measurably when any single gating component is removed, indicating sensitivity to full framework integrity.

---

### Pruning / sparsity

### 4. STOP: Structured On-Policy Pruning of Long-Form Reasoning in Low-Data Regimes (7/10) 🔁
**arxiv** · `2605.13165` · 2026-05-13
👥 Chenjun Xu, Zhennan Zhou, Zhan Su... · 🏷 cs.CL
🔗 [arXiv](https://arxiv.org/abs/2605.13165) · [PDF](https://arxiv.org/pdf/2605.13165v1)
📡 Sources: arxiv
🧪 pruning · on-policy reasoning trace pruning via ECN (Earliest Correct Node) · DeepSeek-R1-Distill-LLaMA-3-8B · cal: self-distilled traces on low-data fine-tuning; cost not quantified in words · perf: 19.4-42.4% reduction in generated tokens; end-to-end latency not explicitly reported

#### Summary
Long CoT reasoning models suffer from overthinking—generating redundant post-solution tokens that inflate inference cost, particularly problematic in low-data fine-tuning regimes where large-scale distillation is unavailable. STOP (Structured On-policy Pruning) addresses this by constructing self-distilled traces, parsing them into structured reasoning trees via node segmentation and taxonomy annotation, then truncating each trace at the Earliest Correct Node (ECN)—the shortest prefix ending at a correct answering conclusion. On DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-LLaMA-3-8B across GSM8K, Math 500, and AIME 2024, STOP achieves 19.4–42.4% token reduction with minimal accuracy degradation while inducing less distributional shift than teacher-guided pruning approaches.

- 🎯 Method: On-policy structured pruning via reasoning-tree construction + ECN truncation removes redundant post-solution reasoning while preserving semantic continuity.
- 📊 Result: 19.4–42.4% token reduction across GSM8K, Math 500, AIME 2024 with accuracy largely preserved in low-data fine-tuning.
- 💡 Innovation: ECN (Earliest Correct Node) identifies shortest correct prefix, reallocating reasoning effort from backtracking/verification toward productive exploration.
- ⚠️ Limitation: Evaluated only in low-data fine-tuning regimes; effectiveness at scale or with full training data not demonstrated.

---

## 📚 Full List (by score, descending)

| # | Title | Score | Topic | Pract | Bucket | Sources | Code | Date |
|---|-------|-------|-------|-------|--------|---------|------|------|
| 1 | [Provable Quantization with Randomized Hadamard Transform](https://arxiv.org/abs/2605.13810) | 7 | 4 | 3 | PTQ (post-training quantization) | arxiv | — | 05-13 |
| 2 | [Prefix Teach, Suffix Fade: Local Teachability Collapse in Strong-to-Weak On-Policy Distillation](https://arxiv.org/abs/2605.13643) | 7 | 4 | 3 | Knowledge distillation | arxiv | — | 05-13 |
| 3 | [STOP: Structured On-Policy Pruning of Long-Form Reasoning in Low-Data Regimes](https://arxiv.org/abs/2605.13165) | 7 | 4 | 3 | Pruning / sparsity | arxiv | — | 05-13 |
| 4 | [GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning](https://arxiv.org/abs/2605.13136) | 7 | 4 | 3 | Knowledge distillation | arxiv | — | 05-13 |


## 🔁 Revisited

- [Provable Quantization with Randomized Hadamard Transform](https://arxiv.org/abs/2605.13810) — score 7
- [Prefix Teach, Suffix Fade: Local Teachability Collapse in Strong-to-Weak On-Policy Distillation](https://arxiv.org/abs/2605.13643) — score 7
- [GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning](https://arxiv.org/abs/2605.13136) — score 7
- [STOP: Structured On-Policy Pruning of Long-Form Reasoning in Low-Data Regimes](https://arxiv.org/abs/2605.13165) — score 7

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
