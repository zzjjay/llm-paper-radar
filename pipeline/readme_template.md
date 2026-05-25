# 📡 LLM Paper Radar

> Daily, automated digest of LLM compression and inference-optimization papers.

A small pipeline that fetches papers from arXiv + HF Daily + OpenReview + watched authors, kills obvious off-topic locally with a keyword prefilter, scores the rest with Claude Sonnet 4.6 against a two-axis rubric (topic relevance × practicality), tags each survivor with one of eight topic buckets (PTQ / Low-bit / QAT / KV cache / Pruning & distillation / Diffusion / Survey & methodology / Trending), and renders two views: a compact **table-only README** for skimming and a **per-day detail page** with summaries, "why this paper" rationale, and related/compared methods. No numeric threshold — anything not hard-gated surfaces, with per-bucket caps controlling digest length. A single cron job keeps it running.

[Today's digest](#-todays-digest) · [How papers are scored](#-how-papers-are-scored) · [Pipeline](#-pipeline) · [Setup your own radar](#-setup-your-own-radar) · [Repo layout](#-repo-layout)

---

## 📰 Today's digest

> Auto-updated daily at 06:00 local time. The README is the **compact table view** — full summaries, why-selected rationale, and related methods live under [`digests/`](digests/), one md per day. See [INDEX.md](INDEX.md) for history.

<!-- LATEST_START -->
<!-- LATEST_END -->

---

## 🧮 How papers are scored

Two stages: a cheap local **keyword prefilter**, then **Claude Sonnet 4.6** scoring with [`prompts/relevance.md`](prompts/relevance.md).

### Stage 1 — keyword prefilter (local, free)

Configured under `filter.prefilter` in [`config.yaml`](config.yaml). Each paper's title + abstract is matched against two word-boundary-aware lists:
- **whitelist** (e.g. `PTQ`, `AWQ`, `MXFP4`, `W4A4`, `BitNet`, `KV cache quantization`, …)
- **blacklist** (e.g. `image classification`, `ImageNet`, `federated learning`, `RAG`, …)

If a paper has **zero whitelist hits AND ≥ `max_blacklist_hits` (default 2) blacklist hits**, it's hard-gated locally with no LLM call. Word boundaries are enforced so `QuIP` doesn't match inside `equipping` and `MIT` doesn't match inside `Amit`.

This is conservative on purpose — anything ambiguous goes to the LLM. Add patterns over time based on what you see in rejected.jsonl.

### Stage 2 — Sonnet two-axis rubric

Sonnet returns a structured JSON breakdown which the orchestrator combines into a 0–10 composite.

| axis | range | what it captures |
|---|---|---|
| `topic_relevance` | 0–5 | How squarely the paper sits in LLM compression. 5 = core PTQ/QAT/KV/low-bit/pruning_distill/diffusion work with proper accuracy benchmarks. 3 = compression-adjacent (sparse attention coupled with quant). 0 = unrelated. |
| `practicality` | 0–5 | Algorithm simplicity + clear OR plausible inference perf + low calibration cost + small GPU memory footprint. 5 = AWQ-style "few lines, big speedup, data-free." Already-shipped numbers are a bonus, not a requirement — a credible deployment path is enough. 0 = complex, intractable calibration, no perf story. |

`relevance_score = topic_relevance + practicality` (so 0–10). **There is no numeric threshold** — every paper with `hard_gate=false` is surfaced. Per-bucket caps below control digest length.

### Hard gate

`hard_gate = true` zeros both axes. Triggered when:
- Topic completely unrelated to compression: RAG, agents, alignment, multimodal-without-compression, pure training algorithms
- **Pure speculative decoding** with no compression angle (coupled spec + quant is in scope, routed by primary contribution)
- **Pure review-article surveys** that only enumerate prior methods with no new measurement — hard_gate. Empirical comparison studies, bottleneck analyses, and evaluation-methodology papers now go to the `survey` bucket instead.
- Anything compression-adjacent that doesn't fit one of the eight topic buckets
- **`practicality = 2`** (only one favorable practicality signal) — the deployment-cost-to-gain ratio is wrong. PR ≥ 3 stays in scope.
- **No benchmark validation AND would otherwise score ≤ 7**: `accuracy_benchmarks` ∈ {none, unknown} AND `topic_relevance + practicality ≤ 7` → hard_gate, unless the paper explicitly compares against an established compression baseline (GPTQ, AWQ, SmoothQuant, QuaRot, BitNet, KIVI, …). A high-relevance + high-practicality paper (TR ≥ 4 AND PR ≥ 4) can still surface without a benchmark.
- Largest model tested clearly < 1B parameters (BERT-base, GPT-2-small)
- **`ptq` bucket — stricter scale rule**: any PTQ paper whose largest experiment is < 7B parameters → hard_gate. Sub-7B PTQ (FLAN-T5-base, CLIP-ViT, GPT-2, OPT-1.3B, Pythia-1.4B) does not predict large-scale behavior — accuracy gaps at 1B routinely flip at 7B+. Modern LLM family + unknown size → default trust. Other buckets keep the < 1B threshold.
- **Unstructured sparsity** that requires novel GPU kernels not yet in shipping inference stacks (vLLM / TensorRT-LLM / SGLang). Pruning needs a credible deployment path on existing kernels — N:M structured sparsity, MoE expert pruning, layer drop are in scope; learnable unstructured-mask methods waiting on speculative hardware are not.

### Topic buckets and per-bucket caps

Eight LLM-pickable buckets — the first seven are strict compression buckets, `trending` is a soft catch-all for compression-adjacent decoding-acceleration work. Papers that don't fit any bucket are hard-gated rather than forced into a catch-all (no `other`). Current caps (from [`config.yaml`](config.yaml) under `render.topic_caps`):

| bucket | cap | what goes here |
|---|---|---|
| **`ptq`** | 8 | Post-training quantization, weight-only / weight-activation / KV-quant when PTQ recipe is the primary contribution. Bit-width ≥ 3. Examples: GPTQ, AWQ, SmoothQuant, QuaRot, SpinQuant, MXFP4 PTQ, NVFP4 |
| **`low_bits`** | 5 | Sub-3-bit (≤ 2-bit) quantization, regardless of training method. Examples: BitNet b1.58, AQLM, VPTQ, QuIP#, ternary, binary |
| **`qat`** | 5 | Quantization-aware training or PTQ + full-network fine-tune. Bit-width ≥ 3. Examples: LLM-QAT, EfficientQAT, PB-LLM |
| **`kv_cache`** | 5 | KV cache compression where the layout / eviction is the main contribution. Examples: KIVI, KVQuant, H2O, StreamingLLM |
| **`pruning_distill`** | 3 | Pruning, sparsity, distillation **with a credible deployment path on existing kernels** (N:M structured, MoE expert pruning, layer drop, SFT-style KD). Unstructured-sparsity methods that depend on speculative GPU kernels → hard_gate. Examples: Wanda, SparseGPT, Sheared LLaMA, MiniLLM |
| **`diffusion`** | 3 | Quant / pruning / distillation / step-distillation on diffusion or flow-matching backbones. Examples: Q-Diffusion, SVDQuant |
| **`survey`** | 3 | Methodology / measurement / cross-method comparison that doesn't propose a new algorithm but gives actionable guidance. Examples: empirical PTQ comparisons, activation/outlier bottleneck studies, LLM-evaluation methodology for compression. Pure review-article surveys still hard_gate. |
| **`trending`** | 3 | Compression-adjacent **decoding-acceleration** work without a direct compression algorithm — parallel/dual-view drafters, spec-decoding frameworks that don't fit the seven strict buckets. Soft catch-all; use sparingly. Routine EAGLE/Medusa variants still hard_gate. Examples: Orthrus, DFlash. |

Bit-width tie-break: a 2-bit PTQ paper goes to `low_bits`, not `ptq`. A 1.58-bit pretrained model goes to `low_bits`, not `qat`. The rule wins over "natural" categorization.

Survey-vs-algorithm tie-break: if the primary contribution is a new algorithm (new loss / rotation / datatype / calibration recipe), bucket by the algorithm even if the paper also contains broad benchmarking. `survey` is only for work whose primary contribution is the measurement / comparison / methodology itself.

In the README compact view, *every* surviving paper appears in the main table (no cap). The per-bucket caps only control which papers get a full detail block on the per-day digest page.

### Table ordering

The README table sorts **bucket-first** (PTQ → Low-bit → QAT → KV cache → Pruning & distillation → Diffusion → Survey → Trending), then within each bucket by a composite score:

```
composite   = relevance_score × 30 + heat_score
heat_score  = trending_bonus + hf_daily_upvotes + star_bonus
trending_bonus = 100 / hf_daily_rank             (rank 1..30, else 0)
star_bonus     = min(log(github_stars + 1) × 3, 25)
```

`relevance_score` (0–10) dominates — a 9/10 paper outweighs ~270 HF upvotes — but `heat_score` lets a viral paper (HF Daily #1 = +100 trending bonus) surface ahead of a same-bucket peer with slightly higher relevance. `star_bonus` adds a soft signal when the paper has an official GitHub repo (read directly from HF's `githubStars` field, no API call): ~1k stars ≈ +21, capped at 25 so a popular framework repo can't outweigh a high-relevance paper. Papers without a known repo simply get 0 — never penalized. Tweak weights via `RELEVANCE_WEIGHT`, `STAR_WEIGHT`, `STAR_BONUS_CAP` in [`pipeline/render.py`](pipeline/render.py).

### 👤 Watched authors

A separate `arxiv_authors` source queries arXiv directly for a curated list of authors / groups (Dan Alistarh / IST Austria, Song Han / MIT HAN Lab, Qualcomm AI Research) over a rolling window. Each per-day detail page has a dedicated **👤 Watched authors** section showing *all* of their papers, bypassing per-bucket caps. The compact README table still surfaces them inline alongside everyone else, but only when they pass `hard_gate` — out-of-scope work from a watched author (e.g. video / world-model papers) stays in the detail page and does not pollute the main table. Edit the list in [`config.yaml`](config.yaml) under `sources.arxiv_authors.authors`.

---

## 🛠 Pipeline

```
   ┌────────────┐     fetchers (one per source, run sequentially in daily.sh)
   │  sources   │ ─── arxiv + arxiv_authors + hf_daily + openreview
   └─────┬──────┘            ↓
         │            data/raw/YYYY-MM-DD/{source}.json
         ↓
   ┌────────────┐     pipeline/dedupe.py
   │   dedupe   │ ─── merge by arXiv id, keep all source attributions
   └─────┬──────┘            ↓
         │            data/deduped/YYYY-MM-DD.json
         ↓
   ┌────────────┐     pipeline/filter.py
   │   filter   │ ─── stage 1: keyword prefilter (local, free) →
   │            │                obvious off-topic → hard_gate, no LLM call
   │            │     stage 2: Claude Sonnet 4.6 + prompts/relevance.md →
   │            │                two-axis score + topic_bucket + breakdown
   └─────┬──────┘            ↓
         │            data/scored/YYYY-MM-DD.json
         ↓
   ┌────────────┐     pipeline/summarize.py  (Claude Opus 4.7, prompts/summary.md)
   │ summarize  │ ─── every non-hard-gated paper → bilingual (zh + en) summary
   │            │                                  + highlights + up to 3 related/compared methods
   └─────┬──────┘            ↓
         │            data/summarized/YYYY-MM-DD.json
         ↓
   ┌────────────┐     pipeline/render.py
   │   render   │ ─── compact README table (bucket-ordered) +
   │            │     per-day zh detail page (+ _en sibling when bilingual
   │            │     summaries exist), grouped by bucket, capped per-bucket
   └────────────┘            ↓
                      digests/YYYY-MM-DD.md (+ _en.md)  +  README.md  +  INDEX.md
```

Each stage is independently runnable from the CLI. Per-day sources take `--backfill-days`; windowed sources (`arxiv_authors`, `openreview`) take `--window-days` and fetch once:

```bash
uv run python -m sources.hf_daily       --backfill-days 0
uv run python -m sources.arxiv          --backfill-days 0
uv run python -m sources.arxiv_authors  --backfill-days 0 --window-days 7
uv run python -m sources.openreview     --backfill-days 0 --window-days 7
uv run python -m pipeline.dedupe        --backfill-days 0
uv run python -m pipeline.filter        --backfill-days 0
uv run python -m pipeline.summarize     --backfill-days 0
uv run python -m pipeline.render        --backfill-days 0
```

`scripts/daily.sh` chains the fetch → dedupe → filter → summarize sequence, then runs three optional wrappers (`auto_paper_river` → `translate_paper_river` → `render` → `snapshot`) and finally `git commit && git push` if anything changed. The wrappers are governed by env vars: set `PAPER_RIVER_SKIP=1` to skip auto-generation of paper-river `.org` files, and `PAPER_RIVER_MAX=N` to cap how many get generated per run.

### 🌊 Paper River (optional companion analyses)

The render step auto-injects a `🌊 Paper River` link on each detail page when a matching `paper-river/<acronym>-<arxiv-id>.org` file exists (e.g. `GSQ-2604.18556.org` for arXiv `2604.18556`). The zh digest (`digests/<date>.md`) links to the zh `.org`; the en digest (`digests/<date>_en.md`) links to the `_en.org` sibling — each language keeps its own link. The render step also accepts a legacy dash form (`*<id-with-dashes>.org`, e.g. `2604-18556`) for old files. Those `.org` files are deep-lineage analyses — "倒读法": recursively trace 5 layers of a paper's intellectual lineage, then walk forward Feynman-style — produced via the `ljg-paper-river` Claude Code skill. By default `scripts/daily.sh` auto-generates them via `scripts/auto_paper_river.py` (set `PAPER_RIVER_SKIP=1` to disable) and auto-translates zh → `_en.org` via `scripts/translate_paper_river.py`. Render works fine without any `.org` file — no file → no link.

**Install the skill.** It ships in the [`lijigang/ljg-skills`](https://github.com/lijigang/ljg-skills) Claude Code plugin marketplace — install in two slash commands, no clone needed (`master` branch = org-mode output, which is what `pipeline/render.py` expects):

```
/plugin marketplace add lijigang/ljg-skills
/plugin install ljg-skills@ljg-skills
```

The plugin bundles the whole `ljg-*` collection (all `ljg-*` skills become available); `ljg-paper-river` is the one this repo uses. Verify with `/skills | grep ljg-paper-river`.

Then invoke per paper (e.g. `/ljg-paper-river https://arxiv.org/abs/<id>`), save the output as `paper-river/<acronym>-<id-slug>.org` from the radar repo root, and re-run `uv run python -m pipeline.render --date <date>` (or the next `daily.sh`).

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

All four sources (`hf_daily`, `arxiv`, `arxiv_authors`, `openreview`) work without credentials.

### 4. Customize the filter rubric

The scoring rubric lives entirely in [`prompts/relevance.md`](prompts/relevance.md) — there is no Python change needed to retarget the radar at a different domain.

What to edit, in rough order of leverage:

1. **`# What we care about`** — swap the Primary / Secondary / Out-of-scope bullets for your topic (e.g. replace "LLM compression" with "RL fine-tuning", "robot learning", "agent evals"). This single section drives most of the model's judgment.
2. **`## topic_relevance (0-5)`** — re-anchor each level for your domain. The Sonnet scorer follows these anchors literally; vague anchors → noisy scores.
3. **`# Hard gates`** — list paper shapes that should always score 0 (e.g. "BERT-base only" for compression; for RL you might gate "tabular-only / no neural net").
4. **`topic_bucket` enum** — these become the section headings in the daily digest. Keep them few (≤10) and mutually distinct; the digest groups + caps by bucket.
5. **Few-shot examples** — at least 3 positive + 2 negative anchors with the exact `topic_relevance` / `practicality` / `topic_bucket` values you want. This is the highest-ROI edit; the model imitates these.

The JSON output schema at the bottom of the prompt must stay structurally the same (`pipeline/filter.py` parses it), but you can change the enum values inside.

After editing, dry-run on existing deduped data without re-fetching:

```bash
uv run python -m pipeline.filter --backfill-days 6 \
    --in-root data/deduped --out-root /tmp/scored-test
ls /tmp/scored-test/                          # inspect a few JSONs by hand
```

### 5. Tune the config knobs

Companion settings in [`config.yaml`](config.yaml) — change these without touching the prompt:

| key | default | what it does |
|---|---|---|
| `filter.model` | `claude-sonnet-4-6` | scoring model; drop to `claude-haiku-4-5` if cost matters more than judgement quality |
| `filter.prefilter.max_blacklist_hits` | 2 | papers with ≥ N blacklist matches AND zero whitelist hits are hard-gated locally, no LLM call |
| `filter.prefilter.whitelist` / `blacklist` | curated for LLM compression | per-pattern weights; word-boundary matched. Tune from rejected.jsonl over time |
| `summarize.model` | `claude-opus-4-7` | bilingual (zh + en) summary model |
| `render.topic_caps` | `{ptq: 8, low_bits: 5, qat: 5, kv_cache: 5, pruning_distill: 3, diffusion: 3, survey: 3, trending: 3, _default: 2}` | max papers per bucket on the per-day detail page; README compact view is uncapped |
| `sources.arxiv.categories` | `[cs.CL, cs.LG, cs.AR]` | arXiv categories pulled at fetch time |
| `sources.arxiv_authors.window_days` | 7 | default fetch window for watched-authors source (CLI `--window-days` overrides) |
| `sources.openreview.venues` | `[ICLR.cc/2026/Conference]` | OpenReview venue IDs to scrape; append `/-/Submission` is handled internally. Add NeurIPS / ICML / COLM entries as they open. |
| `dedupe.source_priority` | hf_daily → openreview → arxiv_authors → arxiv | tie-breaker order when the same paper shows up from multiple sources |

### 6. Smoke-test the chain

```bash
./scripts/daily.sh                       # full run; logs to scripts/log/YYYY-MM-DD.log
./scripts/daily.sh --days 7 --no-fetch   # re-run dedupe→render only, no external API calls
```

If everything is wired up, you'll see `data/raw/`, `data/deduped/`, `data/scored/`, `data/summarized/` populate, then a fresh `digests/YYYY-MM-DD.md` plus an updated `README.md`. `--no-fetch` is handy when you tweak the prompt or filter and want to re-process whatever is already on disk.

### 7. Schedule it

**Option A — host crontab (works behind a corporate Anthropic proxy).** This is what this fork actually uses, because the LLM endpoint sits inside the AMD network and isn't reachable from public CI runners:

```bash
crontab -e
# add (replace path):
0 6 * * * /absolute/path/to/llm-paper-radar/scripts/daily.sh
```

`scripts/daily.sh` sources `~/.bashrc` for the Anthropic env vars, runs the full pipeline (`fetch → dedupe → filter → summarize → render`), and only commits + pushes when something actually changed. Logs land in `scripts/log/YYYY-MM-DD.log`.

**Option B — GitHub Actions (forks with a public sk-ant key).** Three workflows are wired up under `.github/workflows/`: `daily.yml` (schedule commented out — fetch → render → push), `weekly.yml` (Mondays 23:00 UTC — 7-day rollup), and `cleanup.yml` (Sundays 22:00 UTC — prune old raw data). To use them, set repo secrets and (for `daily.yml`) re-enable the schedule line:

| secret | required | what for |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | sk-ant-… key |
| `ANTHROPIC_BASE_URL` | optional | proxy / gateway URL; leave unset for default api.anthropic.com |
| `ANTHROPIC_CUSTOM_HEADERS` | optional | extra headers required by your proxy |
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
├── config.yaml                  # source toggles, models, prefilter, topic_caps
├── seeds.yaml                   # curated index of important papers per bucket (used by paper-triage skill)
├── prompts/
│   ├── relevance.md             # filter rubric (two-axis + buckets + anchors)
│   └── summary.md               # bilingual (zh + en) summary format prompt
├── sources/                     # one fetcher per upstream
│   ├── arxiv.py
│   ├── arxiv_authors.py
│   ├── hf_daily.py
│   └── openreview.py
├── pipeline/
│   ├── config.py                # Pydantic config model
│   ├── llm_client.py            # async Anthropic wrapper with prompt cache
│   ├── dedupe.py
│   ├── filter.py                # two-axis scoring
│   ├── summarize.py             # bilingual zh + en summaries
│   ├── render.py                # bucket grouping + README splicing + Paper River link
│   ├── readme_template.md       # static doc template (this file's source)
│   ├── rollup.py                # N-day rollup compact view used by render
│   └── weekly.py                # 7-day roll-up as a compact table
├── scripts/
│   ├── daily.sh                 # cron entrypoint: fetch → ... → push (--no-fetch skips fetch)
│   ├── snapshot.sh              # captures the current README paper-list into snapshots/
│   ├── auto_paper_river.py      # scan summarized/, invoke ljg-paper-river per missing paper
│   ├── gen_paper_river.sh       # headless wrapper that runs the ljg-paper-river skill
│   ├── translate_paper_river.py # auto-translate zh paper-river/*.org → _en.org
│   ├── seed_add.py              # add a paper to seeds.yaml
│   └── seed_reject.py           # log a paper into data/curation/rejected.jsonl
├── digests/
│   ├── YYYY-MM-DD.md            # daily digest archive (Chinese)
│   └── YYYY-MM-DD_en.md         # English sibling (only days summarized after bilingual prompt landed)
├── snapshots/
│   └── YYYYMMDD-YYYYMMDD-Ndays.md  # per-run paper-list snapshot for tracking history
├── paper-river/                 # optional: ljg-paper-river deep-lineage analyses (see Pipeline)
│   ├── <acronym>-<arxiv-id>.org # zh; render auto-links if present
│   └── <acronym>-<arxiv-id>_en.org  # en sibling; auto-translated by daily.sh
├── data/                        # mostly gitignored; seen.json + summarized/ + curation/ kept
│   ├── raw/                     # gitignored
│   ├── deduped/                 # gitignored
│   ├── scored/                  # gitignored
│   ├── summarized/              # tracked
│   ├── curation/                # tracked: rejected.jsonl + seed-curation logs
│   └── seen.json                # tracked: papers seen across days for 🔁 marker
├── .github/workflows/           # daily.yml + weekly.yml + cleanup.yml
├── tests/
└── pyproject.toml + uv.lock
```

---

## 📜 License

MIT.
