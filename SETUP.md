# 🚀 Setup your own radar

Fork llm-paper-radar and point it at a different topic (RL fine-tuning, robot learning, agent evals, …). Most of the retargeting is in [`prompts/relevance.md`](prompts/relevance.md); the rest is config + scheduling.

For background on what the pipeline does, see the [README](README.md) — this doc only covers running your own copy.

## 1. Fork & clone

```bash
gh repo fork zhaolin-amd/llm-paper-radar --clone
cd llm-paper-radar
uv sync                       # installs deps from pyproject.toml + uv.lock
```

## 2. Configure access to Claude

The pipeline calls Anthropic via the official SDK. You can use either path:

- **Anthropic API directly:** set `ANTHROPIC_API_KEY` to your key and unset `ANTHROPIC_BASE_URL`.
- **Custom proxy / gateway:** set `ANTHROPIC_BASE_URL` and any required `ANTHROPIC_CUSTOM_HEADERS` (e.g. enterprise subscription header). `ANTHROPIC_API_KEY` can stay as a placeholder.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or, for a proxy:
export ANTHROPIC_BASE_URL="https://your-proxy.example.com/Anthropic"
export ANTHROPIC_CUSTOM_HEADERS="Subscription-Key: ..."
```

## 3. (Optional) Add other source credentials

All four sources (`hf_daily`, `arxiv`, `arxiv_authors`, `openreview`) work without credentials.

## 4. Customize the filter rubric

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

## 5. Tune the config knobs

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

## 6. Smoke-test the chain

```bash
./scripts/daily.sh                       # full run; logs to scripts/log/YYYY-MM-DD.log
./scripts/daily.sh --days 7 --no-fetch   # re-run dedupe→render only, no external API calls
```

If everything is wired up, you'll see `data/raw/`, `data/deduped/`, `data/scored/`, `data/summarized/` populate, then a fresh `digests/YYYY-MM-DD.md` plus an updated `README.md`. `--no-fetch` is handy when you tweak the prompt or filter and want to re-process whatever is already on disk.

## 7. Schedule it

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
