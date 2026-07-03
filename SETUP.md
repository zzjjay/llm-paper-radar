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

If you run from a shared/corporate egress IP, arXiv's per-IP rate limit (`429`) bites. The fork mitigates this without an API key: every arXiv call sends a `User-Agent` with a contact `mailto:` (edit `ARXIV_USER_AGENT` in [`sources/base.py`](sources/base.py) to your address), watched-authors fetch in one merged query, and the main day-fetch falls back from the `api/query` endpoint to OAI-PMH when throttled. Days zeroed out by a throttle are retried on the next run by [`scripts/backfill_empty_arxiv.sh`](scripts/backfill_empty_arxiv.sh) (called from `daily.sh`; disable with `BACKFILL_EMPTY_SKIP=1`).

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
| `render.topic_caps` | `{ptq: 8, low_bits: 5, qat: 5, kv_cache: 5, pruning_distill: 3, diffusion: 3, trending: 3, survey: 3, _default: 2}` | max papers per bucket on the per-day detail page; README compact view is uncapped |
| `sources.arxiv.categories` | `[cs.CL, cs.LG, cs.AR]` | arXiv categories pulled at fetch time |
| `sources.arxiv_authors.window_days` | 7 | default fetch window for watched-authors source (CLI `--window-days` overrides) |
| `sources.openreview.venues` | `[ICLR.cc/{year}/Conference, ICML.cc/{year}/Conference, NeurIPS.cc/{year}/Conference, MLSys.org/{year}/Conference, AAAI.org/{year}/Conference, aclweb.org/ACL/{year}/Conference, EMNLP/{year}/Conference]` | OpenReview venue templates to scrape; `/-/Submission` is appended internally. `{year}` is auto-expanded to the current calendar year and the next one (a venue's CFP window typically straddles both), so the config doesn't need a yearly edit; venues that don't exist yet are silently skipped. Pin a specific year by writing it literally (no `{year}`). Add COLM.cc/{year}/Conference and similar as they open. ACL/EMNLP main tracks flow through ARR (`aclweb.org/ACL/ARR/<yr>/<month>`) — swap in if the direct venues stay empty. Only `/-/Submission` notes are fetched — reviews, comments, and rebuttals live under different invitations and are not pulled. |
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
# Simple host-local schedule (fire at host 06:00 every day):
0 6 * * * /absolute/path/to/llm-paper-radar/scripts/daily.sh

# Or — fire at Beijing 14:00 (= UTC 06:00) regardless of host timezone,
# via the included wrapper. By Beijing 14:00 arxiv has finished publishing
# the previous UTC day, so the digest covers a complete publication day:
0 * * * * /absolute/path/to/llm-paper-radar/scripts/cron_wrapper.sh \
    --at-hour 14 /absolute/path/to/llm-paper-radar/scripts/daily.sh
```

`scripts/daily.sh` sources `~/.bashrc` for the Anthropic env vars, runs the full pipeline (`fetch → dedupe → filter → summarize → render`), and only commits + pushes when something actually changed. Logs land in `scripts/log/YYYY-MM-DD.log`. For a manual rerun of a specific arxiv publication day: `./scripts/daily.sh --date 2026-05-20`.

Weekly (`scripts/weekly.sh`) and the longer roll-ups (`scripts/rollup.sh monthly|halfyear|yearly`) each read only from `data/summarized/` — they don't depend on each other or need to run in any particular order relative to one another, only *after* that window's daily runs have landed. Fire them ~2h after `daily.sh` so the day's data is in, and stagger the ones that share a trigger day (monthly/halfyear/yearly all fire on the 1st) by a few minutes so they don't `git push` at the same time:

```bash
# Weekly: Beijing Monday 16:00 → 7-day rollup.
0 * * * * /absolute/path/to/llm-paper-radar/scripts/cron_wrapper.sh \
    --at-hour 16 --dow 1 /absolute/path/to/llm-paper-radar/scripts/weekly.sh

# Monthly: Beijing 1st-of-month 16:00 → previous calendar month.
0 * 1 * * /absolute/path/to/llm-paper-radar/scripts/cron_wrapper.sh \
    --at-hour 16 /absolute/path/to/llm-paper-radar/scripts/rollup.sh monthly

# Half-year: Beijing Jan-1/Jul-1 16:05 → the half-year that just ended.
5 * 1 1,7 * /absolute/path/to/llm-paper-radar/scripts/cron_wrapper.sh \
    --at-hour 16 /absolute/path/to/llm-paper-radar/scripts/rollup.sh halfyear

# Yearly: Beijing Jan-1 16:10 → previous calendar year.
10 * 1 1 * /absolute/path/to/llm-paper-radar/scripts/cron_wrapper.sh \
    --at-hour 16 /absolute/path/to/llm-paper-radar/scripts/rollup.sh yearly
```

**Option B — GitHub Actions (forks with a public sk-ant key).** Two workflows are wired up under `.github/workflows/`: `daily.yml` (schedule commented out — fetch → render → push) and `weekly.yml` (Mondays 23:00 UTC — 7-day rollup). `data/summarized/` is retained permanently (it is the source for the weekly + monthly/half-year/yearly rollups), so there is no cleanup workflow. To use them, set repo secrets and (for `daily.yml`) re-enable the schedule line:

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
