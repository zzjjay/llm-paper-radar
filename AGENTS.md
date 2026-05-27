# AGENTS.md — llm-paper-radar

Repo-level guidance for coding agents (Claude Code, etc.) operating in this repo.

## Don't silently skip expensive pipeline steps

When running `scripts/daily.sh`, `scripts/snapshot.sh`, or any wrapper, **do
not** add `PAPER_RIVER_SKIP=1` (or `PAPER_RIVER_MAX=<small N>`, or any
other knob that disables/throttles work the user didn't ask to skip) on
your own initiative — even when you think it'll take too long.

The `auto_paper_river` step runs the `ljg-paper-river` skill in headless
mode for every paper surfaced in the current rollup window (default
`--window-days 1`, matching `daily.sh`'s `--days`). Typical cron runs
generate a handful of `.org` files — but a `--all-history` backfill or
a long `--days` window can balloon to hundreds × 5-10 min each. The
output `.org` files are the whole point of the run.

If you're worried about runtime, **ask** before skipping:

> "auto_paper_river will run /ljg-paper-river per paper for N papers
> (~M minutes each, ~T total). Run it / cap at K / skip entirely?"

Use `AskUserQuestion` with options like:
- Run all (default — matches cron behavior)
- Cap at N this run (set `PAPER_RIVER_MAX=N`)
- Skip this step (set `PAPER_RIVER_SKIP=1`)

Same principle applies to anything else in this repo: don't disable
fetch, dedupe, summarize, translate, render, push, or cron-equivalent
behavior unless the user explicitly asks. The cron-driven invocations
(`cron/daily_wrapper.sh` etc.) intentionally run the full pipeline with
no caps; manual runs should default to the same unless the user opts out.

## Manual reruns

When the user says "rerun", "force", or "redo today" — match what cron
would do. Pass `--force` if needed for idempotency, but keep all the
heavy steps enabled by default.
