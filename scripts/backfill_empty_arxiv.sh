#!/usr/bin/env bash
# Backfill days whose arxiv.json is still empty after a throttle-failed fetch.
#
# Why this exists: the daily cron hits export.arxiv.org from a shared corporate
# egress IP and periodically gets 429-stormed (see scripts/log/2026-06-0*.log).
# When the primary api/query path exhausts its budget it now falls back to
# OAI-PMH, but a day can still end up with an empty data/raw/<day>/arxiv.json if
# both paths failed at cron time. Those days would otherwise stay empty forever
# (the digest is built off HF Daily trending alone → ~1 paper). This sweep
# re-attempts each still-empty recent day on a later, hopefully un-throttled run
# and re-renders it if papers come back.
#
# This is an EXPLICIT recovery step, not a silent skip: every still-empty day is
# logged, every re-fetch outcome (recovered / still-empty / failed) is logged
# with a WARN on failure. It never overwrites a non-empty arxiv.json (the source
# itself guards that — sources/arxiv.py).
#
# Usage:
#   backfill_empty_arxiv.sh [LOOKBACK_DAYS]   # default 7
#   BACKFILL_EMPTY_SKIP=1 daily.sh            # disable from the daily cron
#
# A "genuine zero day" (arxiv really published nothing, e.g. a quiet weekend —
# api/query returns opensearch:totalResults=0) writes an empty arxiv.json that
# is CORRECT. We cannot distinguish that from a throttle-zero by file content
# alone, so this sweep re-attempts both; a genuine zero simply comes back empty
# again and is reported as "still empty (likely genuine zero day)".

set -uo pipefail

LOOKBACK="${1:-7}"
if ! [[ "$LOOKBACK" =~ ^[0-9]+$ ]] || [[ "$LOOKBACK" -lt 1 ]]; then
    echo "LOOKBACK_DAYS must be a positive integer, got: $LOOKBACK" >&2
    exit 1
fi

PROJECT_ROOT="/proj/xcohdstaff7/zhaolin/code/llm-paper-radar"
cd "$PROJECT_ROOT" || { echo "cannot cd to $PROJECT_ROOT" >&2; exit 1; }

RAW_ROOT="data/raw"
recovered=0
still_empty=0
failed=0
attempted=0

echo "[$(date -Is)] backfill_empty_arxiv: scanning last ${LOOKBACK} day(s)"

# Walk target days yesterday-UTC back through LOOKBACK days. Day 0 (today-UTC)
# is intentionally skipped: arxiv hasn't finished publishing it yet, so an empty
# file there is expected, not a failure.
for delta in $(seq 1 "$LOOKBACK"); do
    day="$(date -u -d "${delta} days ago" +%Y-%m-%d)"
    f="${RAW_ROOT}/${day}/arxiv.json"

    # Non-empty (> 2 bytes, i.e. more than "[]") → nothing to do.
    if [[ -f "$f" && "$(stat -c%s "$f" 2>/dev/null || echo 0)" -gt 2 ]]; then
        continue
    fi

    attempted=$((attempted + 1))
    echo "[$(date -Is)] backfill_empty_arxiv: ${day} has empty/missing arxiv.json — re-fetching"

    # Re-fetch ONLY the arxiv source for this one day. RADAR_TARGET_DATE pins
    # every component to this publication day. The source's own clobber guard
    # refuses to overwrite a non-empty file, so this is safe even if a parallel
    # run populated it meanwhile.
    if ! RADAR_TARGET_DATE="$day" uv run python -m sources.arxiv --force; then
        echo "[$(date -Is)] backfill_empty_arxiv: WARN ${day} re-fetch returned non-zero"
        failed=$((failed + 1))
        sleep 30
        continue
    fi

    # Did we actually get papers this time?
    if [[ -f "$f" && "$(stat -c%s "$f" 2>/dev/null || echo 0)" -gt 2 ]]; then
        echo "[$(date -Is)] backfill_empty_arxiv: ${day} recovered — re-running dedupe→render"
        ok=1
        for step in dedupe filter summarize render; do
            if ! RADAR_TARGET_DATE="$day" uv run python -m "pipeline.${step}" --force; then
                echo "[$(date -Is)] backfill_empty_arxiv: WARN ${day} pipeline step '${step}' failed"
                ok=0
                break
            fi
        done
        if [[ "$ok" -eq 1 ]]; then
            recovered=$((recovered + 1))
        else
            failed=$((failed + 1))
        fi
    else
        echo "[$(date -Is)] backfill_empty_arxiv: ${day} still empty (likely genuine zero day, or still throttled)"
        still_empty=$((still_empty + 1))
    fi

    # Polite gap between days so the sweep itself doesn't trip the throttle.
    sleep 30
done

echo "[$(date -Is)] backfill_empty_arxiv: done — attempted=${attempted} recovered=${recovered} still_empty=${still_empty} failed=${failed}"

# Non-zero exit only on hard failures, so the caller can surface a real problem
# without treating "genuine zero weekend" as an error.
if [[ "$failed" -gt 0 ]]; then
    exit 1
fi
