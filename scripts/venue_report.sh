#!/usr/bin/env bash
# Fetch, score, and group a conference's accepted papers for the venue
# trend report. This is stages 1-3 only; the analysis + report writing is
# done by the top model reading the abstracts — see skills/venue-trend/SKILL.md.
set -euo pipefail

VENUE="${1:?usage: venue_report.sh <venue> e.g. MLSys.org/2026/Conference}"
CONF_SLUG="$(echo "$VENUE" | cut -d'.' -f1 | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')-$(echo "$VENUE" | cut -d'/' -f2)"

echo "[1/3] fetching accepted papers for $VENUE"
uv run python -m sources.openreview_venue --venue "$VENUE"

echo "[2/3] scoring papers for LLM inference deployment relevance"
uv run python -m pipeline.venue_filter \
  --in-path "data/raw/${CONF_SLUG}/accepted.json" \
  --out-path "data/scored/${CONF_SLUG}.json"

echo "[3/3] grouping by subfield"
uv run python -m pipeline.venue_group \
  --scored-path "data/scored/${CONF_SLUG}.json" \
  --out-path "data/scored/${CONF_SLUG}-grouped.json"

echo "done: data/scored/${CONF_SLUG}-grouped.json"
