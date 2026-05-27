#!/usr/bin/env bash
# Save the current README paper list to snapshots/<start>-<end>-<N>days.md.
# Called from daily.sh after render; can also be run manually any time
# to capture the current state of the working tree.
#
# Usage:
#   ./scripts/snapshot.sh            # window (start/end/N) all parsed from README
#   LABEL=rescreen ./scripts/snapshot.sh   # appended before .md
#
# Output:
#   snapshots/<YYYYMMDD>-<YYYYMMDD>-<N>days[-<label>].md
#     - top: window line + scanned/surfaced line + git sha + snapshot ts
#     - body: the LATEST_START..LATEST_END block of README

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f README.md ]]; then
    echo "snapshot: README.md not found" >&2
    exit 1
fi

# Best-effort: generate paper-river for any surfaced paper missing one,
# then translate any zh paper-river without an _en sibling. Both share
# paper-river/ with daily.sh, so file-existence dedup means snapshot
# won't redo what daily already did. PAPER_RIVER_SKIP=1 disables both.
# Non-fatal — snapshot proceeds even if generation/translation flakes.
if [[ "${PAPER_RIVER_SKIP:-0}" -ne 1 ]]; then
    uv run python scripts/auto_paper_river.py --no-warn-sleep \
        || echo "snapshot: auto_paper_river failed, continuing"
    uv run python scripts/translate_paper_river.py --all \
        || echo "snapshot: translate_paper_river failed, continuing"
fi

# Parse window from the header line. Two formats are supported:
#   "> 📅 Publication date: YYYY-MM-DD (UTC)"          (single-day, current)
#   "> 📅 Window: YYYY-MM-DD → YYYY-MM-DD"             (multi-day rollup, legacy)
WINDOW_LINE="$(grep -m1 -E "^> 📅 (Publication date|Window)" README.md || true)"
if [[ -z "$WINDOW_LINE" ]]; then
    echo "snapshot: could not find '📅 Publication date' or '📅 Window' line in README.md" >&2
    exit 1
fi
# Extract all YYYY-MM-DD dates; 1 (single-day) or 2 (rollup) expected.
DATES=( $(echo "$WINDOW_LINE" | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2}") )
if [[ "${#DATES[@]}" -eq 1 ]]; then
    START_DASH="${DATES[0]}"
    END_DASH="${DATES[0]}"
elif [[ "${#DATES[@]}" -ge 2 ]]; then
    START_DASH="${DATES[0]}"
    END_DASH="${DATES[1]}"
else
    echo "snapshot: failed to parse date(s) from: $WINDOW_LINE" >&2
    exit 1
fi
START="${START_DASH//-/}"
END="${END_DASH//-/}"

# Window length N is always derived from the parsed window so the filename
# never disagrees with the content. Single-day → N=1. DAYS env is intentionally
# ignored here.
START_EPOCH="$(date -u -d "$START_DASH" +%s)"
END_EPOCH="$(date -u -d "$END_DASH" +%s)"
DIFF=$(( (END_EPOCH - START_EPOCH) / 86400 ))
N=$(( DIFF == 0 ? 1 : DIFF ))

NAME="${START}-${END}-${N}days"
[[ -n "${LABEL:-}" ]] && NAME="${NAME}-${LABEL}"

mkdir -p snapshots
OUT="snapshots/${NAME}.md"

SCANNED_LINE="$(grep -m1 -E "^> 📊 Scanned" README.md || true)"
TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

{
    echo "<!-- snapshot: ${NAME} -->"
    echo "<!-- ts_utc=${TS_UTC} git_head=${GIT_HEAD} -->"
    echo
    [[ -n "$WINDOW_LINE"  ]] && echo "$WINDOW_LINE"
    [[ -n "$SCANNED_LINE" ]] && echo "$SCANNED_LINE"
    echo
    awk '/<!-- LATEST_START -->/{flag=1} flag; /<!-- LATEST_END -->/{flag=0}' README.md
} > "$OUT"

echo "snapshot → $OUT"
