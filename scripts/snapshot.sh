#!/usr/bin/env bash
# Save the current README paper list + INDEX to backups/<utc-timestamp>-w<window>d/.
# Called from daily.sh after render; can also be run manually any time
# to capture the current state of the working tree.
#
# Usage:
#   ./scripts/snapshot.sh            # uses DAYS env var or "manual"
#   DAYS=7 ./scripts/snapshot.sh     # tagged with -w7d
#   LABEL=rescreen ./scripts/snapshot.sh   # custom label suffix
#
# Output layout:
#   backups/<UTC-YYYYMMDD-HHMMSS>-w<N>d/
#     paper-list.md       # the LATEST_START..LATEST_END block of README
#                         # (the only part that changes per run)
#     INDEX.md            # full INDEX (small, copied verbatim)
#     bucket-counts.txt   # cheap diff helper: paper count per bucket
#     header.txt          # window + scanned/surfaced summary
#     meta.txt            # git HEAD sha + timestamp

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f README.md ]]; then
    echo "snapshot: README.md not found" >&2
    exit 1
fi

TS="$(date -u +%Y%m%d-%H%M%S)"
# Window tag: numeric DAYS becomes "w7d", non-numeric (e.g. manual label) used verbatim.
DAYS_RAW="${DAYS:-manual}"
if [[ "$DAYS_RAW" =~ ^[0-9]+$ ]]; then
    TAG="w${DAYS_RAW}d"
else
    TAG="$DAYS_RAW"
fi
[[ -n "${LABEL:-}" ]] && TAG="${TAG}-${LABEL}"
SNAP_DIR="backups/${TS}-${TAG}"
mkdir -p "$SNAP_DIR"

# Extract the LATEST_START..LATEST_END block from README. That marker pair
# wraps the digest content; everything outside is static documentation.
awk '/<!-- LATEST_START -->/{flag=1} flag; /<!-- LATEST_END -->/{flag=0}' \
    README.md > "$SNAP_DIR/paper-list.md"

# INDEX.md is ~10KB even for 184 days — copy whole.
[[ -f INDEX.md ]] && cp INDEX.md "$SNAP_DIR/INDEX.md"

# Per-bucket paper count for cheap diff of "what changed in distribution".
awk -F'|' '/^\| [0-9]/{print $3}' README.md \
    | sed 's/^ *//;s/ *$//' \
    | sort | uniq -c | sort -rn > "$SNAP_DIR/bucket-counts.txt"

# Top-level header (window + scanned/surfaced counts).
grep -E "^> 📅 Window|^> 📊 Scanned" README.md | head -2 > "$SNAP_DIR/header.txt"

# Provenance: git sha + UTC timestamp for traceability.
{
    echo "ts_utc=${TS}"
    echo "window=${DAYS:-manual}"
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    [[ -n "${LABEL:-}" ]] && echo "label=${LABEL}"
} > "$SNAP_DIR/meta.txt"

echo "snapshot → $SNAP_DIR"
