#!/usr/bin/env bash
# Periodic rollup: re-aggregate already-rendered daily digests into a single
# compact table for a longer cadence — monthly / halfyear / yearly — under
# rollups/<cadence>/<start>-<end>.md. Reads only data/summarized/ (no fetch,
# no LLM); mirrors weekly.sh's env + git plumbing for cron.
#
# Window = the PREVIOUS completed calendar period relative to the anchor date
# (default: today UTC), NOT a trailing N days:
#   monthly  : 1st → last day of the previous month
#   halfyear : the half-year that just ended (Jan–Jun or Jul–Dec)
#   yearly   : Jan 1 → Dec 31 of the previous year
#
# Designed to fire on the 1st (and, for halfyear, also Jul 1), AFTER that day's
# daily.sh, so the period's final day is already in data/summarized/. The
# renderer hard-aborts if any day in the window is missing, so a too-early run
# fails loudly instead of committing a partial table.
#
# Usage:
#   rollup.sh monthly                 # previous month relative to today UTC
#   rollup.sh halfyear
#   rollup.sh yearly
#   rollup.sh monthly 2026-06-01      # anchor a specific run date (backfill)

set -uo pipefail

CADENCE="${1:-}"
ANCHOR="${2:-}"

case "$CADENCE" in
    monthly|halfyear|yearly) ;;
    *) echo "usage: rollup.sh <monthly|halfyear|yearly> [anchor-YYYY-MM-DD]" >&2; exit 1 ;;
esac

if [[ -n "$ANCHOR" ]]; then
    if ! [[ "$ANCHOR" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "anchor must be YYYY-MM-DD, got: $ANCHOR" >&2
        exit 1
    fi
    DATEREF=(-d "$ANCHOR")
else
    DATEREF=()
fi

# All boundary math is in UTC to match the daily/weekly pipeline's clock.
Y=$(date -u "${DATEREF[@]}" +%Y)
M=$(date -u "${DATEREF[@]}" +%m)
FIRST_THIS_MONTH=$(date -u "${DATEREF[@]}" +%Y-%m-01)

case "$CADENCE" in
    monthly)
        LABEL="Monthly"
        START=$(date -u -d "${FIRST_THIS_MONTH} -1 month" +%Y-%m-01)
        END=$(date -u -d "${FIRST_THIS_MONTH} -1 day" +%Y-%m-%d)
        TAG=$(date -u -d "$START" +%Y-%m)
        ;;
    halfyear)
        LABEL="Half-Year"
        # 10# forces base-10 so a leading-zero month (08, 09) isn't read as octal.
        if (( 10#$M >= 7 )); then          # Jul–Dec → the half that ended is Jan–Jun this year
            START="${Y}-01-01"; END="${Y}-06-30"; TAG="${Y}-H1"
        else                                # Jan–Jun → the half that ended is Jul–Dec last year
            START="$((Y - 1))-07-01"; END="$((Y - 1))-12-31"; TAG="$((Y - 1))-H2"
        fi
        ;;
    yearly)
        LABEL="Yearly"
        START="$((Y - 1))-01-01"; END="$((Y - 1))-12-31"; TAG="$((Y - 1))"
        ;;
esac

PROJECT_ROOT="/proj/xcohdstaff7/zhaolin/code/llm-paper-radar"
OUT_DIR="snapshots/${CADENCE}"
LOG_DIR="${PROJECT_ROOT}/scripts/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/rollup-${CADENCE}-$(date +%Y-%m-%d).log"

exec >>"$LOG_FILE" 2>&1
echo "================================================================"
echo "[$(date -Is)] rollup.sh start (cadence=${CADENCE} window=${START}..${END} tag=${TAG})"

# Pull in user env (ANTHROPIC_* / PATH / uv). Same -u dance as weekly.sh.
# shellcheck disable=SC1090
set +u
source "$HOME/.bashrc" || true
set -u

cd "$PROJECT_ROOT"

export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

echo "[$(date -Is)] git pull --rebase"
git pull --rebase --autostash || { echo "git pull failed, aborting"; exit 3; }

echo "[$(date -Is)] step: pipeline.rollup_digest ${LABEL} ${START}..${END}"
uv run python -m pipeline.rollup_digest \
    --start "$START" --end "$END" --label "$LABEL" --out-dir "$OUT_DIR" \
    || { echo "rollup_digest failed"; exit 4; }

if [[ -z "$(git status --porcelain "$OUT_DIR")" ]]; then
    echo "[$(date -Is)] no ${CADENCE} changes to commit"
else
    git add "$OUT_DIR"
    if [[ -z "$(git diff --cached --name-only)" ]]; then
        echo "[$(date -Is)] nothing staged after add"
    else
        git commit -m "📅 ${LABEL} digest ${TAG}" || { echo "commit failed"; exit 5; }
        git push || { echo "push failed"; exit 6; }
        echo "[$(date -Is)] pushed ${CADENCE} digest ${TAG}"
    fi
fi

echo "[$(date -Is)] rollup.sh done"
