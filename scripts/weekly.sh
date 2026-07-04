#!/usr/bin/env bash
# Weekly rollup: re-aggregate the last 7 daily digests into weekly/<start>-<end>.md.
# Designed for cron (Mondays at Beijing 16:00, ~2h after daily.sh on Mondays
# so the week's last digest is in before the rollup runs). Mirrors daily.sh's
# env setup so git push + uv work the same way under cron.
#
# Default window: yesterday UTC and the 6 days before it. When the cron
# fires on Beijing Monday 16:00 (= UTC Monday 08:00), yesterday UTC is
# Sunday, so the window becomes [last Mon, last Sun] — matching the
# historical backfilled rollups (e.g. weekly/20260518-20260524.md).
# Using "today UTC" instead would produce [last Tue, this Mon] and skip
# the previous Monday whenever the cron fires.
#
# Usage:
#   weekly.sh                       # default: yesterday UTC and the 6 days before
#   weekly.sh --end-date 2026-05-25 # explicit end date (manual rerun)

set -uo pipefail

END_DATE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --end-date) END_DATE="$2"; shift 2 ;;
        --end-date=*) END_DATE="${1#*=}"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Default to yesterday UTC when no --end-date is given. `date -u -d
# 'yesterday'` works on GNU coreutils (the deploy host). Manual reruns
# can pin any date via --end-date.
if [[ -z "$END_DATE" ]]; then
    END_DATE=$(date -u -d 'yesterday' +%Y-%m-%d)
fi

PROJECT_ROOT="/proj/xcohdstaff7/zhaolin/code/llm-paper-radar"
LOG_DIR="${PROJECT_ROOT}/scripts/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/weekly-$(date +%Y-%m-%d).log"

exec >>"$LOG_FILE" 2>&1
echo "================================================================"
echo "[$(date -Is)] weekly.sh start (end_date=${END_DATE})"

# Pull in user env (ANTHROPIC_BASE_URL / ANTHROPIC_CUSTOM_HEADERS / PATH / uv).
# Same -u dance as daily.sh.
# shellcheck disable=SC1090
set +u
source "$HOME/.bashrc" || true
set -u

cd "$PROJECT_ROOT"

export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

echo "[$(date -Is)] git pull --rebase"
git pull --rebase --autostash || { echo "git pull failed, aborting"; exit 3; }

echo "[$(date -Is)] step: pipeline.weekly --end-date ${END_DATE}"
uv run python -m pipeline.weekly --end-date "$END_DATE" || { echo "weekly failed"; exit 4; }

if [[ -z "$(git status --porcelain snapshots/weekly/ README.md)" ]]; then
    echo "[$(date -Is)] no weekly changes to commit"
else
    git add snapshots/weekly/ README.md
    if [[ -z "$(git diff --cached --name-only)" ]]; then
        echo "[$(date -Is)] nothing staged after add"
    else
        WEEK_TAG=$(date -u +%Y-W%V)
        git commit -m "📅 Weekly digest ${WEEK_TAG}" || { echo "commit failed"; exit 5; }
        git push || { echo "push failed"; exit 6; }
        echo "[$(date -Is)] pushed weekly digest ${WEEK_TAG}"
    fi
fi

echo "[$(date -Is)] weekly.sh done"
