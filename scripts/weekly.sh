#!/usr/bin/env bash
# Weekly rollup: re-aggregate the last 7 daily digests into weekly/<start>-<end>.md.
# Designed for cron (Mondays at Beijing 16:00, ~2h after daily.sh on Mondays
# so the week's last digest is in before the rollup runs). Mirrors daily.sh's
# env setup so git push + uv work the same way under cron.
#
# Usage:
#   weekly.sh                       # today's 7-day window (ending today UTC)
#   weekly.sh --end-date 2026-05-25 # explicit end date

set -uo pipefail

END_DATE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --end-date) END_DATE="$2"; shift 2 ;;
        --end-date=*) END_DATE="${1#*=}"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

PROJECT_ROOT="/proj/xcohdstaff7/zhaolin/code/llm-paper-radar"
LOG_DIR="${PROJECT_ROOT}/scripts/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/weekly-$(date +%Y-%m-%d).log"

exec >>"$LOG_FILE" 2>&1
echo "================================================================"
echo "[$(date -Is)] weekly.sh start (end_date=${END_DATE:-today})"

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

WEEKLY_ARGS=()
if [[ -n "$END_DATE" ]]; then
    WEEKLY_ARGS+=(--end-date "$END_DATE")
fi

echo "[$(date -Is)] step: pipeline.weekly"
uv run python -m pipeline.weekly "${WEEKLY_ARGS[@]}" || { echo "weekly failed"; exit 4; }

if [[ -z "$(git status --porcelain weekly/)" ]]; then
    echo "[$(date -Is)] no weekly changes to commit"
else
    git add weekly/
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
