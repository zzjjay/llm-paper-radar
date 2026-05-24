#!/usr/bin/env bash
# Daily pipeline: fetch -> dedupe -> filter -> summarize -> render -> commit + push.
# Designed for cron. Sources ~/.bashrc so the AMD Anthropic proxy env vars are present.
#
# Usage:
#   daily.sh                 # default 7-day window (matches Tue/Fri cron cadence)
#   daily.sh --days 60       # backfill last 60 days
#   DAYS=14 daily.sh         # env var also works

set -uo pipefail

DAYS="${DAYS:-7}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --days) DAYS="$2"; shift 2 ;;
        --days=*) DAYS="${1#*=}"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if ! [[ "$DAYS" =~ ^[0-9]+$ ]] || [[ "$DAYS" -lt 1 ]]; then
    echo "--days must be a positive integer, got: $DAYS" >&2
    exit 1
fi

BACKFILL=$((DAYS - 1))

PROJECT_ROOT="/proj/xcohdstaff7/zhaolin/code/llm-paper-radar"
LOG_DIR="${PROJECT_ROOT}/scripts/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d).log"

exec >>"$LOG_FILE" 2>&1
echo "================================================================"
echo "[$(date -Is)] daily.sh start (window=${DAYS}d, backfill=${BACKFILL})"

# Pull in user env (ANTHROPIC_BASE_URL / ANTHROPIC_CUSTOM_HEADERS / PATH / uv).
# Disable -u while sourcing: ~/.bashrc references vars like LS_COLORS that are
# unset in cron's environment, which would otherwise kill the script here.
# shellcheck disable=SC1090
set +u
source "$HOME/.bashrc" || true
set -u

if [[ -z "${ANTHROPIC_CUSTOM_HEADERS:-}" ]]; then
    echo "[$(date -Is)] ERROR: ANTHROPIC_CUSTOM_HEADERS not set after sourcing ~/.bashrc"
    exit 2
fi

cd "$PROJECT_ROOT"

# Make git play nice in cron — non-interactive SSH, predictable identity.
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

# Pull latest so we don't push on a stale base.
echo "[$(date -Is)] git pull --rebase"
git pull --rebase --autostash || { echo "git pull failed, aborting"; exit 3; }

run_step() {
    local name="$1"; shift
    echo "[$(date -Is)] step: $name"
    if ! "$@"; then
        echo "[$(date -Is)] step '$name' failed (exit $?)"
        return 1
    fi
}

# Fetch all sources over the requested window. Per-day sources loop via
# --backfill-days; windowed sources (arxiv_authors, openreview) take
# --window-days and fetch once. Source-level failures print and continue.
for src in hf_daily arxiv; do
    echo "[$(date -Is)] fetch: $src (--backfill-days ${BACKFILL})"
    uv run python -m "sources.$src" --backfill-days "${BACKFILL}" \
        || echo "  ($src returned non-zero, continuing)"
done
for src in arxiv_authors openreview; do
    echo "[$(date -Is)] fetch: $src (--window-days ${DAYS})"
    uv run python -m "sources.$src" --backfill-days 0 --window-days "${DAYS}" \
        || echo "  ($src returned non-zero, continuing)"
done

run_step "dedupe"    uv run python -m pipeline.dedupe    --backfill-days "${BACKFILL}" || exit 4
run_step "filter"    uv run python -m pipeline.filter    --backfill-days "${BACKFILL}" || exit 5
run_step "summarize" uv run python -m pipeline.summarize --backfill-days "${BACKFILL}" || exit 6
run_step "render"    uv run python -m pipeline.render    --backfill-days "${BACKFILL}" || exit 7

DATE_STR=$(date -u +%Y-%m-%d)

# Commit + push if anything changed. The window may produce multiple digest
# files, so stage the whole digests/ tree rather than just today's file.
if [[ -z "$(git status --porcelain)" ]]; then
    echo "[$(date -Is)] no changes to commit"
else
    git add data/summarized digests/ README.md INDEX.md data/seen.json 2>/dev/null || true
    if [[ -z "$(git diff --cached --name-only)" ]]; then
        echo "[$(date -Is)] nothing staged after add"
    else
        git commit -m "📚 Digest ${DATE_STR} (window=${DAYS}d)" || { echo "commit failed"; exit 8; }
        git push || { echo "push failed"; exit 9; }
        echo "[$(date -Is)] pushed digest for ${DATE_STR}"
    fi
fi

echo "[$(date -Is)] daily.sh done"
