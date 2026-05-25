#!/usr/bin/env bash
# Daily pipeline: fetch -> dedupe -> filter -> summarize -> render -> commit + push.
# Designed for cron. Sources ~/.bashrc so the AMD Anthropic proxy env vars are present.
#
# Usage:
#   daily.sh                 # default 7-day window (matches Tue/Fri cron cadence)
#   daily.sh --days 60       # backfill last 60 days
#   daily.sh --no-fetch      # skip the source fetch step; re-run dedupe→render
#                              against whatever is already in data/raw/
#   DAYS=14 daily.sh         # env var also works

set -uo pipefail

DAYS="${DAYS:-7}"
NO_FETCH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --days) DAYS="$2"; shift 2 ;;
        --days=*) DAYS="${1#*=}"; shift ;;
        --no-fetch) NO_FETCH=1; shift ;;
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
# --no-fetch skips this section entirely so we can re-run dedupe→render
# against whatever raw data is already on disk (no external API calls).
if [[ "$NO_FETCH" -eq 1 ]]; then
    echo "[$(date -Is)] fetch: skipped (--no-fetch)"
else
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
fi

run_step "dedupe"    uv run python -m pipeline.dedupe    --backfill-days "${BACKFILL}" || exit 4
run_step "filter"    uv run python -m pipeline.filter    --backfill-days "${BACKFILL}" || exit 5
run_step "summarize" uv run python -m pipeline.summarize --backfill-days "${BACKFILL}" || exit 6

# Auto-generate paper-river .org files for every surfaced paper that
# doesn't have one yet. Delegates to scripts/auto_paper_river.py which
# scans data/summarized/, dedupes against paper-river/, and invokes
# scripts/gen_paper_river.sh per paper (which runs the ljg-paper-river
# Claude Code skill in headless mode). Default has NO cap — set
# PAPER_RIVER_MAX=N to cap per run (e.g. 2 in cron, unlimited for
# weekend backfills). PAPER_RIVER_SKIP=1 disables this step entirely.
# Non-fatal: pipeline ships even if generation fails.
if [[ "${PAPER_RIVER_SKIP:-0}" -eq 1 ]]; then
    echo "[$(date -Is)] auto_paper_river: skipped (PAPER_RIVER_SKIP=1)"
else
    run_step "auto_paper_river" uv run python scripts/auto_paper_river.py --no-warn-sleep \
        || echo "  (auto_paper_river failed, continuing)"
fi

# Auto-translate any zh paper-river/*.org that lacks an _en.org sibling
# BEFORE render, so the resulting `_en.org` files get picked up by this
# run's render step. Runs AFTER auto_paper_river so the just-generated
# zh files get translated in the same run. Non-fatal.
run_step "translate_paper_river" uv run python scripts/translate_paper_river.py --all \
    || echo "  (paper-river translation failed, continuing)"

run_step "render"    uv run python -m pipeline.render    --backfill-days "${BACKFILL}" || exit 7

# Snapshot the rendered paper list into snapshots/<start>-<end>-<N>days.md
# so each run leaves a browsable record (git history is preserve-only;
# this gives a side-by-side comparison surface without `git show` games).
DAYS="${DAYS}" run_step "snapshot" ./scripts/snapshot.sh || echo "  (snapshot failed, continuing)"

DATE_STR=$(date -u +%Y-%m-%d)

# Commit + push if anything changed. The window may produce multiple digest
# files, so stage the whole digests/ tree rather than just today's file.
if [[ -z "$(git status --porcelain)" ]]; then
    echo "[$(date -Is)] no changes to commit"
else
    git add data/summarized digests/ README.md INDEX.md data/seen.json snapshots/ paper-river/ 2>/dev/null || true
    if [[ -z "$(git diff --cached --name-only)" ]]; then
        echo "[$(date -Is)] nothing staged after add"
    else
        git commit -m "📚 Digest ${DATE_STR} (window=${DAYS}d)" || { echo "commit failed"; exit 8; }
        git push || { echo "push failed"; exit 9; }
        echo "[$(date -Is)] pushed digest for ${DATE_STR}"
    fi
fi

echo "[$(date -Is)] daily.sh done"
