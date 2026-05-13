#!/usr/bin/env bash
# Daily pipeline: fetch -> dedupe -> filter -> summarize -> render -> commit + push.
# Designed for cron. Sources ~/.bashrc so the AMD Anthropic proxy env vars are present.

set -uo pipefail

PROJECT_ROOT="/proj/xcohdstaff7/zhaolin/code/llm-paper-radar"
LOG_DIR="${PROJECT_ROOT}/scripts/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y-%m-%d).log"

exec >>"$LOG_FILE" 2>&1
echo "================================================================"
echo "[$(date -Is)] daily.sh start"

# Pull in user env (ANTHROPIC_BASE_URL / ANTHROPIC_CUSTOM_HEADERS / PATH / uv).
# shellcheck disable=SC1090
source "$HOME/.bashrc" || true

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

# Fetch all sources (today only). Source-level failures (e.g. credentials missing)
# print and continue per-source, so don't abort the whole job on one bad source.
for src in hf_daily arxiv reddit semantic_scholar twitter_rsshub; do
    echo "[$(date -Is)] fetch: $src"
    uv run python -m "sources.$src" --backfill-days 0 || echo "  ($src returned non-zero, continuing)"
done

run_step "dedupe"    uv run python -m pipeline.dedupe    --backfill-days 0 || exit 4
run_step "filter"    uv run python -m pipeline.filter    --backfill-days 0 || exit 5
run_step "summarize" uv run python -m pipeline.summarize --backfill-days 0 || exit 6
run_step "render"    uv run python -m pipeline.render    --backfill-days 0 || exit 7

DATE_STR=$(date -u +%Y-%m-%d)

# Commit + push if anything changed.
if [[ -z "$(git status --porcelain)" ]]; then
    echo "[$(date -Is)] no changes to commit"
else
    git add data/summarized "digests/${DATE_STR}.md" README.md INDEX.md data/seen.json 2>/dev/null || true
    if [[ -z "$(git diff --cached --name-only)" ]]; then
        echo "[$(date -Is)] nothing staged after add"
    else
        git commit -m "📚 Daily digest ${DATE_STR}" || { echo "commit failed"; exit 8; }
        git push || { echo "push failed"; exit 9; }
        echo "[$(date -Is)] pushed digest for ${DATE_STR}"
    fi
fi

echo "[$(date -Is)] daily.sh done"
