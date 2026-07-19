#!/usr/bin/env bash
# Shared failure-alert helper for the cron scripts (daily.sh / weekly.sh /
# rollup.sh / backfill_empty_arxiv.sh).
#
# Why this exists: for a month, backfill_empty_arxiv.sh's render step failed
# on every run (unsupported --force flag) but only ever `echo`'d a WARN into
# a log file nobody was tailing. The pipeline "succeeded" from cron's point
# of view the whole time. A failure that isn't reported to a human is
# indistinguishable from no failure at all — so any step whose failure was
# previously "log it and continue" now also emails, in addition to
# continuing. This does NOT change control flow (non-fatal stays non-fatal);
# it only makes non-fatal failures visible instead of silent.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/alert.sh"
#   alert_failure "daily.sh" "step 'render' failed (exit 7)" "$LOG_FILE"
#
# Best-effort: a broken mail setup must never take down the pipeline, so
# send failures here are only echoed, never fatal (no `set -e` interaction).

ALERT_EMAIL="${ALERT_EMAIL:-zhaolin@amd.com}"

alert_failure() {
    local script="$1" reason="$2" log_file="${3:-}"
    local subject="[llm-paper-radar] ${script} FAILED: ${reason}"
    local body
    if [[ -n "$log_file" && -f "$log_file" ]]; then
        body="$(printf '%s\n\n--- last 100 lines of %s ---\n' "$reason" "$log_file"; tail -n 100 "$log_file")"
    else
        body="$reason"
    fi
    if command -v mail >/dev/null 2>&1; then
        if ! echo "$body" | mail -s "$subject" "$ALERT_EMAIL" 2>/dev/null; then
            echo "[$(date -Is)] alert_failure: WARN 'mail' returned non-zero, alert not confirmed sent: ${reason}"
        fi
    else
        echo "[$(date -Is)] alert_failure: WARN 'mail' command not found, cannot send alert: ${reason}"
    fi
}
