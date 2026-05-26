#!/usr/bin/env bash
# Cron wrapper that gates execution on Asia/Shanghai (Beijing) wall-clock time.
#
# Ubuntu's vixie cron does NOT honor per-user CRON_TZ/TZ for the schedule
# fields — `man 5 crontab` explicitly recommends this wrapper pattern.
#
# Crontab fires this every hour. The wrapper computes the current
# Beijing time and only runs the underlying script when the hour matches
# (and, for the weekly job, the weekday matches).
#
# Usage:
#   cron_wrapper.sh --at-hour 6        daily.sh        # fire at Beijing 06:00
#   cron_wrapper.sh --at-hour 8 --dow 1 weekly.sh      # fire at Beijing Monday 08:00
#
# --dow: Beijing day of week, 1=Mon .. 7=Sun (POSIX %u).
# Remaining args after the gate flags are the script + any flags it takes.

set -euo pipefail

AT_HOUR=""
WANT_DOW=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --at-hour) AT_HOUR="$2"; shift 2 ;;
        --dow)     WANT_DOW="$2"; shift 2 ;;
        --)        shift; break ;;
        *)         break ;;
    esac
done

if [[ -z "$AT_HOUR" ]]; then
    echo "cron_wrapper: --at-hour is required" >&2
    exit 2
fi

if [[ $# -lt 1 ]]; then
    echo "cron_wrapper: missing target script" >&2
    exit 2
fi

NOW_HOUR=$(TZ=Asia/Shanghai date +%-H)
NOW_DOW=$(TZ=Asia/Shanghai date +%u)
NOW_FULL=$(TZ=Asia/Shanghai date -Iseconds)

if [[ "$NOW_HOUR" != "$AT_HOUR" ]]; then
    exit 0
fi
if [[ -n "$WANT_DOW" && "$NOW_DOW" != "$WANT_DOW" ]]; then
    exit 0
fi

echo "[$(date -Is)] cron_wrapper: gate passed (Beijing=${NOW_FULL}, hour=${NOW_HOUR}, dow=${NOW_DOW}); exec: $*"
exec "$@"
