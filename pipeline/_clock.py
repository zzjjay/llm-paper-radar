"""Pipeline-canonical 'today' helper.

`today_utc()` returns the date that the pipeline should treat as its
target — interpreted as the **arxiv publication date** of the batch
being processed. Every source / pipeline stage reads from here so the
folder / digest / README title all line up with the papers' real
`<published>` field.

Resolution order (highest priority first):

1. `RADAR_TARGET_DATE=YYYY-MM-DD` — explicit pin, used by
   `daily.sh --date YYYY-MM-DD` for manual reruns of a specific
   publication day. Backfills should set this.
2. `RADAR_DAY_OFFSET=N` — subtract N UTC days from now. `daily.sh`
   sets `1` so the cron (fires at Beijing 14:00 = UTC 06:00) targets
   yesterday-UTC's complete batch.
3. Default — `datetime.now(UTC)` truncated to midnight.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta


def today_utc() -> datetime:
    """UTC midnight for the pipeline's notion of 'today' (= the target
    arxiv publication date). See module docstring for resolution order."""
    fixed = os.environ.get("RADAR_TARGET_DATE")
    if fixed:
        return datetime.fromisoformat(fixed).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
        )
    offset = int(os.environ.get("RADAR_DAY_OFFSET", "0"))
    return (datetime.now(UTC) - timedelta(days=offset)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
