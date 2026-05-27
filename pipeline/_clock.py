"""Pipeline-canonical 'today' helper.

The cron job fires at Beijing 14:00 (= UTC 06:00) and runs the pipeline
against **yesterday UTC** — by that hour arxiv has finished publishing
the previous calendar day's batch, so a Beijing-noon-ish window lands
on a complete dataset.

`scripts/daily.sh` sets `RADAR_DAY_OFFSET=1` so all sources / pipeline
stages treat "today" as "yesterday UTC". Manual CLI invocations without
the env var keep the literal-now behavior, which is what one-off
backfills usually want.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta


def today_utc() -> datetime:
    """UTC midnight for the pipeline's notion of 'today'. Respects the
    `RADAR_DAY_OFFSET` env var (integer days to subtract). Default 0 =
    real now."""
    offset = int(os.environ.get("RADAR_DAY_OFFSET", "0"))
    return (datetime.now(UTC) - timedelta(days=offset)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
