"""Builds the click-ranking training set from raw events in Postgres.

One row per impression, labeled by whether it was clicked (via impression_id).
Features use window functions bounded to ROWS BETWEEN UNBOUNDED PRECEDING AND
1 PRECEDING so nothing sees its own outcome or the future.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_dataset")

PG_DSN = os.getenv("PG_DSN", "postgresql+psycopg://rtrec:rtrec@localhost:5432/rtrec")
OUT_PATH = os.getenv("TRAINING_SET_PATH", "/work/data/training/impressions.parquet")

QUERY = """
WITH base AS (
    SELECT event_id, event_type, user_id, item_id, category, position,
           is_new_user, price_tier, impression_id, session_id, ts
    FROM events
    WHERE event_type IN ('impression', 'click')
),
enriched AS (
    SELECT
        event_id, event_type, user_id, item_id, category, position,
        is_new_user, price_tier, impression_id, session_id, ts,
        COUNT(*) FILTER (WHERE event_type = 'click')      OVER item_w      AS item_clicks_before,
        COUNT(*) FILTER (WHERE event_type = 'impression') OVER item_w      AS item_impressions_before,
        COUNT(*) FILTER (WHERE event_type = 'click')      OVER user_cat_w  AS user_category_clicks_before,
        COUNT(*) FILTER (WHERE event_type = 'click')      OVER user_w      AS user_clicks_before,
        COUNT(*) FILTER (WHERE event_type = 'impression') OVER user_item_w AS user_item_impressions_before
    FROM base
    WINDOW
        item_w      AS (PARTITION BY item_id             ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        user_cat_w  AS (PARTITION BY user_id, category    ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        user_w      AS (PARTITION BY user_id              ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        user_item_w AS (PARTITION BY user_id, item_id     ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    e.ts, e.user_id, e.item_id, e.session_id::text AS session_id, e.category, e.position,
    e.is_new_user, e.price_tier,
    e.item_clicks_before, e.item_impressions_before,
    e.user_category_clicks_before, e.user_clicks_before, e.user_item_impressions_before,
    (c.event_id IS NOT NULL)::int AS label
FROM enriched e
LEFT JOIN events c ON c.impression_id = e.event_id AND c.event_type = 'click'
WHERE e.event_type = 'impression' AND e.impression_id IS NOT NULL
ORDER BY e.ts;
"""


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    log.info("querying point-in-time features from %s", PG_DSN.split("@")[-1])
    engine = create_engine(PG_DSN)
    try:
        df = pd.read_sql(QUERY, engine)
    finally:
        engine.dispose()

    df.to_parquet(OUT_PATH, index=False)
    log.info("wrote %d rows (%.3f%% positive) -> %s",
              len(df), 100 * df["label"].mean(), OUT_PATH)


if __name__ == "__main__":
    main()
