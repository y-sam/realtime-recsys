"""Builds the two-tower retrieval training set from raw events in Postgres.

Same point-in-time discipline as build_dataset.py (window functions, nothing sees
its own outcome or the future) but a different feature split. A two-tower model
can only use features available to ONE side alone -- the user embedding is computed
once per request and dotted against precomputed item embeddings, so nothing that
requires knowing BOTH the user and the specific candidate item (e.g. fatigue:
"has this user seen this exact item before") can live in either tower. That signal
stays in the ranking model (training/train_ranker.py), which sees the full
candidate set and can afford per-pair features. Retrieval's job is a good candidate
SET, not the final order.

The user tower gets a full per-category affinity vector (one share per category,
not just the single category of "this" row) so it doesn't need to know the
candidate's category in advance -- CATEGORIES below must match world.py's fixed
catalog; if the simulator's category list changes, this does too.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_retrieval_dataset")

PG_DSN = os.getenv("PG_DSN", "postgresql+psycopg://rtrec:rtrec@localhost:5432/rtrec")
OUT_PATH = os.getenv("RETRIEVAL_SET_PATH", "/work/data/training/retrieval.parquet")

# Must match world.py's CATEGORIES list.
CATEGORIES = ["action", "comedy", "drama", "docu", "sports",
              "music", "kids", "horror", "reality", "news"]

_category_share_cols = ",\n        ".join(
    f"COUNT(*) FILTER (WHERE event_type = 'click' AND category = '{c}') OVER user_w "
    f"AS user_clicks_{c}_before"
    for c in CATEGORIES
)

QUERY = f"""
WITH base AS (
    SELECT event_id, event_type, user_id, item_id, category, position,
           is_new_user, price_tier, impression_id, ts
    FROM events
    WHERE event_type IN ('impression', 'click')
),
enriched AS (
    SELECT
        event_id, event_type, user_id, item_id, category, is_new_user, price_tier,
        impression_id, ts,
        COUNT(*) FILTER (WHERE event_type = 'click')      OVER item_w AS item_clicks_before,
        COUNT(*) FILTER (WHERE event_type = 'impression') OVER item_w AS item_impressions_before,
        COUNT(*) FILTER (WHERE event_type = 'click')      OVER user_w AS user_clicks_before,
        {_category_share_cols}
    FROM base
    WINDOW
        item_w AS (PARTITION BY item_id  ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
        user_w AS (PARTITION BY user_id  ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    e.ts, e.user_id, e.item_id, e.category, e.is_new_user, e.price_tier,
    e.item_clicks_before, e.item_impressions_before, e.user_clicks_before,
    {", ".join(f"e.user_clicks_{c}_before" for c in CATEGORIES)},
    (c.event_id IS NOT NULL)::int AS label
FROM enriched e
LEFT JOIN events c ON c.impression_id = e.event_id AND c.event_type = 'click'
-- same reasoning as build_dataset.py: only rows after impression_id existed have a
-- trustworthy label.
WHERE e.event_type = 'impression' AND e.impression_id IS NOT NULL
ORDER BY e.ts;
"""


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    log.info("querying point-in-time retrieval features from %s", PG_DSN.split("@")[-1])
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
