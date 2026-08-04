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

Streams via a server-side cursor and writes Parquet incrementally, same fix and
same reasoning as build_dataset.py -- see that file for why DAYS_BACK is safe.
"""
from __future__ import annotations

import logging
import os
import resource

import pandas as pd
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_retrieval_dataset")

PG_DSN = os.getenv("PG_DSN", "postgresql://rtrec:rtrec@localhost:5432/rtrec").replace(
    "postgresql+psycopg://", "postgresql://"
)
OUT_PATH = os.getenv("RETRIEVAL_SET_PATH", "/work/data/training/retrieval.parquet")
DAYS_BACK = os.getenv("DAYS_BACK")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200000"))

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
WHERE e.event_type = 'impression' AND e.impression_id IS NOT NULL
{"AND e.ts > now() - make_interval(days => %(days_back)s)" if DAYS_BACK else ""}
ORDER BY e.ts;
"""


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    log.info("querying point-in-time retrieval features from %s (days_back=%s, batch_size=%d)",
              PG_DSN.split("@")[-1], DAYS_BACK or "all", BATCH_SIZE)

    params = {"days_back": int(DAYS_BACK)} if DAYS_BACK else {}
    writer = None
    schema = None
    total_rows = 0
    total_positive = 0

    with psycopg.connect(PG_DSN) as conn, conn.cursor(name="build_retrieval_dataset") as cur:
        cur.execute(QUERY, params)
        colnames = [d.name for d in cur.description]
        while True:
            rows = cur.fetchmany(BATCH_SIZE)
            if not rows:
                break
            batch_df = pd.DataFrame(rows, columns=colnames)
            table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(OUT_PATH, schema)
            writer.write_table(table)
            total_rows += len(batch_df)
            total_positive += int(batch_df["label"].sum())
            log.info("wrote batch: %d rows so far", total_rows)

    if writer is not None:
        writer.close()

    pct_positive = 100 * total_positive / total_rows if total_rows else 0.0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    log.info("wrote %d rows (%.3f%% positive) -> %s", total_rows, pct_positive, OUT_PATH)
    log.info("peak RSS: %.1f MB", peak_rss_mb)


if __name__ == "__main__":
    main()
