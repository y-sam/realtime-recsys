"""Batch aggregation of windowed features.

Runs hourly. For each window (1h/1d/7d) it aggregates raw events in Postgres,
reshapes the result in pandas, then writes:

  - Parquet under data/features/  -> training (offline store)
  - Redis under the `f:b:` prefix -> serving (online store)

The `f:b:` prefix keeps these separate from the streaming consumer's `f:u:`/`f:i:`
keys. The consumer owns "right now" (fatigue counters, recent interactions); this DAG
owns long windows that aren't worth holding in the consumer's memory. Serving reads
both and never aggregates.

Aggregation runs in SQL on purpose -- see docs/adr/0001.
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import pandas as pd
import pendulum
import redis
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

log = logging.getLogger(__name__)

WINDOWS = {"1h": 1, "1d": 24, "7d": 168}
WIDEST_WINDOW_HOURS = max(WINDOWS.values())

SQL_DIR = Path(__file__).parent / "sql"
FEATURES_DIR = Path(os.getenv("FEATURES_DIR", "/opt/airflow/data/features"))
REDIS_URL = os.getenv("REDIS_URL", "redis://host.docker.internal:6379/0")
POSTGRES_CONN_ID = "postgres_offline"

# Batch features are refreshed hourly; expire them well after that so a failed run
# degrades serving gracefully instead of blanking the features outright.
BATCH_TTL = 3 * 24 * 3600

# Bayesian smoothing prior: pulls CTR of low-traffic items toward the global mean
# instead of letting 1 click on 2 impressions read as 50%.
CTR_PRIOR_CLICKS = 1.0
CTR_PRIOR_IMPRESSIONS = 20.0


def _read_sql(name: str, params: dict) -> pd.DataFrame:
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    sql = (SQL_DIR / name).read_text()
    # pandas 2.2 only recognizes a SQLAlchemy connectable when sqlalchemy>=2.0 is
    # importable, but Airflow 2.10 pins sqlalchemy<2.0 -- a raw psycopg2 connection
    # is therefore the only option here, not a legacy oversight. The resulting
    # UserWarning is expected and harmless; silence just that one warning.
    with hook.get_conn() as conn, warnings.catch_warnings():
        # pandas attributes this warning to our call site (via stacklevel), not to
        # pandas.io.sql, so the filter can't be scoped by module -- match the message.
        warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy")
        return pd.read_sql(sql, conn, params=params)


def _smoothed_ctr(clicks: pd.Series, impressions: pd.Series) -> pd.Series:
    return (clicks + CTR_PRIOR_CLICKS) / (impressions + CTR_PRIOR_IMPRESSIONS)


def _write_parquet(df: pd.DataFrame, name: str, window_end) -> str:
    out_dir = FEATURES_DIR / name / f"date={window_end.strftime('%Y-%m-%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{window_end.strftime('%H%M%S')}.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    return str(path)


@dag(
    dag_id="batch_features",
    schedule="@hourly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=5)},
    tags=["features", "batch"],
)
def batch_features():

    @task
    def user_features(data_interval_end=None) -> str:
        """One row per user, one column per (metric, window). Written as Parquet."""
        window_end = data_interval_end or pendulum.now("UTC")

        frames = []
        for label, hours in WINDOWS.items():
            df = _read_sql("user_features.sql", {"window_end": window_end, "hours": hours})
            if df.empty:
                continue
            df["ctr"] = _smoothed_ctr(df["clicks"], df["impressions"])
            df["cvr"] = _smoothed_ctr(df["purchases"], df["clicks"])
            metrics = ["impressions", "clicks", "add_to_carts", "purchases",
                       "revenue", "sessions", "distinct_items_clicked", "ctr", "cvr"]
            frames.append(df.set_index("user_id")[metrics].add_suffix(f"_{label}"))

        if not frames:
            raise ValueError(f"no events in any window ending {window_end}")

        out = pd.concat(frames, axis=1).fillna(0.0)

        # affinity: clicks per category over the widest window, pivoted wide
        aff = _read_sql("user_category_affinity.sql",
                        {"window_end": window_end, "hours": WIDEST_WINDOW_HOURS})
        if not aff.empty:
            wide = aff.pivot(index="user_id", columns="category", values="clicks")
            total = wide.sum(axis=1).replace(0, pd.NA)
            wide = wide.div(total, axis=0).fillna(0.0)   # normalized share, not raw counts
            out = out.join(wide.add_prefix("affinity_"), how="left").fillna(0.0)

        path = _write_parquet(out.reset_index(), "user_features", window_end)
        log.info("user features: %d rows, %d columns -> %s", len(out), out.shape[1], path)
        return path

    @task
    def item_features(data_interval_end=None) -> str:
        """One row per item, one column per (metric, window)."""
        window_end = data_interval_end or pendulum.now("UTC")

        frames, meta = [], None
        for label, hours in WINDOWS.items():
            df = _read_sql("item_features.sql", {"window_end": window_end, "hours": hours})
            if df.empty:
                continue
            df["ctr"] = _smoothed_ctr(df["clicks"], df["impressions"])
            if meta is None:
                meta = df.set_index("item_id")[["category", "price_tier"]]
            metrics = ["impressions", "clicks", "purchases", "revenue",
                       "distinct_users_clicked", "avg_position", "ctr"]
            frames.append(df.set_index("item_id")[metrics].add_suffix(f"_{label}"))

        if not frames:
            raise ValueError(f"no events in any window ending {window_end}")

        out = pd.concat(frames, axis=1).fillna(0.0)
        out = meta.join(out, how="right")

        # popularity rank over the widest window: a retrieval fallback that, unlike
        # the streaming ZSET, is stable and covers the long tail.
        out["popularity_rank"] = out["clicks_7d"].rank(ascending=False, method="min")

        path = _write_parquet(out.reset_index(), "item_features", window_end)
        log.info("item features: %d rows, %d columns -> %s", len(out), out.shape[1], path)
        return path

    @task
    def push_to_redis(user_path: str, item_path: str) -> dict:
        """Publish both feature sets to the online store under `f:b:`."""
        r = redis.from_url(REDIS_URL, decode_responses=True)
        written = {}

        for path, key_col, prefix in (
            (user_path, "user_id", "f:b:u"),
            (item_path, "item_id", "f:b:i"),
        ):
            df = pd.read_parquet(path)
            pipe = r.pipeline(transaction=False)
            for n, row in enumerate(df.to_dict("records"), start=1):
                key = f"{prefix}:{row.pop(key_col)}"
                pipe.hset(key, mapping={k: str(v) for k, v in row.items()})
                pipe.expire(key, BATCH_TTL)
                if n % 1000 == 0:      # bound the pipeline so memory stays flat
                    pipe.execute()
                    pipe = r.pipeline(transaction=False)
            pipe.execute()
            written[prefix] = len(df)

        log.info("pushed to redis: %s", written)
        return written

    push_to_redis(user_features(), item_features())


batch_features()
