"""Phase 2 — batch aggregation of windowed features.

Runs hourly: reads events from the offline store (Postgres/Parquet), computes
1h/1d/7d windows per user and per item, then writes:
  - Parquet to /opt/airflow/data/features/  -> training (offline)
  - Redis                                    -> serving (online)

The streaming consumer covers "right now"; this DAG covers the long windows that
aren't worth holding in the consumer's memory. They coexist: the online store has both.
"""
from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

WINDOWS = {"1h": 1, "1d": 24, "7d": 168}


@dag(
    dag_id="batch_features",
    schedule="@hourly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["features", "batch"],
)
def batch_features():

    @task
    def extract(data_interval_end=None) -> str:
        """Read events from the offline store up to the window end; return the raw parquet path."""
        raise NotImplementedError("phase 2: Kafka -> Postgres sink + extract goes here")

    @task
    def aggregate_user_features(raw_path: str) -> str:
        """impressions/clicks/CTR/category affinity over 1h, 1d, 7d."""
        raise NotImplementedError

    @task
    def aggregate_item_features(raw_path: str) -> str:
        """popularity, smoothed CTR, conversion rate accounting for the delayed reward."""
        raise NotImplementedError

    @task
    def push_to_redis(*feature_paths: str) -> None:
        """Write batch features to the online store (prefix `f:b:`) with a TTL."""
        raise NotImplementedError

    raw = extract()
    push_to_redis(aggregate_user_features(raw), aggregate_item_features(raw))


batch_features()
