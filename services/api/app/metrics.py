"""Phase 5 observability. Scrape-time metrics use their own Postgres/Kafka clients,
separate from the /recommend hot path. See docs/adr/0002-simulator-recommender-decoupling.md
for why recsys_simulated_stream_ctr is not recommendation CTR.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime

import psycopg
from confluent_kafka import Consumer, TopicPartition
from confluent_kafka.admin import AdminClient
# confluent-kafka 2.6.1 only exposes this under a private name.
from confluent_kafka.admin import _ConsumerGroupTopicPartitions as ConsumerGroupTopicPartitions
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from redis.asyncio import Redis

log = logging.getLogger("metrics")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")
EVENTS_TOPIC = os.getenv("EVENTS_TOPIC", "user_events")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "online-features")
PG_DSN = os.getenv("PG_DSN", "postgresql://rtrec:rtrec@localhost:5432/rtrec")
N_ITEMS = int(os.getenv("N_ITEMS", "2000"))

SERVED_ZSET = "obs:served:1h"

STAGE_SECONDS = Histogram(
    "recsys_recommend_stage_seconds", "Time spent per /recommend stage", ["stage"],
)
REQUESTS_TOTAL = Counter(
    "recsys_recommend_requests_total", "Total /recommend calls", ["cold_start"],
)
ANN_ELIGIBLE_TOTAL = Counter(
    "recsys_ann_eligible_total", "Requests where two-tower retrieval was eligible to run",
)
ANN_HIT_TOTAL = Counter(
    "recsys_ann_hit_total", "Of eligible requests, how many had a two-tower-sourced item in the final top-k",
)

CATALOG_COVERAGE = Gauge("recsys_catalog_coverage_ratio", "Distinct items served / total catalog size")
CATALOG_GINI = Gauge("recsys_catalog_concentration_gini", "Gini coefficient of served-recommendation counts")
CATALOG_TOP10_SHARE = Gauge("recsys_catalog_top10_share", "Share of served slots going to the 10 most-served items")
CONSUMER_LAG = Gauge("recsys_consumer_lag_messages", f"Offset lag for the '{CONSUMER_GROUP}' consumer group")
FEATURE_FRESHNESS = Gauge("recsys_feature_freshness_seconds", "Seconds since a sampled f:u:*:stats key was written")
SIMULATED_STREAM_CTR = Gauge("recsys_simulated_stream_ctr", "clicks/impressions in the raw event stream, last 1h")

_OFFLINE_METRICS_PATH = os.path.join(os.getenv("MODEL_DIR", "/models"), "ranker_metrics.json")
OFFLINE_AUC = Gauge("recsys_offline_ranker_auc", "LightGBM ranker validation AUC as of last training run")
OFFLINE_NDCG10 = Gauge("recsys_offline_ranker_ndcg10", "LightGBM ranker validation NDCG@10 as of last training run")
OFFLINE_TRAINED_AT = Gauge("recsys_offline_ranker_trained_at_timestamp", "Training run timestamp")


def _gini(counts: list[float]) -> float:
    if not counts:
        return 0.0
    values = sorted(counts)
    n = len(values)
    total = sum(values)
    if total == 0:
        return 0.0
    weighted_sum = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


async def _collect_catalog_metrics(r: Redis) -> None:
    served = await r.zrange(SERVED_ZSET, 0, -1, withscores=True)
    if not served:
        CATALOG_COVERAGE.set(0.0)
        CATALOG_GINI.set(0.0)
        CATALOG_TOP10_SHARE.set(0.0)
        return
    counts = [score for _, score in served]
    total = sum(counts)
    CATALOG_COVERAGE.set(min(len(served) / N_ITEMS, 1.0))
    CATALOG_GINI.set(_gini(counts))
    top10 = sum(sorted(counts, reverse=True)[:10])
    CATALOG_TOP10_SHARE.set(top10 / total if total else 0.0)


async def _collect_freshness(r: Redis) -> None:
    now = time.time()
    freshest = None
    cursor = 0
    scanned = 0
    while scanned < 200:
        cursor, keys = await r.scan(cursor, match="f:u:*:stats", count=50)
        for key in keys:
            ts_str = await r.hget(key, "last_seen_ts")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str).timestamp()
                    freshest = ts if freshest is None else max(freshest, ts)
                except ValueError:
                    continue
        scanned += len(keys)
        if cursor == 0:
            break
    FEATURE_FRESHNESS.set(now - freshest if freshest is not None else -1.0)


def _consumer_lag_sync() -> float:
    admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
    watermark_client = Consumer({"bootstrap.servers": KAFKA_BOOTSTRAP, "group.id": "metrics-lag-checker"})
    try:
        metadata = watermark_client.list_topics(EVENTS_TOPIC, timeout=5)
        partitions = list(metadata.topics[EVENTS_TOPIC].partitions.keys())

        offsets_future = admin.list_consumer_group_offsets([ConsumerGroupTopicPartitions(CONSUMER_GROUP)])
        committed = {}
        for group, result in offsets_future.items():
            for tp in result.result().topic_partitions:
                committed[tp.partition] = tp.offset

        total_lag = 0
        for p in partitions:
            low, high = watermark_client.get_watermark_offsets(
                TopicPartition(EVENTS_TOPIC, p), timeout=5, cached=False
            )
            committed_offset = committed.get(p, low)
            if committed_offset < 0:
                committed_offset = low
            total_lag += max(high - committed_offset, 0)
        return float(total_lag)
    finally:
        watermark_client.close()


def _simulated_stream_ctr_sync() -> float:
    with psycopg.connect(PG_DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                count(*) FILTER (WHERE event_type = 'click')::float
                / NULLIF(count(*) FILTER (WHERE event_type = 'impression'), 0)
            FROM events
            WHERE ts > now() - interval '1 hour'
        """)
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0


def _load_offline_metrics_sync() -> None:
    if not os.path.exists(_OFFLINE_METRICS_PATH):
        return
    with open(_OFFLINE_METRICS_PATH) as f:
        m = json.load(f)
    OFFLINE_AUC.set(m.get("auc", 0.0))
    OFFLINE_NDCG10.set(m.get("ndcg10", 0.0))
    OFFLINE_TRAINED_AT.set(m.get("trained_at", 0))


async def collect_extra_metrics(r: Redis) -> None:
    try:
        await _collect_catalog_metrics(r)
    except Exception:
        log.exception("catalog metrics collection failed")
    try:
        await _collect_freshness(r)
    except Exception:
        log.exception("feature freshness check failed")
    try:
        CONSUMER_LAG.set(await asyncio.to_thread(_consumer_lag_sync))
    except Exception:
        log.exception("consumer lag check failed")
    try:
        SIMULATED_STREAM_CTR.set(await asyncio.to_thread(_simulated_stream_ctr_sync))
    except Exception:
        log.exception("simulated stream ctr query failed")
    try:
        await asyncio.to_thread(_load_offline_metrics_sync)
    except Exception:
        log.exception("loading offline ranker metrics failed")


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
