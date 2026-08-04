"""Kafka -> Postgres sink: persists raw events as the training source of truth.

Delivery semantics: at-least-once with idempotent writes.

    consume batch -> COPY into UNLOGGED staging -> INSERT ... ON CONFLICT DO NOTHING
                  -> COMMIT (Postgres) -> commit offsets (Kafka)

Offsets are committed only after the database transaction succeeds. A crash
anywhere before that replays the batch; the primary key on (event_id, ts) makes
the replay a no-op. Committing offsets first would silently drop events — tolerable
for the online feature store, not for the dataset a model is trained on.

Why COPY into staging instead of executemany INSERT: COPY is an order of magnitude
faster, but it cannot express ON CONFLICT. Staging gives us both — bulk load speed
and idempotency — at the cost of one extra table.

This runs as a separate consumer group from the online feature consumer, so the
sink can lag, crash, or replay history without affecting serving latency.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone

import psycopg
from confluent_kafka import Consumer, KafkaError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sink")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")
TOPIC = os.getenv("EVENTS_TOPIC", "user_events")
GROUP = os.getenv("CONSUMER_GROUP", "postgres-sink")
PG_DSN = os.getenv("PG_DSN", "postgresql://rtrec:rtrec@localhost:5432/rtrec")

BATCH_SIZE = int(os.getenv("SINK_BATCH_SIZE", 500))
BATCH_TIMEOUT_S = float(os.getenv("SINK_BATCH_TIMEOUT_S", 5.0))
RETENTION_DAYS = int(os.getenv("SINK_RETENTION_DAYS", 30))

COLUMNS = (
    "event_id", "event_type", "user_id", "item_id", "session_id", "surface",
    "device", "position", "is_new_user", "category", "price_tier", "value",
    "impression_id", "ts",
)

_running = True


def _stop(*_):
    global _running
    _running = False


def to_row(ev: dict) -> tuple:
    """Kafka JSON -> tuple in COLUMNS order. Raises on anything malformed."""
    return (
        ev["event_id"],
        ev["event_type"],
        ev["user_id"],
        ev["item_id"],
        ev["session_id"],
        ev["surface"],
        ev["device"],
        int(ev["position"]),
        bool(ev["is_new_user"]),
        ev["category"],
        int(ev["price_tier"]),
        float(ev.get("value", 0.0)),
        ev.get("impression_id"),   # None for events published before this column existed
        datetime.fromisoformat(ev["ts"]),
    )


def write_batch(conn: psycopg.Connection, rows: list[tuple]) -> int:
    """One transaction: COPY to staging, upsert into events, truncate staging."""
    cols = ", ".join(COLUMNS)
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS events_staging "
            f"(LIKE events INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
        )
        with cur.copy(f"COPY events_staging ({cols}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
        cur.execute(
            f"INSERT INTO events ({cols}) SELECT {cols} FROM events_staging "
            f"ON CONFLICT (event_id, ts) DO NOTHING"
        )
        inserted = cur.rowcount
    conn.commit()
    return inserted


def maintain_partitions(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT ensure_event_partitions(7)")
        cur.execute("SELECT drop_old_event_partitions(%s)", (RETENTION_DAYS,))
    conn.commit()
    log.info("partitions ensured (+7d) and pruned (keep %dd)", RETENTION_DAYS)


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    conn = psycopg.connect(PG_DSN, autocommit=False)
    maintain_partitions(conn)
    last_maintenance = time.time()

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP,
        "auto.offset.reset": "earliest",   # the sink wants history, unlike the feature consumer
        "enable.auto.commit": False,       # we commit only after Postgres commits
    })
    consumer.subscribe([TOPIC])
    log.info("sink connected: %s topic=%s group=%s", KAFKA_BOOTSTRAP, TOPIC, GROUP)

    total, t0 = 0, time.time()
    while _running:
        msgs = consumer.consume(num_messages=BATCH_SIZE, timeout=BATCH_TIMEOUT_S)
        if not msgs:
            continue

        rows, malformed = [], 0
        for msg in msgs:
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.warning("consume error: %s", msg.error())
                continue
            try:
                rows.append(to_row(json.loads(msg.value())))
            except Exception as exc:
                malformed += 1
                log.warning("skipping malformed event: %s", exc)

        if rows:
            try:
                inserted = write_batch(conn, rows)
            except Exception:
                conn.rollback()
                log.exception("batch failed, offsets not committed; will replay")
                time.sleep(2)
                continue

            consumer.commit(asynchronous=False)  # only now is the batch durable
            total += inserted
            if malformed:
                log.warning("batch had %d malformed events", malformed)

        if time.time() - last_maintenance > 3600:
            maintain_partitions(conn)
            last_maintenance = time.time()

        if total and total % 5000 < len(rows):
            log.info("rows persisted: %d (%.1f/s)", total, total / max(time.time() - t0, 1e-6))

    log.info("shutting down")
    consumer.close()
    conn.close()


if __name__ == "__main__":
    main()
