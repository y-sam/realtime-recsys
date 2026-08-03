"""Kafka -> Redis consumer: materializes the low-latency online features.

Keys written to Redis (namespace `f:`):

    f:u:{user_id}:recent        LIST   last N interacted item_ids (retrieval context)
    f:u:{user_id}:cat           HASH   category -> click count (online affinity)
    f:u:{user_id}:seen:{item}   STRING impression count, 6h TTL  (fatigue)
    f:u:{user_id}:stats         HASH   impressions/clicks/purchases/last_seen_ts
    f:i:{item_id}:stats         HASH   impressions/clicks (online CTR -> ranking + popularity)
    f:pop:1h                    ZSET   top items by clicks in the last hour (cold-start fallback)

Golden rule: serving NEVER aggregates. It only issues GET/HGETALL.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time

import redis
from confluent_kafka import Consumer, KafkaError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("consumer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")
TOPIC = os.getenv("EVENTS_TOPIC", "user_events")
GROUP = os.getenv("CONSUMER_GROUP", "online-features")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

RECENT_MAX = 50
SEEN_TTL = 6 * 3600
USER_TTL = 30 * 24 * 3600

_running = True


def _stop(*_):
    global _running
    _running = False


def apply_event(pipe: redis.client.Pipeline, ev: dict) -> None:
    uid, iid = ev["user_id"], ev["item_id"]
    etype = ev["event_type"]
    ukey = f"f:u:{uid}"

    if etype == "impression":
        pipe.hincrby(f"{ukey}:stats", "impressions", 1)
        pipe.hincrby(f"f:i:{iid}:stats", "impressions", 1)
        seen = f"{ukey}:seen:{iid}"
        pipe.incr(seen)
        pipe.expire(seen, SEEN_TTL)

    elif etype == "click":
        pipe.hincrby(f"{ukey}:stats", "clicks", 1)
        pipe.hincrby(f"f:i:{iid}:stats", "clicks", 1)
        pipe.hincrby(f"{ukey}:cat", ev["category"], 1)
        pipe.lpush(f"{ukey}:recent", iid)
        pipe.ltrim(f"{ukey}:recent", 0, RECENT_MAX - 1)
        pipe.zincrby("f:pop:1h", 1, iid)

    elif etype in ("add_to_cart", "purchase"):
        pipe.hincrby(f"{ukey}:stats", etype, 1)
        if etype == "purchase":
            pipe.hincrbyfloat(f"{ukey}:stats", "revenue", float(ev.get("value", 0.0)))

    pipe.hset(f"{ukey}:stats", "last_seen_ts", ev["ts"])
    pipe.expire(f"{ukey}:stats", USER_TTL)
    pipe.expire(f"{ukey}:recent", USER_TTL)
    pipe.expire(f"{ukey}:cat", USER_TTL)


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    r = redis.from_url(REDIS_URL, decode_responses=True)
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC])
    log.info("consumer connected: %s topic=%s group=%s", KAFKA_BOOTSTRAP, TOPIC, GROUP)

    processed, t0 = 0, time.time()
    while _running:
        msgs = consumer.consume(num_messages=200, timeout=1.0)
        if not msgs:
            continue

        pipe = r.pipeline(transaction=False)
        for msg in msgs:
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.warning("consume error: %s", msg.error())
                continue
            try:
                apply_event(pipe, json.loads(msg.value()))
                processed += 1
            except Exception as exc:  # a malformed event must not kill the consumer
                log.warning("dropped event: %s", exc)
        pipe.execute()

        if processed % 2000 < len(msgs):
            log.info("events processed: %d (%.1f/s)", processed, processed / max(time.time() - t0, 1e-6))

    consumer.close()


if __name__ == "__main__":
    main()
