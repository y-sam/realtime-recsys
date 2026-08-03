"""Real-time event simulator -> Kafka/Redpanda topic.

Generates user sessions at random intervals (Poisson process) and publishes events
shaped like:

    {
      "event_id": "...", "event_type": "impression|click|add_to_cart|purchase",
      "user_id": "u_42", "item_id": "i_00031", "session_id": "...",
      "surface": "home_feed", "device": "ios", "position": 3,
      "is_new_user": false, "category": "drama", "price_tier": 2,
      "value": 0.0, "ts": "2026-08-02T12:00:00.123456+00:00"
    }

Purchases are scheduled into the future, so the stream carries a genuine delayed reward.
"""
from __future__ import annotations

import heapq
import json
import logging
import random
import signal
import time
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer

from app.config import settings
from app.world import SURFACES, World

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("simulator")

_running = True


def _stop(*_):  # graceful shutdown
    global _running
    _running = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_event(event_type: str, user, item, session_id: str, surface: str,
               position: int, value: float = 0.0) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "user_id": user.user_id,
        "item_id": item.item_id,
        "session_id": session_id,
        "surface": surface,
        "device": user.device,
        "position": position,
        "is_new_user": user.is_new,
        "category": item.category,
        "price_tier": item.price_tier,
        "value": value,
        "ts": now_iso(),
    }


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    rng = random.Random(settings.seed)
    world = World(settings.n_users, settings.n_items, seed=settings.seed)
    producer = Producer({
        "bootstrap.servers": settings.kafka_bootstrap,
        "linger.ms": 50,
        "compression.type": "lz4",
        "client.id": "event-simulator",
    })

    def publish(event: dict) -> None:
        producer.produce(
            topic=settings.events_topic,
            key=event["user_id"].encode(),   # partition by user: per-user ordering guaranteed
            value=json.dumps(event).encode(),
        )
        producer.poll(0)

    delayed: list[tuple[float, dict]] = []   # (timestamp, event) heap for delayed rewards
    interval = 1.0 / max(settings.events_per_second, 0.1)
    emitted = 0
    t0 = time.time()

    log.info("simulator started: %s events/s -> %s (topic=%s)",
             settings.events_per_second, settings.kafka_bootstrap, settings.events_topic)

    while _running:
        # 1) flush any delayed conversions that are now due
        now = time.time()
        while delayed and delayed[0][0] <= now:
            _, ev = heapq.heappop(delayed)
            ev["ts"] = now_iso()
            publish(ev)
            emitted += 1

        # 2) one session: the user is shown a slate of items
        user = world.pick_user(settings.new_user_rate)
        session_id = str(uuid.uuid4())
        surface = rng.choice(SURFACES)
        slate = world.pick_items(k=rng.randint(3, 10))

        for pos, item in enumerate(slate):
            p_click = world.click_probability(user, item, settings.base_ctr, settings.fatigue_decay)
            p_click *= 1.0 / (1.0 + 0.15 * pos)          # position bias
            publish(make_event("impression", user, item, session_id, surface, pos))
            world.register_impression(user, item)
            emitted += 1

            if rng.random() < p_click:
                publish(make_event("click", user, item, session_id, surface, pos))
                emitted += 1

                if rng.random() < 0.35:
                    publish(make_event("add_to_cart", user, item, session_id, surface, pos))
                    emitted += 1

                if rng.random() < settings.purchase_given_click:
                    delay = rng.expovariate(1.0 / settings.reward_delay_s)
                    value = round(rng.uniform(5, 40) * item.price_tier, 2)
                    ev = make_event("purchase", user, item, session_id, surface, pos, value=value)
                    heapq.heappush(delayed, (time.time() + delay, ev))

        if emitted and emitted % 500 < len(slate):
            log.info("events published: %d (%.1f/s) | pending rewards: %d",
                     emitted, emitted / max(time.time() - t0, 1e-6), len(delayed))

        time.sleep(rng.expovariate(1.0 / interval))  # random inter-arrival times (Poisson)

    log.info("shutting down, flushing producer...")
    producer.flush(10)


if __name__ == "__main__":
    main()
