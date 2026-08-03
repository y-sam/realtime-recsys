# Real-time Recommendation Engine

A low-latency recommendation system fed by a continuous stream of synthetic user
events. Not a notebook: a running service with an online/offline feature store,
orchestrated batch training, and observability.

**Demo:** `https://<host>/recommend?user_id=u_42&k=10` · **Event console:** `:8080`

## Architecture

```
                  ┌──────────────┐
                  │  simulator   │  Poisson-spaced sessions, fatigue,
                  │  (Python)    │  cold-start, delayed reward
                  └──────┬───────┘
                         │ produce (key = user_id)
                  ┌──────▼───────┐
                  │   Redpanda   │  topic: user_events (3 partitions)
                  └──┬────────┬──┘
         streaming   │        │   batch sink
            ┌────────▼──┐  ┌──▼──────────┐
            │ consumer  │  │  Postgres   │  offline store (training)
            │  Kafka→   │  └──┬──────────┘
            │  Redis    │     │  ┌──────────────┐
            └────┬──────┘     └─►│   Airflow    │ 1h/1d/7d windows,
                 │               │   (batch)    │ retraining, push to Redis
            ┌────▼──────┐ ◄──────└──────────────┘
            │   Redis   │  online store: recent items, affinity, fatigue, CTR, top-1h
            └────┬──────┘
            ┌────▼──────┐
            │  FastAPI  │  retrieval → ranking → rerank   (target p50 < 100ms)
            └───────────┘
```

## What the stream models, and why

Every property of the simulator exists because the system has to handle it:

| Stream property | Problem it forces you to solve |
|---|---|
| Zipf popularity | retrieval can't just be top-popular |
| Latent user×category affinity | there is real signal for the two-tower to learn |
| ~5% of sessions from new users | cold-start is continuous, not an edge case |
| Re-impressions decay CTR | fatigue modeling in the rerank stage |
| `purchase` arrives ~45s after the click | delayed reward: the label doesn't exist at serving time |
| Position bias in the slate | ranking must discount position |

## Running locally

```bash
cp .env.example .env
make dev        # brings everything up + Redpanda Console on :8080
make topics
make smoke      # watch 10 raw events flow through
make rows       # count what landed in the offline store
make rec        # call the serving endpoint
make airflow-up # Airflow on :8081 (only when working on pipelines)
```

## Roadmap

- [x] **1. Data + simulator** — continuous event stream into Redpanda
- [x] **2. Feature pipeline** — Kafka→Postgres sink, 1h/1d/7d window DAGs, push to Redis
- [ ] **3. Models** — two-tower retrieval (PyTorch) + LightGBM ranking; cold-start via context features
- [ ] **4. Serving** — swap the heuristic ranker for the model, ANN retrieval, p50 < 100ms
- [ ] **5. Observability** — p50/p95 latency, simulated CTR, catalog coverage and distribution

## Infrastructure decisions (trade-offs)

**A single VPS running `docker compose`, not one PaaS per service.** Spreading Kafka +
Redis + Postgres + Airflow + API across separate free tiers costs more and breaks more:
Upstash Kafka was discontinued in March 2025, Redpanda Serverless is a 30-day trial with
$100 in credits and then pay-as-you-go, Railway and Fly.io no longer offer a permanent
free tier, and Render's free tier sleeps after 15 minutes (30–60s cold start — a
non-starter for a latency demo). One ~8GB host runs the whole stack for less than the
sum of the add-ons.

**Redpanda instead of Kafka.** Identical Kafka API, no ZooKeeper/KRaft to operate, and
~512MB of RAM instead of ~2GB. The producer and consumer code is plain `confluent-kafka`:
moving to MSK or Confluent is a `bootstrap.servers` change.

**Airflow in a separate compose file.** It's the heaviest component and doesn't need the
same uptime as the API. Bring it up on demand, run the DAG, bring it down.

**At-least-once into the offline store, at-most-once tolerated online.** The sink
commits Kafka offsets only after the Postgres transaction lands, and writes are made
idempotent by a primary key on `(event_id, ts)` — a replayed batch is a no-op. The
online feature consumer runs a separate group with auto-commit: losing a few
impression counts on a crash is acceptable there, losing training rows is not.

**Aggregate in SQL, transform in pandas.** Memory is the binding constraint on a
single host, so windowed `GROUP BY` runs in Postgres and only the aggregate crosses
into the worker; reshaping, smoothing, and serialization happen in pandas where they
are readable and testable. Full reasoning in [ADR 0001](docs/adr/0001-aggregate-in-sql-transform-in-pandas.md).

**Streaming and batch share one online store.** The consumer covers "right now" (fatigue,
recent interactions); Airflow covers long windows. The API doesn't know the difference —
it reads everything from Redis with `GET`/`HGETALL`, with no aggregation in the request path.
