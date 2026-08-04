# Real-time Recommendation Engine

A low-latency recommendation system fed by a continuous stream of synthetic user
events. Not a notebook: a running service with an online/offline feature store,
orchestrated batch training, and observability.

**Demo:** `https://<host>/recommend?user_id=u_42&k=10` (placeholder — not deployed publicly
yet, see [DEPLOY.md](DEPLOY.md)) · **Showcase UI:** `:8501` · **Event console:** `:8080` · **Metrics:** `:3000`

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
            │  FastAPI  │  retrieval → ranking → rerank*  (target p50 < 100ms)
            └───────────┘
```
<sub>*rerank is currently a measured no-op — final truncation/assembly only, no diversity or business-rule logic yet.</sub>



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
make dev        # brings everything up + Redpanda Console :8080, Grafana :3000, showcase UI :8501
make topics
make smoke      # watch 10 raw events flow through
make rows       # count what landed in the offline store
make rec        # call the serving endpoint
make airflow-up # Airflow on :8081 (only when working on pipelines)
```

`docker-compose.yml` pins `name: rtrec`, so every checkout on the same host shares the
same Postgres/Redpanda volumes regardless of which directory `make dev` runs from —
convenient for restarting the same deployment, but it means a second checkout on a host
that already has one running attaches to that data instead of starting empty. A genuinely
new host starts empty as expected.

## Roadmap

- [x] **1. Data + simulator** — continuous event stream into Redpanda
- [x] **2. Feature pipeline** — Kafka→Postgres sink, 1h/1d/7d window DAGs, push to Redis
- [x] **3. Models** — LightGBM ranking and two-tower retrieval (PyTorch), both trained on
      point-in-time-correct features and validated offline before promotion. Two "weak
      result" claims from an earlier pass were both wrong, for two unrelated reasons —
      retrieval's training objective and cold-start's evaluation split. See the retrieval
      and cold-start notes below, and
      [ADR 0003](docs/adr/0003-two-tower-catalog-scale-investigation.md) for the full
      investigation. `training/build_dataset.py` and `build_retrieval_dataset.py` stream
      via a server-side cursor and write Parquet incrementally (peak RSS ~614MB, was
      OOMing a 3GB container at ~4.2M rows before the fix).
- [x] **4. Serving** — the LightGBM ranker drives ranking for warm users (`RANKING_MODE`,
      reversible without a deploy). Retrieval for warm users defaults to
      `RETRIEVAL_MODE=two_tower_only` — chosen by measuring candidate-set recall across
      all three options against real held-out clicks, not by narrative; see the
      retrieval note below for the numbers. Retrieval is exact brute-force search over
      the ~2k-item catalog, not an ANN index — unnecessary at this scale.
      **Service p50/p95 (on-host, no TLS/proxy):** `<pending production measurement,
      see DEPLOY.md §6a>`. **End-to-end p50/p95 (public URL, includes network + TLS +
      nginx, measured from `<location>`):** `<pending production measurement, see
      DEPLOY.md §6b>`. Model artifacts (`models/`) are version-pinned in git rather than
      generated at boot, so a fresh clone always serves the exact model the code was
      validated against — `GET /health` reports the loaded ranker/two-tower hash so
      you can confirm which artifact is live without opening the container.
- [x] **5. Observability** — p50/p95/p99 `/recommend` latency by stage, catalog
      coverage/concentration, retrieval (two-tower hit rate) and pipeline health
      (consumer lag, feature freshness), offline ranker metrics (AUC, NDCG@10). Grafana dashboard on
      `:3000` (`make dev`). See [ADR 0002](docs/adr/0002-simulator-recommender-decoupling.md)
      for why "simulated CTR" is explicitly not recommendation CTR yet.
- [ ] **6. Closed-loop online evaluation** — have the simulator call `/recommend` and
      simulate clicks against the returned slate instead of sampling independently,
      enabling a real online model-vs-heuristic CTR comparison. Planned, not built:
      it's a bigger architectural change than the rest of observability, and depends on
      having a model worth comparing online in the first place. See ADR 0002.

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

**The simulator never calls `/recommend`.** It samples independently via `world.py`'s
Zipf/affinity model, deliberately decoupled from anything this system recommends — that's
what keeps every offline evaluation in `training/` (AUC, NDCG@10, recall@K) honest: the
ground truth doesn't know what the model would have done. The cost is that there's no
live recommendation-CTR signal yet, only an environment-baseline one. Full reasoning,
and the condition for closing that loop, in
[ADR 0002](docs/adr/0002-simulator-recommender-decoupling.md).

**Two-tower retrieval was broken, not catalog-limited — and fixing it changed how
retrieval is configured.** The original recall@K table only compared the model against
random retrieval, which made "real but modest" sound like a catalog-size ceiling. Adding
a popularity baseline (top-K by impression count, no personalization at all) showed the
model was statistically indistinguishable from random — mean percentile rank of the
actual clicked item among 2,000 candidates was 0.4857, where 0.5 is chance — while
popularity alone beat it 3.8-11x at every K. The cause was the training objective:
pointwise BCE against ~4% CTR negatives has almost no pressure to rank candidates
against each other, which is exactly what recall@K needs. Rewriting the objective as
in-batch sampled softmax (logQ correction for popularity-biased in-batch negatives,
duplicate-item masking for a catalog this concentrated, towers and features otherwise
unchanged) fixed it:

| K | model | popularity | random |
|---|---|---|---|
| 50 | 0.3008 | 0.3006 | 0.0250 |
| 100 | 0.4118 | 0.4025 | 0.0500 |
| 200 | 0.5329 | 0.5221 | 0.1000 |

Positive-control rank (the number that matters most, since it doesn't depend on a
choice of K): mean 0.4857 → **0.1977**, median **0.0845**. K=50 is reported as a tie
(+0.0002 is inside the noise of an unseeded training run), not rounded up to a win; K=100
and K=200 show a real margin. Full path in
[ADR 0003](docs/adr/0003-two-tower-catalog-scale-investigation.md).

That result changed the retrieval config, not just the model: measuring candidate-set
recall (does the retrieved pool contain the item the user actually clicked, matching
what serving really does) across the three ways `RETRIEVAL_MODE` can build that pool —
popularity-only: 0.5221, blended (old default): 0.5239, two-tower-only: **0.5329** —
showed the "blended" default was barely different from popularity-only in practice: its
average candidate pool was only ~201 items against a ~250 budget, because ~49 of the
model's top 50 picks were already in the popularity-200 list at this catalog's
concentration. The union wasn't adding much to blend. `RETRIEVAL_MODE=two_tower_only`
is now the default for warm users; cold-start is unaffected, since the model was never
eligible there.

Two-tower beating popularity is only established at this catalog's size (2,000 items).
Whether the margin holds, grows, or shrinks at a larger catalog is a real question —
this environment just isn't the right size to answer it. That's recorded as an open
question, not a pending task: regenerating at scale (`generate_dataset.py`, ~21M rows)
was considered and deliberately not run, because there was no working model to test the
hypothesis against until the objective fix above landed. Worth revisiting now that there
is one, but not started.

**Cold-start does have validated signal — an earlier claim that it didn't was wrong,
and wrong for a specific, fixable reason.** The original "neither model showed
validated signal for users with no history" came from splitting on `is_new_user`,
which turned out to be a permanent per-user label, not a cold-start indicator:
`services/simulator/app/world.py` set it once when a user was created and never
updated it, so a user spawned as "new" hours earlier still carried `is_new_user=true`
on every event since, no matter how much history they'd built up by then. In the
validation set that split, that flag was `true` on 84% of rows — nowhere near the
~5% new-user session rate the simulator targets. Re-measured on an actual behavioral
cold-start proxy (`user_clicks_before == 0` at scoring time, matching the API's own
`cold_start` threshold), the LightGBM ranker beats a pure popularity baseline even at
zero prior clicks (AUC 0.615 vs 0.535), and by more once a user has some history (0.798
vs 0.637 at 20+ prior clicks). Fixed at the source: `World.User` now tracks real
lifetime impression count, and `is_new_user` on emitted events decays with it instead
of staying stuck at its creation-time value. Promoting cold-start scoring to serving
is still a separate, not-yet-made decision — offline AUC on historical rows isn't the
same as validating it against the live candidate pool — but it's no longer blocked by
a claim that turned out to be a measurement bug, not a finding.

**The `is_new_user` fix's effect on LightGBM's own metrics is inconclusive, and stays
that way for now.** Retraining right after the `world.py` fix moved AUC/logloss/NDCG@10
by amounts too small to attribute to anything (0.7074→0.7071, 0.1384→0.1440,
0.6492→0.6522) — because the fix only changes newly-generated events, and the training
set was still ~95% pre-fix data at retrain time. A clean before/after read needs enough
post-fix traffic to actually dominate the training window, which takes real wall-clock
time to accumulate, not a rerun. Declaring this open rather than chasing a number that
isn't measurable yet.
