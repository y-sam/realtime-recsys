# Real-time Recommendation Engine

A low-latency recommendation system fed by a continuous stream of synthetic user
events. Not a notebook: a running service with an online/offline feature store,
orchestrated batch training, and observability.

## The simulated business

**The scenario.** This recommends titles in a paid content marketplace: users
browse a catalog of films and documentaries available to buy or rent
individually, see a personalized slate, and the business earns when the right
title reaches the right user before they buy. Every field in the event schema
follows from that: `price_tier` is a rental/purchase price band, and
`add_to_cart` / `purchase` are real transaction steps, not stand-ins for
something else.

**The catalog.** ~2,000 items across 10 genres (action, comedy, drama, docu,
sports, music, kids, horror, reality, news), each with a `price_tier` (1-4)
independent of genre or popularity. Popularity follows a Zipf distribution
(`1/rank^0.9`), so a small head of titles takes most of the traffic and the
long tail is real rather than padding. Each user carries a latent per-genre
affinity — 2-4 favorite genres weighted 1.5-4x higher than the rest — which is
the signal the two-tower model exists to learn.

**The event stream.** `impression → click → add_to_cart → purchase`, each
event carrying the `impression_id` of the exact slot that produced it. An
impression is a title shown in a slate; a click opens its detail page;
`add_to_cart` adds it to a cart pre-checkout; `purchase` is the completed
transaction, `value` set from `price_tier`. Purchase lands ~45s (mean) after
the click, modeling real checkout friction — which means the label a model
would train on doesn't exist yet at the moment a recommendation is served.
`training/build_dataset.py`'s window functions (`ROWS BETWEEN UNBOUNDED
PRECEDING AND 1 PRECEDING`) exist specifically so nothing trains on an
outcome it couldn't have known at serving time.

**What success would mean, commercially.** Click-through on recommended
titles, cart-to-purchase conversion, and catalog utilization (do
recommendations pull from more than the top 50 titles, or just reinforce
what's already popular) are the real-world metrics this system would move.
None of them are measured against real users here: the simulator samples
independently and never calls `/recommend` (see
[ADR 0002](docs/adr/0002-simulator-recommender-decoupling.md)), so every
offline number elsewhere in this README — AUC, NDCG@10, recall@K — is a
proxy for those metrics, not a live business result.

**Why synthetic, not a public dataset.** A public click-log dataset is
static: load it, split it, done — and that removes the actual engineering
problem this project exists to solve. A live, continuous, unbounded stream is
what turns the online/offline feature split, feature freshness, and
cold-start into real problems instead of a train/test split. There's no
fixed "the data" to load; the online store has to stay current against a
stream that never stops, and cold-start has to be handled continuously
because new users and items keep arriving while the system is running. That
property doesn't survive contact with a static dataset.

The live simulator (`services/simulator/`) runs continuously at a small,
conversational scale — 2,000 items, 5,000 users — so the whole stack (Kafka,
feature freshness, retraining) stays exercised in real time. A separate,
standalone batch generator, [`training/generate_dataset.py`](training/generate_dataset.py),
can seed the same `events`/`items` schema at a much larger scale (20,000
items, 50,000 users, 28 days, ~20M rows) with the same class of behavioral
mechanics — Zipf popularity with ~10%-of-head rotation per day, position bias,
within-session fatigue, diurnal and weekend seasonality, and both cold-start
users (arriving mid-window) and cold-start items (held out of every event
entirely) — for scale-sensitive experiments the live stream isn't sized for.
It's a distinct tool, not a drop-in replacement for the live stream: its
categories are generic codes (`cat_00`-`cat_39`) rather than the live
simulator's named genres, and it loads via `psycopg2` rather than the rest of
`training/`'s `psycopg3`. It ships its own validation pass (position-bias
ratio, fatigue-decay curve, diurnal amplitude, referential integrity) so a
generated dataset can be checked before it's trained on. Used for the
catalog-scale question raised and left open in
[ADR 0003](docs/adr/0003-two-tower-catalog-scale-investigation.md).

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
| Position bias in the slate | ranking must discount position (see the IPW note below — it wasn't, for a while) |

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
      point-in-time-correct features and validated offline before promotion. Three
      separate offline-metrics-vs-served-behavior bugs have been found and fixed in this
      project so far — cold-start's evaluation split, the two-tower's training objective,
      and the ranker's position feature — and they share a pattern worth reading once:
      [ADR 0004](docs/adr/0004-offline-metrics-vs-served-behavior.md). See the retrieval,
      cold-start, and position-bias notes below for each one's specifics, and
      [ADR 0003](docs/adr/0003-two-tower-catalog-scale-investigation.md) for the
      retrieval/cold-start investigation in full. `training/build_dataset.py` and
      `build_retrieval_dataset.py` stream via a server-side cursor and write Parquet
      incrementally (peak RSS ~614MB, was OOMing a 3GB container at ~4.2M rows before
      the fix).
- [x] **4. Serving** — the LightGBM ranker drives ranking for warm users (`RANKING_MODE`,
      reversible without a deploy). Retrieval for warm users defaults to
      `RETRIEVAL_MODE=two_tower_only` — chosen by measuring candidate-set recall across
      all three options against real held-out clicks, not by narrative; see the
      retrieval note below for the numbers. Retrieval is exact brute-force search over
      the ~2k-item catalog, not an ANN index — unnecessary at this scale. Service
      latency (below) is measured, not aspirational, and single-worker throughput
      turned out to be the real bottleneck — see the latency note below. Model
      artifacts (`models/`) are version-pinned in git rather than
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

**Position was leaking into the ranker as a feature, not being discounted — and it was
the 4th-highest-gain feature doing it.** `services/api/app/main.py` fed every candidate
a made-up `position=0` at serving time (`SCORING_POSITION`, since real position isn't
known before ranking — it's ranking's *output*), while training used real historical
position on every row. That's train/serve skew on a feature ranked 4th by gain
(70,520, ahead of `item_ctr_before`), and it showed up as a symptom before anyone went
looking for a metric: a served top-30 for a genuinely diverse candidate set collapsed to
**2 distinct `model_score` values**, because the dominant tree splits keyed on a feature
now falsely constant sent every candidate down the same branch.

Dropping `position` from the feature set would have removed the skew but thrown away the
debiasing this system is supposed to do (see the stream-properties table above). Instead:
`position` was removed from the feature vector *and* the position bias was moved into the
training labels' sample weights via inverse propensity weighting (IPW) — clicked examples
weighted `1/θ(position)`, non-clicks weight 1, the standard correction for click data
under an examine-then-click factorization (Joachims et al., 2017). θ was estimated from
the training split's marginal CTR by position, not assumed:

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| θ (estimated) | 1.000 | 0.859 | 0.773 | 0.683 | 0.611 | 0.560 | 0.515 | 0.472 | 0.434 | 0.409 |

This is a clean, unconfounded estimate in this environment because the simulator assigns
slate position independently of relevance (items are drawn by popularity weight; a pure
multiplicative decay is applied afterward to the already-computed click probability). The
estimator was chosen before checking the simulator's actual decay function
(`1/(1+0.15·pos)`, `services/simulator/app/main.py:119`) against it — read only
afterward, as validation, not as a shortcut. It matches within ~2 points at every
position (pos 9: estimated 0.409 vs. true 0.426).

Before/after retrain:

| metric | before | after | Δ |
|---|---|---|---|
| AUC | 0.7071 | 0.6866 | -0.0205 |
| logloss | 0.1440 | 0.1360 | -0.0080 (improved) |
| NDCG@10 | 0.6522 | 0.6681 | +0.0159 (improved) |
| Spearman(prediction, position) | -0.3052 | -0.0141 | debiasing confirmed |

AUC dropping was expected — the old number was partly borrowed from a feature never
available at inference. NDCG@10 *rising* was not predicted going in, and it's the more
interesting result once explained rather than just reported: AUC is pointwise over the
whole validation set, so any feature globally correlated with the label inflates it —
including `position`, which the simulator assigns by draw order and which therefore
carries no information about which candidate is actually best *within* a given session.
NDCG@10 is grouped by `session_id` and measures exactly that within-session choice. The
old model was measurably injecting noise into the one ranking decision that matters while
its aggregate metric looked fine — the offline number was inflated by a feature that was
actively degrading real ranking quality. The Spearman drop (-0.3052 → -0.0141) is what
confirms the IPW correction actually took, rather than the fix being "remove the feature
and hope": the debiased model's predictions are close to uncorrelated with the position
it never sees, directly or through a correlated proxy.

**Limitation, stated plainly:** the propensity curve above was validated against a
*known* ground-truth decay function only because this is a synthetic simulator where
that ground truth exists. In production, propensity has to be *estimated* from
observational data (result randomization, an EM-style examination model, or similar),
and that estimate can be wrong in ways nothing here would catch. IPW as a method
transfers directly; this write-up's clean validation does not — a real deployment needs
its own propensity-estimation validation before the correction can be trusted the way it
can be here.

This is the third offline-metrics-vs-served-behavior bug found in this project (after
`is_new_user` and the two-tower objective above). The pattern across all three — and what
actually surfaced each one, since it was never a smarter offline metric — is recorded in
[ADR 0004](docs/adr/0004-offline-metrics-vs-served-behavior.md).

**Single-request latency looked fine; concurrency exposed the real bottleneck.**
Measured on the deploy host (Vultr `vc2-4c-8gb`, 4 shared vCPU / 8GB, x86_64, Atlanta)
against `127.0.0.1:8000` directly — no TLS, no nginx, no network hop. This is the
service number, not an end-to-end one; see [DEPLOY.md](DEPLOY.md) for the measurement
method.

| workers | concurrency | p50 | p95 | p99 | throughput |
|---|---|---|---|---|---|
| 1 | c=1 | 26.7ms | 36.0ms | 46.9ms | 35.7 req/s |
| 1 | c=4 | 98.5ms | 145.8ms | 178.3ms | 38.5 req/s |
| 1 | c=10 | 257.3ms | 338.4ms | 366.6ms | 37.8 req/s |
| 4 | c=4 | 37.7ms | 79.9ms | 104.4ms | 84.9 req/s |
| 4 | c=10 | 100.4ms | 153.9ms | 188.9ms | 98.1 req/s |

At c=1, ~27ms p50 looks like nothing to fix. The signature of the actual problem only
shows up across the concurrency sweep: with 1 worker, **throughput stayed flat at
~36-38 req/s regardless of concurrency** — c=10 got no more work done per second than
c=1, just with each request waiting longer behind the others. `services/api/Dockerfile`
ran uvicorn with no `--workers` flag; LightGBM and PyTorch inference is synchronous, so
it blocks the single event loop and every concurrent request serializes behind whichever
one is currently scoring. Fixed by making the worker count configurable
(`UVICORN_WORKERS`, `.env.example`) instead of hardcoded at one — 4 workers on this
4-vCPU host raised throughput to ~85-98 req/s and cut p50 at c=10 from 257ms to 100ms.
Local dev still defaults to 1 worker (simpler logs, one model copy in RAM); production
should set it to the host's vCPU count. Each worker loads its own copy of the ranker and
two-tower artifacts — irrelevant at ~612KB today, a real cost if the models grow.

These numbers were measured with the full stack up — 10 containers plus the simulator
producing continuously — sharing those same 4 shared vCPUs, so this is a loaded-host
number, not an isolated benchmark; a dedicated host or less contention would likely look
better in both configurations. No end-to-end (public URL, TLS, nginx) number is reported
here: nginx/TLS were never set up for this deploy, so there's nothing to measure, and no
placeholder is left standing in where that number would go.
