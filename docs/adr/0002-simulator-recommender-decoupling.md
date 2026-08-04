# ADR 0002 — Simulator and recommender are decoupled, on purpose

**Status:** accepted · **Date:** 2026-08-04

## Context

Phase 5 (observability) needed a decision on how to measure "is this system's
recommendations any good" live, not just offline. The obvious metric is CTR of
served recommendations: show a slate, see if the user clicks it.

But `services/simulator` doesn't call `/recommend`. It samples items directly via
`world.py`'s Zipf popularity and latent user-category affinity, generates a session
independent of whatever the API would have recommended, and publishes the resulting
events to Kafka. The recommender reads that same event stream downstream (via the
consumer and the offline store) but never influences what the simulator does next.

## Decision

Keep it that way, for now. Do not wire the simulator to call `/recommend` and
condition its simulated clicks on the returned slate.

`services/api/app/metrics.py` exposes `recsys_simulated_stream_ctr` -- clicks over
impressions in the raw event stream -- but it is explicitly not labeled or treated
as recommendation CTR anywhere, including the Grafana dashboard, which carries a
text panel spelling this out next to the number.

## Rationale

**This is what makes offline evaluation trustworthy.** Every offline number in this
project -- `training/compare_rankers.py`'s heuristic-vs-model AUC comparison,
`train_ranker.py`'s NDCG@10, `train_two_tower.py`'s recall@K -- depends on the
simulated stream being ground truth that does not know what the recommender would
have done. If the simulator's clicks were themselves a function of served
recommendations, every subsequent offline evaluation on that data would be
evaluating a model partly against its own past decisions, and "the model beats the
heuristic" would stop being a clean claim.

**Closing the loop now would answer a question that doesn't exist yet.** A live CTR
comparison is only informative once there is something to compare -- a model whose
online performance might differ from the heuristic's in a way offline metrics can't
see (position bias, exposure effects, feedback loops). That condition is met now
(the LightGBM ranker and two-tower retrieval are both live, see `services/api/app/main.py`),
but closing the loop is still a separate, larger change from the rest of Phase 5:
it means the simulator's behavior would depend on live recommendations instead of
independent ground truth, which is a real architectural shift, not an additive
metric. It's recorded as Phase 6 in the README roadmap, deliberately not built here.

**The naming has to hold even after Phase 6 exists.** Once the loop is closed,
there will be a genuine served-recommendation CTR metric, and it needs a name that
doesn't collide with today's environment-baseline number. `recsys_simulated_stream_ctr`
stays what it is; the future metric gets its own name when it exists.

## Consequences

- There is no live signal today for "are the promoted model and blended retrieval
  actually converting better than the heuristic would have" -- only the offline
  metrics computed in `training/`. That gap is real and is exactly what Phase 6
  closes.
- `recsys_simulated_stream_ctr` will read identically regardless of `RANKING_MODE`
  or `RETRIEVAL_MODE`, by construction. That's not a bug in the metric; it's the
  point of this ADR.
- When Phase 6 is built: the simulator calls `/recommend` for (some share of)
  sessions and simulates clicks against the returned slate using `world.py`'s
  existing affinity model, instead of `world.pick_items()`'s independent sampling.
  That produces a real online counterfactual comparison between rankers -- the
  strongest evaluation this project can offer, once it exists.
