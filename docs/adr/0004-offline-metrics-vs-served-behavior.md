# ADR 0004 — Offline metrics vs. served behavior: three measurement bugs, one pattern

**Status:** accepted · **Date:** 2026-08-06

## Context

This is the third time in this project that an offline metric looked fine —
or at least gave no obvious signal of a problem — while what the system
actually served was wrong. Each time, the offline number was computed
correctly on the data available to it; the bug was in what the data (or the
feature) actually represented, not in the arithmetic. Recording the pattern
here, not just the fix, because the same signature is likely to recur on
whatever gets built next.

## The three cases

**1. `is_new_user` as a permanent flag** ([ADR 0003](0003-two-tower-catalog-scale-investigation.md)).
Cold-start's "neither model shows validated signal" conclusion came from a
split on `is_new_user`, which `services/simulator/app/world.py` set once at
user creation and never updated. The split was `true` on 84% of validation
rows against a ~5% true new-user rate. No metric flagged this — it surfaced
because 84% was implausible against a known target, and someone checked.

**2. Pointwise BCE in the two-tower** ([ADR 0003](0003-two-tower-catalog-scale-investigation.md)).
Recall@K described as "real but modest" was, once a popularity baseline was
added, statistically indistinguishable from random (positive-control rank
0.4857, where 0.5 is chance). The training loss itself never signaled this —
BCE against ~4% CTR negatives can look like it's converging while having
almost no pressure to rank candidates against each other. It surfaced only
once the *right comparison* (popularity, not random) was added.

**3. Position train/serve skew** (this ADR). Offline AUC of 0.7071 looked
like a reasonable, unremarkable number — nothing about it demanded scrutiny.
What surfaced the bug was looking at real served output: a top-30 for a
genuinely diverse candidate set collapsing to 2 distinct `model_score`
values. No offline metric caught it; a five-minute look at a live response
did.

## The pattern

In all three cases the offline number was correct arithmetic on the data it
was given, and none of the three was caught by a better or more careful
offline metric. Each was caught by one of:

- comparing a measured proxy against a **known ground-truth ratio** it should
  plausibly match (84% vs. 5%);
- comparing against the **right baseline**, not just against random or
  against nothing (popularity, not random);
- **inspecting real served output directly**, not just the training-time
  offline number (2 distinct scores in a live top-30).

None of these are metrics you compute once and trust going forward — they're
habits of checking the number against something outside itself. Recorded as
practice for whatever's built next in this project:

- Any measured proxy standing in for a real-world quantity (a flag, a label,
  a split) should be sanity-checked against a known target ratio where one
  exists, not assumed correct because it's the field that was already there.
- An offline ranking/retrieval metric is only informative relative to a
  baseline. Random is the weakest possible baseline; use the strongest cheap
  one available (popularity, a heuristic) before calling a number good.
- Periodically look at real served output, not only the offline eval number
  computed at training time — a collapsed or degenerate distribution in
  actual responses is a signal no aggregate offline metric is guaranteed to
  surface.
- Any feature used at training time has to be checked for availability at
  serving time. If it isn't truly available then, it cannot be fed a
  made-up constant — either engineer the bias away (as below) or make its
  absence explicit.

## This case's specifics: position bias in the LightGBM ranker

`position` was the ranker's 4th-highest feature by gain (70,520, behind three
user/item interaction features and ahead of `item_ctr_before`), computed from
real historical serve position at training time. At serving time,
`services/api/app/main.py` fed every candidate `SCORING_POSITION = 0` — a
made-up constant, since position isn't actually known before ranking; it's
ranking's *output*. A feature that dominant, pinned to a single constant
value it never had in training, sends every candidate down the same tree
branch: hence the collapse to 2 distinct scores.

**Fix: inverse propensity weighting (IPW), not feature removal alone.**
Dropping `position` from the feature set would have removed the skew but
also thrown away the debiasing the system is supposed to do (the README
lists position bias as a stream property specifically because "ranking must
discount position"). Instead, `position` was removed from the feature
vector *and* the position bias was moved into the training labels' sample
weights, following the examine-then-click factorization from Joachims et al.
("Unbiased Learning-to-Rank with Biased Feedback", 2017): clicked examples
are weighted `1/θ(position)`, non-clicks weight 1, where θ is the position's
click-examination propensity relative to position 0.

θ was estimated from the training split's marginal CTR by position, not
assumed:

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| θ (estimated) | 1.000 | 0.859 | 0.773 | 0.683 | 0.611 | 0.560 | 0.515 | 0.472 | 0.434 | 0.409 |

This is a clean, unconfounded estimate in this environment specifically
because the simulator assigns slate position independently of relevance
(items are drawn by popularity weight; a pure multiplicative position-decay
factor is applied afterward to the already-computed click probability, in
`services/simulator/app/main.py`). The estimator was chosen before looking
at the simulator's actual decay function — `1/(1+0.15·pos)`, read at
`services/simulator/app/main.py:119` only afterward, as a validation check.
It matches the estimated curve within ~2 points at every position (e.g. pos
9: estimated 0.409 vs. true 0.426), confirming the estimator is sound here,
not merely plausible.

## Outcome

**Served output** — distinct `model_score` values in a live top-k, five
users sampled:

| | before | after |
|---|---|---|
| top-10 | 2 | 9-10 |
| top-30 | 2 | 22-27 |

**Offline metrics**, before/after retrain:

| metric | before | after | Δ |
|---|---|---|---|
| AUC | 0.7071 | 0.6866 | -0.0205 |
| logloss | 0.1440 | 0.1360 | -0.0080 (improved) |
| NDCG@10 | 0.6522 | 0.6681 | +0.0159 (improved) |
| Spearman(prediction, position) | -0.3052 | -0.0141 | debiasing confirmed |

AUC dropping was expected: the old number was partly borrowed from a feature
unavailable at inference. NDCG@10 *rising* was not predicted going in, and
it's the more informative result once explained rather than just reported.
AUC is pointwise over the entire validation set, so any feature globally
correlated with the label inflates it — including `position`, which the
simulator assigns by draw order and which therefore carries no information
about which candidate is actually best *within* a given session. NDCG@10 is
grouped by `session_id` and measures exactly that within-session choice. A
model leaning on position to look good in aggregate was measurably injecting
noise into the one ranking decision that matters, not improving it — the
aggregate metric was inflated by a feature that was actively degrading real
ranking quality. The Spearman drop (-0.3052 → -0.0141) is what confirms the
IPW correction actually took, rather than the fix being "remove the feature
and hope": the debiased model's predictions are close to uncorrelated with
the position it never sees, directly or through a correlated proxy feature.

## Limitation, stated plainly

The propensity curve above was validated against a *known* ground-truth
decay function only because this is a synthetic simulator where that ground
truth exists and could be read from source. In a real production system, the
propensity has to be *estimated* from observational data — via result
randomization, an EM-style examination model, or similar — and that estimate
can be wrong in ways nothing here would catch. IPW as a method transfers
directly to production. The clean validation in this write-up does not: a
real deployment needs its own propensity-estimation validation (e.g.
periodic randomized-position experiments) before the correction can be
trusted the way it can be here.

## A fourth instance: a dashboard query, not a model or a dataset

The Grafana panel "Cold-start share of traffic" showed a version of the same
pattern, but as two distinct, compounding findings, not one.

**Instrumentation**: `prometheus_client` counters are in-process and
lazily create a label the first time it's incremented. Any container
restart (this project's `UVICORN_WORKERS` change among them) resets that
registry to empty, and Prometheus marks a label stale once it stops
appearing in scrapes — so a label can go missing from queries for as long
as that code path goes unexercised after a restart. Not a PromQL defect;
a property of where the counter's state lives.

**Query**: `/recommend` is only called manually (the simulator never
calls it — ADR 0002), so traffic is sparse and bursty. Against a short
`rate()` window that produced a bare `NaN` whenever both label values had
zero samples in-window (0/0, not "no data"), and readings computed from
too few requests to be representative.

The initial suspicion — that the denominator was silently dropping a
missing label's contribution — was tested directly and didn't hold:
`sum()` over an absent series contributes nothing, which is the same
arithmetic as contributing zero. There was no computation bug in the
original expression; the real issues were the 0/0 case and window
representativeness under sparse traffic.

Fixed by widening to `increase()` over 1h (matching this dashboard's
existing two-tower hit-rate convention), `or vector(0)` on the numerator
so a genuine zero renders as `0%`, and a `> 0` guard on the denominator so
true zero-traffic windows render as **No data** instead of `NaN`.

**Known, unfixed instance of the same query shape**: "Two-tower retrieval
hit rate (1h)" has the identical unguarded division and would show the
same bare `NaN` if `eligible` ever hit zero in a window. Not fixed here.
