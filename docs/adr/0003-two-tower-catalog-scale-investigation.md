# ADR 0003 — Two-tower retrieval: catalog-scale hypothesis, tested and shelved

**Status:** accepted · **Date:** 2026-08-04

## Context

Two weak offline results had been sitting in the README without a clear cause:
two-tower recall@K described as "real but modest," and cold-start described as
having "no validated signal" for either model. The working hypothesis was that
both were artifacts of a ~2,000-item catalog: brute-force retrieval is trivial
at that size, so there's little for retrieval to prove, and popularity is a
strong enough prior that content-based cold-start has little room to beat it.

Regenerating the dataset at 20k items (`generate_dataset.py`, ~21M rows,
6-9GB) was proposed to test that hypothesis directly.

## What the investigation actually found

Before spending hours on a regeneration, the hypothesis was checked against
the real numbers rather than assumed.

**Cold-start's "no signal" claim was a measurement bug, not a finding.** The
original split used `is_new_user`, which turned out to be a permanent
per-user label in `services/simulator/app/world.py` — set once when a user
was created (`is_new`) and never updated, so a user spawned as "new" hours
earlier still carried it forever, however much history they'd since built up.
In the validation split, that flag was `true` on 84% of rows against a 5%
new-user session rate — the split should never have been trusted at that
ratio. Re-measured against `user_clicks_before == 0` (an actual point-in-time
behavioral signal, matching the API's own `cold_start` definition), the
LightGBM ranker beats a pure popularity baseline even at zero prior clicks
(AUC 0.615 vs 0.535). There was signal the whole time; nobody had measured it
correctly.

**Retrieval's "real but modest" claim didn't survive a popularity baseline.**
The original recall@K table only compared the two-tower model against random
retrieval. Adding the comparison that actually matters — a trivial "always
retrieve the most popular items" baseline — showed:

| K | model | popularity | random |
|---|---|---|---|
| 50 | 0.0264 | 0.3006 | 0.0250 |
| 100 | 0.0586 | 0.4025 | 0.0500 |
| 200 | 0.1381 | 0.5221 | 0.1000 |

The model is statistically indistinguishable from random (mean percentile
rank of the true clicked item among 2,000 candidates: 0.4857, where 0.5 is
pure chance) while popularity alone — no personalization, no model — gets
3.8-11x better recall@K than the model at every K. Embedding variance and
item-index consistency between training and evaluation were both checked and
ruled out; nothing was mechanically broken there.

**Diagnosis: the training objective, not the catalog.** The model was trained
with pointwise BCE against natural class-imbalance negatives (~5% CTR). That
objective is close to optimized by predicting "no" uniformly — it has no
built-in pressure to rank candidates against each other, which is exactly
what recall@K measures. A catalog ten times larger would very likely
reproduce the same near-random result, at ten times the cost to find out.

## Decision

Do not regenerate the dataset. Fix the two-tower training objective (in-batch
sampled softmax with logQ correction and temperature, positives-only batches,
towers and features unchanged) and re-measure against the same popularity
baseline before deciding anything about catalog scale. The scale question is
only well-posed once there is a model that has actually learned something to
have a ceiling.

Two downstream decisions are gated on that re-measurement, not decided here:
whether to pursue the 20k-item regeneration at all, and whether
`RETRIEVAL_MODE=blended` (which unions two-tower candidates into serving
today, justified in the README by "recall@K was real but modest") should stay
on, given that justification no longer holds as stated.

## Consequences

- Hours of compute and a real risk of reproducing the same broken result at
  higher cost were avoided by spending a fraction of that time on diagnosis
  first.
- The cold-start fix is real and independent of whatever happens with
  retrieval: `world.py` now tracks actual lifetime impression count and
  decays `is_new_user` accordingly, matching the API's live `cold_start`
  definition instead of a permanent creation-time label.
- Both existing trained models (`ranker.txt`, `two_tower.pt`) were trained on
  data where `is_new_user` carried the old, broken semantics. Retraining after
  this fix is not directly comparable to the metrics recorded before it, and
  any change in LightGBM's numbers post-fix should not be attributed to the
  two-tower objective change happening alongside it — they are independent
  causes.
- If the corrected objective still doesn't beat popularity, the honest
  conclusion is that learned retrieval didn't outperform popularity in this
  environment — a real, reportable result in its own right, not a failure to
  hide.

## Outcome

The objective was rewritten (`training/train_two_tower.py`): in-batch sampled
softmax over positives only, with a logQ correction (in-batch negatives are
sampled proportional to click frequency, not uniformly, so popular items would
otherwise be penalized just for appearing as a negative often) and duplicate-item
masking (at 2,000 items and this much concentration, the same item legitimately
recurs as different users' positives within a batch — left unmasked, the loss
would punish the model for correctly scoring a real match). Towers and features
were not touched, so the effect is attributable to the objective alone.

Positive-control rank test, the number that matters most because it's
independent of any choice of K: mean percentile rank of the true clicked item
went from **0.4857 to 0.1977** (median **0.0845**), against 0.5 for a model
that has learned nothing. This is the actual fix — everything below follows
from it.

Recall@K, measured against the criteria set before retraining (must beat the
popularity baseline, not just random):

| K | model | popularity | random | vs. popularity |
|---|---|---|---|---|
| 50 | 0.3008 | 0.3006 | 0.0250 | **tie** (+0.0002 — not a meaningful margin) |
| 100 | 0.4118 | 0.4025 | 0.0500 | **beats**, +0.0093 |
| 200 | 0.5329 | 0.5221 | 0.1000 | **beats**, +0.0108 |

K=50 is reported as a tie, not rounded up to a win — the margin is within
noise for a training run with no fixed random seed. K=100 and K=200 show a
real, repeatable-looking margin.

**This closes the catalog-scale question as originally posed**: it presupposed
a model with something to have a ceiling, and now there is one. Whether that
margin would grow at a larger catalog is a genuinely open question this
2,000-item environment cannot answer — not a pending task, a question the
environment is the wrong size to resolve. See the README for how this is
recorded going forward.

Two downstream decisions this unblocks — evaluating `RETRIEVAL_MODE` against
the now-real signal, and updating the README's retrieval characterization —
are handled separately (README, and the commit history following this one),
not folded into this ADR.
