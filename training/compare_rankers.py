"""Compares the LightGBM ranker against the live heuristic on identical held-out data.

Answers one question: is the model in services/api/app/main.py:rank() worth trusting
with the actual ranking decision, or should it stay in shadow? Both scorers see the
exact same point-in-time features build_dataset.py produced for the same validation
split train_ranker.py evaluates the model on -- an apples-to-apples comparison, not
two different samples.

HEURISTIC_SCORE is copied from services/api/app/main.py:rank() and kept in sync
manually -- if that formula changes, update this too. Two approximations, both
affecting only the cold-start minority of rows:
  - cold_start is approximated by is_new_user (the API's real definition is "fewer
    than 5 recorded impressions", not something build_dataset.py captured).
  - the live heuristic's cold-start-only "+ 0.0002 * pop" popularity boost is
    omitted -- pop is the retrieval-time f:pop:1h score, not logged per-impression.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from train_ranker import (FEATURES, IN_PATH, META_PATH, MODEL_PATH, VALID_FRACTION,
                           add_engineered_features)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare_rankers")


def heuristic_score(df: pd.DataFrame) -> pd.Series:
    ctr = df["item_ctr_before"]                                    # same (clicks+1)/(imps+20) formula
    total_clicks = df["user_clicks_before"]
    category_clicks = df["user_category_clicks_before"]
    share = (category_clicks / total_clicks.replace(0, np.nan)).fillna(0.0)
    affinity = 1.0 + share
    fatigue = 0.6 ** df["user_item_impressions_before"]
    return ctr * affinity * fatigue


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    df = add_engineered_features(df)
    split = int(len(df) * (1 - VALID_FRACTION))
    valid_df = df.iloc[split:]
    log.info("evaluating on %d held-out rows (%.3f%% positive)",
              len(valid_df), 100 * valid_df["label"].mean())

    booster = lgb.Booster(model_file=MODEL_PATH)
    model_pred = booster.predict(valid_df[FEATURES])
    model_auc = roc_auc_score(valid_df["label"], model_pred)
    model_loss = log_loss(valid_df["label"], model_pred)

    heur_pred = heuristic_score(valid_df)
    heur_auc = roc_auc_score(valid_df["label"], heur_pred)
    # logloss isn't a fair comparison: the heuristic's score was never fit to be a
    # calibrated probability (it can exceed 1), unlike the model's binary objective.
    # AUC is rank-order-only and scale-invariant, so it's the metric that's actually comparable.

    corr = np.corrcoef(model_pred, heur_pred)[0, 1]

    log.info("model:     auc=%.4f logloss=%.4f (logloss is meaningful here -- calibrated probability)",
              model_auc, model_loss)
    log.info("heuristic: auc=%.4f (logloss omitted -- score isn't a calibrated probability)", heur_auc)
    log.info("score correlation (pearson): %.4f", corr)
    log.info("verdict: %s",
              "model beats heuristic on held-out AUC" if model_auc > heur_auc
              else "heuristic still wins or ties on held-out AUC -- do not promote yet")


if __name__ == "__main__":
    main()
