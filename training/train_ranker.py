"""Trains a LightGBM click-ranker on the dataset from build_dataset.py.

Objective: P(click | impression). This replaces nothing in serving yet -- that's
phase 4 (services/api/app/main.py:rank()). Phase 3 is just: does a model trained
on our own online-store signals beat the heuristic on held-out data.

The engineered ratio features (item_ctr_before, user_category_share_before) reuse
the exact Bayesian smoothing prior as the online heuristic and the batch DAG
(CTR_PRIOR_CLICKS=1.0, CTR_PRIOR_IMPRESSIONS=20.0 in airflow/dags/batch_features.py)
so the model and the system it's meant to replace speak the same units.

Split is by time, not random: the last VALID_FRACTION of rows (by ts, the dataset
is already ordered) are held out, matching how the model will actually be used --
predicting forward, never backward.
"""
from __future__ import annotations

import json
import logging
import os

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_ranker")

IN_PATH = os.getenv("TRAINING_SET_PATH", "/work/data/training/impressions.parquet")
MODEL_PATH = os.getenv("MODEL_PATH", "/work/models/ranker.txt")
# feature order + category encoding serving needs to build a matching raw feature
# vector without pulling pandas into the latency-sensitive API -- see api/app/main.py.
META_PATH = os.getenv("MODEL_META_PATH", "/work/models/ranker.meta.json")
VALID_FRACTION = 0.2

# Matches CTR_PRIOR_CLICKS / CTR_PRIOR_IMPRESSIONS in airflow/dags/batch_features.py.
PRIOR_CLICKS = 1.0
PRIOR_IMPRESSIONS = 20.0

CATEGORICAL = ["category"]
FEATURES = [
    "position", "is_new_user", "price_tier", "category",
    "item_impressions_before", "item_clicks_before", "item_ctr_before",
    "user_clicks_before", "user_category_clicks_before", "user_category_share_before",
    "user_item_impressions_before",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df["item_ctr_before"] = ((df["item_clicks_before"] + PRIOR_CLICKS)
                              / (df["item_impressions_before"] + PRIOR_IMPRESSIONS))
    df["user_category_share_before"] = ((df["user_category_clicks_before"] + PRIOR_CLICKS)
                                         / (df["user_clicks_before"] + PRIOR_IMPRESSIONS))
    df["is_new_user"] = df["is_new_user"].astype(int)
    df["category"] = df["category"].astype("category")
    return df


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    log.info("loaded %d rows (%.3f%% positive)", len(df), 100 * df["label"].mean())
    df = add_engineered_features(df)

    split = int(len(df) * (1 - VALID_FRACTION))
    train_df, valid_df = df.iloc[:split], df.iloc[split:]
    log.info("split: %d train (up to %s) / %d valid (from %s)",
              len(train_df), train_df["ts"].max(), len(valid_df), valid_df["ts"].min())

    train_set = lgb.Dataset(train_df[FEATURES], label=train_df["label"],
                             categorical_feature=CATEGORICAL, free_raw_data=False)
    valid_set = lgb.Dataset(valid_df[FEATURES], label=valid_df["label"],
                             categorical_feature=CATEGORICAL, reference=train_set, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }
    booster = lgb.train(
        params, train_set, num_boost_round=500, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    valid_pred = booster.predict(valid_df[FEATURES], num_iteration=booster.best_iteration)
    auc = roc_auc_score(valid_df["label"], valid_pred)
    loss = log_loss(valid_df["label"], valid_pred)
    log.info("validation: auc=%.4f logloss=%.4f (best_iteration=%d)",
              auc, loss, booster.best_iteration)

    importance = pd.Series(booster.feature_importance(importance_type="gain"), index=FEATURES)
    log.info("feature importance (gain):\n%s", importance.sort_values(ascending=False).to_string())

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    booster.save_model(MODEL_PATH, num_iteration=booster.best_iteration)
    log.info("saved -> %s", MODEL_PATH)

    # category codes must match training order exactly -- LightGBM's saved categorical
    # splits are keyed by integer code, not by the string label.
    meta = {"features": FEATURES, "category_codes":
            {cat: code for code, cat in enumerate(df["category"].cat.categories.tolist())}}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("saved -> %s", META_PATH)


if __name__ == "__main__":
    main()
