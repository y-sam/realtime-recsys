"""Trains a LightGBM click-ranker (P(click | impression)) on build_dataset.py's output.
Split is by time, not random: the last VALID_FRACTION of rows are held out.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_ranker")

IN_PATH = os.getenv("TRAINING_SET_PATH", "/work/data/training/impressions.parquet")
MODEL_PATH = os.getenv("MODEL_PATH", "/work/models/ranker.txt")
META_PATH = os.getenv("MODEL_META_PATH", "/work/models/ranker.meta.json")
METRICS_PATH = os.getenv("METRICS_PATH", "/work/models/ranker_metrics.json")
VALID_FRACTION = 0.2
NDCG_K = 10

PRIOR_CLICKS = 1.0
PRIOR_IMPRESSIONS = 20.0
MIN_PROPENSITY = 0.05  # floor so a thin high-position bucket can't produce a runaway IPW weight

CATEGORICAL = ["category"]
FEATURES = [
    "is_new_user", "price_tier", "category",
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


def estimate_position_propensity(train_df: pd.DataFrame) -> pd.Series:
    """theta(k) = P(click | shown at position k) / P(click | shown at position 0).

    Position is assigned to the training-time slate independently of item
    relevance (the simulator draws the slate by popularity weight, then applies
    a pure multiplicative position decay to the already-computed click
    probability) -- so this marginal ratio is a direct, unconfounded estimate
    of the examination propensity, not a mix of position and relevance effects.
    """
    ctr_by_position = train_df.groupby("position")["label"].mean()
    theta = (ctr_by_position / ctr_by_position.loc[0]).clip(lower=MIN_PROPENSITY)
    return theta


def ipw_weights(df: pd.DataFrame, theta: pd.Series) -> pd.Series:
    """Inverse-propensity weight: 1/theta(position) for clicks, 1 for non-clicks.

    Unbiased estimator of relevance under examine-then-click factorization
    (Joachims et al., "Unbiased Learning-to-Rank with Biased Feedback", 2017):
    E[click / theta(position)] = relevance, independent of position.
    """
    theta_of_row = df["position"].map(theta).fillna(MIN_PROPENSITY)
    return df["label"].astype(float) * (1.0 / theta_of_row) + (1 - df["label"])


def ndcg_at_k(df: pd.DataFrame, pred: pd.Series, k: int = NDCG_K) -> float:
    """NDCG@k grouped by real session_id."""
    def dcg(labels: list[int]) -> float:
        return sum(label / math.log2(i + 2) for i, label in enumerate(labels))

    scored = df[["session_id", "label"]].copy()
    scored["pred"] = pred.values
    scores = []
    for _, group in scored.groupby("session_id"):
        if len(group) < 2 or group["label"].sum() == 0:
            continue
        ranked_labels = group.sort_values("pred", ascending=False)["label"].tolist()[:k]
        ideal_labels = sorted(group["label"].tolist(), reverse=True)[:k]
        idcg = dcg(ideal_labels)
        if idcg == 0:
            continue
        scores.append(dcg(ranked_labels) / idcg)
    return float(sum(scores) / len(scores)) if scores else 0.0


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    log.info("loaded %d rows (%.3f%% positive)", len(df), 100 * df["label"].mean())
    df = add_engineered_features(df)

    split = int(len(df) * (1 - VALID_FRACTION))
    train_df, valid_df = df.iloc[:split], df.iloc[split:]
    log.info("split: %d train (up to %s) / %d valid (from %s)",
              len(train_df), train_df["ts"].max(), len(valid_df), valid_df["ts"].min())

    theta = estimate_position_propensity(train_df)
    log.info("estimated position propensity (relative to position 0):\n%s", theta.to_string())
    train_weight = ipw_weights(train_df, theta)

    train_set = lgb.Dataset(train_df[FEATURES], label=train_df["label"], weight=train_weight,
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
    ndcg10 = ndcg_at_k(valid_df, pd.Series(valid_pred, index=valid_df.index))
    log.info("validation: auc=%.4f logloss=%.4f ndcg@%d=%.4f (best_iteration=%d)",
              auc, loss, NDCG_K, ndcg10, booster.best_iteration)

    importance = pd.Series(booster.feature_importance(importance_type="gain"), index=FEATURES)
    log.info("feature importance (gain):\n%s", importance.sort_values(ascending=False).to_string())

    # Diagnostic, not a training input: position is excluded from FEATURES, but if IPW
    # actually corrected the bias, the model's predictions should no longer be able to
    # reconstruct position from other, correlated features either.
    position_corr = float(pd.Series(valid_pred, index=valid_df.index).corr(
        valid_df["position"], method="spearman"))
    log.info("spearman(prediction, position) on valid, position excluded from features: %.4f",
              position_corr)

    compare_model_path = os.getenv("COMPARE_MODEL_PATH")
    compare_meta_path = os.getenv("COMPARE_META_PATH")
    if compare_model_path and compare_meta_path:
        with open(compare_meta_path) as f:
            compare_features = json.load(f)["features"]
        compare_booster = lgb.Booster(model_file=compare_model_path)
        compare_pred = compare_booster.predict(valid_df[compare_features])
        compare_corr = float(pd.Series(compare_pred, index=valid_df.index).corr(
            valid_df["position"], method="spearman"))
        log.info("comparison model (%s) spearman(prediction, position): %.4f",
                  compare_model_path, compare_corr)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    booster.save_model(MODEL_PATH, num_iteration=booster.best_iteration)
    log.info("saved -> %s", MODEL_PATH)

    meta = {"features": FEATURES, "category_codes":
            {cat: code for code, cat in enumerate(df["category"].cat.categories.tolist())}}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("saved -> %s", META_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "auc": auc, "logloss": loss, "ndcg10": ndcg10, "trained_at": time.time(),
            "position_propensity": {str(k): v for k, v in theta.to_dict().items()},
            "position_spearman_corr": position_corr,
        }, f, indent=2)
    log.info("saved -> %s", METRICS_PATH)


if __name__ == "__main__":
    main()
