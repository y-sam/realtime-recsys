"""Trains a two-tower retrieval model on the dataset from build_retrieval_dataset.py.

Objective: dot(user_embedding, item_embedding) predicts P(click), pointwise BCE
against the same abundant natural negatives (unclicked impressions) the ranker
uses. This is retrieval, not ranking -- see build_retrieval_dataset.py's docstring
for why fatigue and other (user, item) cross features can't live in either tower.

Item embeddings are precomputed once from each item's most recent known features
and saved as a static table (models/item_embeddings.npy); that's not a shortcut,
it's how two-tower retrieval actually serves in production -- item embeddings are
batch-refreshed, only the user embedding is computed live per request.

Evaluated with recall@K, the metric retrieval actually needs (ranking's AUC answers
a different question -- "is this one item good", not "is the right item in the
candidate set at all"): for each held-out click, is the true item among the
top-K nearest items to that impression's user embedding, versus the K/n_items
recall a random candidate set would get by chance.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_two_tower")

IN_PATH = os.getenv("RETRIEVAL_SET_PATH", "/work/data/training/retrieval.parquet")
MODEL_PATH = os.getenv("TOWER_MODEL_PATH", "/work/models/two_tower.pt")
ITEM_EMB_PATH = os.getenv("ITEM_EMB_PATH", "/work/models/item_embeddings.npy")
META_PATH = os.getenv("TOWER_META_PATH", "/work/models/two_tower.meta.json")
VALID_FRACTION = 0.2

# Matches CTR_PRIOR_CLICKS / CTR_PRIOR_IMPRESSIONS elsewhere in this codebase.
PRIOR_CLICKS = 1.0
PRIOR_IMPRESSIONS = 20.0

CATEGORIES = ["action", "comedy", "drama", "docu", "sports",
              "music", "kids", "horror", "reality", "news"]
ITEM_NUMERIC = ["price_tier", "item_impressions_before", "item_clicks_before", "item_ctr_before"]
USER_NUMERIC = ["user_clicks_before", "is_new_user"] + [f"user_share_{c}" for c in CATEGORIES]

CAT_EMBED_DIM = 8
HIDDEN = 64
EMBED_DIM = 32
BATCH_SIZE = 4096
EPOCHS = 6
LR = 1e-3
RECALL_KS = (50, 100, 200)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df["item_ctr_before"] = ((df["item_clicks_before"] + PRIOR_CLICKS)
                              / (df["item_impressions_before"] + PRIOR_IMPRESSIONS))
    for c in CATEGORIES:
        df[f"user_share_{c}"] = ((df[f"user_clicks_{c}_before"] + PRIOR_CLICKS)
                                  / (df["user_clicks_before"] + PRIOR_IMPRESSIONS))
    df["is_new_user"] = df["is_new_user"].astype(int)
    df["category_code"] = df["category"].map({c: i for i, c in enumerate(CATEGORIES)})
    return df


class Tower(nn.Module):
    def __init__(self, n_numeric: int, n_categories: int | None = None):
        super().__init__()
        self.cat_embed = nn.Embedding(n_categories, CAT_EMBED_DIM) if n_categories else None
        in_dim = n_numeric + (CAT_EMBED_DIM if n_categories else 0)
        self.mlp = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, EMBED_DIM))

    def forward(self, numeric: torch.Tensor, cat_idx: torch.Tensor | None = None) -> torch.Tensor:
        x = numeric if self.cat_embed is None else torch.cat([self.cat_embed(cat_idx), numeric], dim=-1)
        return self.mlp(x)


def standardize(df: pd.DataFrame, cols: list[str], stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Z-score using TRAIN-split stats. Same stats must be applied at serving time,
    which is why they're saved to META_PATH rather than recomputed per environment."""
    if stats is None:
        stats = {c: (float(df[c].mean()), float(df[c].std()) or 1.0) for c in cols}
    arr = np.column_stack([(df[c].to_numpy() - stats[c][0]) / stats[c][1] for c in cols])
    return arr.astype(np.float32), stats


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    log.info("loaded %d rows (%.3f%% positive)", len(df), 100 * df["label"].mean())
    df = add_engineered_features(df)

    split = int(len(df) * (1 - VALID_FRACTION))
    train_df, valid_df = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    log.info("split: %d train (up to %s) / %d valid (from %s)",
              len(train_df), train_df["ts"].max(), len(valid_df), valid_df["ts"].min())

    item_num, item_stats = standardize(train_df, ITEM_NUMERIC)
    user_num, user_stats = standardize(train_df, USER_NUMERIC)
    item_cat = train_df["category_code"].to_numpy()
    labels = train_df["label"].to_numpy().astype(np.float32)

    item_tower = Tower(len(ITEM_NUMERIC), n_categories=len(CATEGORIES))
    user_tower = Tower(len(USER_NUMERIC))
    opt = torch.optim.Adam(list(item_tower.parameters()) + list(user_tower.parameters()), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    item_num_t = torch.from_numpy(item_num)
    item_cat_t = torch.from_numpy(item_cat).long()
    user_num_t = torch.from_numpy(user_num)
    labels_t = torch.from_numpy(labels)

    n = len(train_df)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            item_emb = item_tower(item_num_t[idx], item_cat_t[idx])
            user_emb = user_tower(user_num_t[idx])
            logits = (item_emb * user_emb).sum(-1)
            loss = loss_fn(logits, labels_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        log.info("epoch %d: train bce=%.4f", epoch, total_loss / n)

    # --- static item embedding table: each item's most recent known features ---
    latest_items = (df.sort_values("ts").groupby("item_id").tail(1)
                     .reset_index(drop=True))
    latest_item_num, _ = standardize(latest_items, ITEM_NUMERIC, stats=item_stats)
    latest_item_cat = latest_items["category_code"].to_numpy()
    item_tower.eval()
    with torch.no_grad():
        item_embeddings = item_tower(torch.from_numpy(latest_item_num),
                                      torch.from_numpy(latest_item_cat).long()).numpy()
    item_ids = latest_items["item_id"].tolist()
    log.info("item embedding table: %d items x %d dims", *item_embeddings.shape)

    # --- recall@K on held-out clicks: is the true item among the top-K nearest items
    # to that impression's (point-in-time) user embedding? ---
    valid_pos = valid_df[valid_df["label"] == 1].reset_index(drop=True)
    valid_user_num, _ = standardize(valid_pos, USER_NUMERIC, stats=user_stats)
    user_tower.eval()
    with torch.no_grad():
        valid_user_emb = user_tower(torch.from_numpy(valid_user_num)).numpy()

    scores = valid_user_emb @ item_embeddings.T   # (n_valid_pos, n_items)
    item_id_to_col = {iid: i for i, iid in enumerate(item_ids)}
    true_cols = np.array([item_id_to_col.get(iid, -1) for iid in valid_pos["item_id"]])
    ranks = (-scores).argsort(axis=1)

    log.info("recall@K on %d held-out clicks (n_items=%d, random baseline = K/n_items):",
              len(valid_pos), len(item_ids))
    for k in RECALL_KS:
        topk = ranks[:, :k]
        hit = (topk == true_cols[:, None]).any(axis=1).mean()
        log.info("  recall@%-4d model=%.4f  random_baseline=%.4f", k, hit, k / len(item_ids))

    # The API only ever runs the USER tower live (item embeddings are the precomputed
    # static table above) -- exporting its weights as plain arrays lets serving do the
    # forward pass with a few lines of numpy instead of adding torch to that image.
    w1 = user_tower.mlp[0].weight.detach().numpy()  # (hidden, n_numeric)
    b1 = user_tower.mlp[0].bias.detach().numpy()
    w2 = user_tower.mlp[2].weight.detach().numpy()  # (embed_dim, hidden)
    b2 = user_tower.mlp[2].bias.detach().numpy()

    def numpy_user_forward(x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ w1.T + b1, 0.0)
        return h @ w2.T + b2

    # verify the numpy replica matches torch exactly before trusting it in serving.
    check = numpy_user_forward(valid_user_num[:64])
    with torch.no_grad():
        torch_check = user_tower(torch.from_numpy(valid_user_num[:64])).numpy()
    max_diff = float(np.abs(check - torch_check).max())
    assert max_diff < 1e-4, f"numpy user-tower replica diverges from torch by {max_diff}"
    log.info("numpy user-tower replica verified against torch (max diff %.2e)", max_diff)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({"item_tower": item_tower.state_dict(), "user_tower": user_tower.state_dict()}, MODEL_PATH)
    np.save(ITEM_EMB_PATH, item_embeddings)
    with open(META_PATH, "w") as f:
        json.dump({
            "item_ids": item_ids, "categories": CATEGORIES,
            "item_numeric": ITEM_NUMERIC, "user_numeric": USER_NUMERIC,
            "item_stats": item_stats, "user_stats": user_stats,
            "cat_embed_dim": CAT_EMBED_DIM, "hidden": HIDDEN, "embed_dim": EMBED_DIM,
            "user_tower_weights": {
                "w1": w1.tolist(), "b1": b1.tolist(),
                "w2": w2.tolist(), "b2": b2.tolist(),
            },
        }, f, indent=2)
    log.info("saved -> %s, %s, %s", MODEL_PATH, ITEM_EMB_PATH, META_PATH)


if __name__ == "__main__":
    main()
