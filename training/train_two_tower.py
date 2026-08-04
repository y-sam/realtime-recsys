"""Trains a two-tower retrieval model on the dataset from build_retrieval_dataset.py.

Objective: in-batch sampled softmax over positives only (dot(user_embedding,
item_embedding) as the logit for the true item vs every other item in the same
batch). A prior pointwise-BCE version trained against natural class-imbalance
negatives (~5% CTR) and came out statistically indistinguishable from random on
recall@K -- that objective is close to optimized by predicting "no" uniformly,
with no pressure to rank candidates against each other. See
docs/adr/0003-two-tower-catalog-scale-investigation.md.

Two corrections that in-batch softmax needs to not be misleading at this
catalog's concentration (77% of impressions in 25% of items):
  - logQ correction: in-batch negatives are sampled proportional to how often
    an item is clicked, not uniformly, so popular items appear as a negative
    far more often than their true relevance warrants. Subtracting log Q(item)
    (its empirical click frequency) from its logit removes that bias --
    without it the loss just relearns "penalize popular items."
  - duplicate-item masking: with 2,000 items and this much concentration, the
    same item legitimately appears multiple times as different users'
    positives within one batch. Without masking those out, the objective
    would penalize the model for scoring a real match highly just because it
    also happens to be the (correct) answer for a different row.

This is retrieval, not ranking -- see build_retrieval_dataset.py's docstring
for why fatigue and other (user, item) cross features can't live in either tower.

Item embeddings are precomputed once from each item's most recent known features
and saved as a static table (models/item_embeddings.npy); that's not a shortcut,
it's how two-tower retrieval actually serves in production -- item embeddings are
batch-refreshed, only the user embedding is computed live per request.

Evaluated three ways: recall@K against both a random baseline AND a popularity
baseline (recall@K alone is misleading here -- see the ADR), and a positive-
control check (percentile rank of the true item among all candidates; 0.5 means
the model has learned nothing, regardless of what recall@K says).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

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
# Smaller than a typical in-batch-softmax batch on purpose: at 2,000 items and
# this much popularity concentration, a large batch all but guarantees the top
# items appear dozens of times as duplicate positives (see duplicate masking above).
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
EPOCHS = int(os.getenv("EPOCHS", "40"))
LR = 1e-3
TEMPERATURE = float(os.getenv("SOFTMAX_TEMPERATURE", "0.1"))
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

    # Stats are computed from the full train_df (all impressions), not the
    # positives-only slice below -- these calibrate "typical value range" for
    # standardization and are reused at serving time, so they should reflect
    # the general population, not just clicks.
    _, item_stats = standardize(train_df, ITEM_NUMERIC)
    _, user_stats = standardize(train_df, USER_NUMERIC)

    positives = train_df[train_df["label"] == 1].reset_index(drop=True)
    pos_item_num, _ = standardize(positives, ITEM_NUMERIC, stats=item_stats)
    pos_user_num, _ = standardize(positives, USER_NUMERIC, stats=user_stats)
    pos_item_cat = positives["category_code"].to_numpy()

    item_code_of = {iid: i for i, iid in enumerate(positives["item_id"].unique())}
    item_code = positives["item_id"].map(item_code_of).to_numpy()
    item_freq = positives["item_id"].value_counts(normalize=True)
    log_q_by_code = np.zeros(len(item_code_of), dtype=np.float32)
    for iid, code in item_code_of.items():
        log_q_by_code[code] = np.log(item_freq[iid])

    log.info("training on %d positives (%d unique items) with in-batch softmax, "
              "batch_size=%d temperature=%.3f epochs=%d",
              len(positives), len(item_code_of), BATCH_SIZE, TEMPERATURE, EPOCHS)

    item_tower = Tower(len(ITEM_NUMERIC), n_categories=len(CATEGORIES))
    user_tower = Tower(len(USER_NUMERIC))
    opt = torch.optim.Adam(list(item_tower.parameters()) + list(user_tower.parameters()), lr=LR)

    pos_item_num_t = torch.from_numpy(pos_item_num)
    pos_item_cat_t = torch.from_numpy(pos_item_cat).long()
    pos_user_num_t = torch.from_numpy(pos_user_num)
    item_code_t = torch.from_numpy(item_code).long()
    log_q_t = torch.from_numpy(log_q_by_code)

    n = len(positives)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            b = len(idx)
            item_emb = item_tower(pos_item_num_t[idx], pos_item_cat_t[idx])
            user_emb = user_tower(pos_user_num_t[idx])

            logits = (user_emb @ item_emb.T) / TEMPERATURE
            batch_codes = item_code_t[idx]
            logits = logits - log_q_t[batch_codes].unsqueeze(0)

            same_item = batch_codes.unsqueeze(0) == batch_codes.unsqueeze(1)
            diag = torch.eye(b, dtype=torch.bool)
            logits = logits.masked_fill(same_item & ~diag, float("-inf"))

            targets = torch.arange(b)
            loss = F.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * b
        log.info("epoch %d: train softmax-ce=%.4f (random guess in a batch of ~%d = %.4f)",
                  epoch, total_loss / n, BATCH_SIZE, np.log(BATCH_SIZE))

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
    n_items = len(item_ids)

    # Positive control: where does the model rank the item the user actually
    # clicked, among all n_items candidates? 0.5 = random, regardless of what
    # recall@K reports -- see docs/adr/0003.
    sample = min(3000, len(true_cols))
    true_rank = np.array([np.where(ranks[i] == true_cols[i])[0][0] for i in range(sample)])
    pct_rank = true_rank / n_items
    log.info("positive-control: percentile rank of the true item (0.5=random, lower=better): "
              "mean=%.4f median=%.4f (n=%d)", pct_rank.mean(), np.median(pct_rank), sample)

    # Popularity baseline: always retrieve the globally most-impressed items,
    # no personalization at all. recall@K vs random alone doesn't say whether
    # the model learned anything useful -- this is the baseline that does.
    pop_score = latest_items.set_index("item_id").loc[item_ids, "item_impressions_before"].to_numpy()
    pop_order = np.argsort(-pop_score)

    log.info("recall@K on %d held-out clicks (n_items=%d):", len(valid_pos), n_items)
    for k in RECALL_KS:
        hit_model = (ranks[:, :k] == true_cols[:, None]).any(axis=1).mean()
        pop_topk = set(pop_order[:k].tolist())
        hit_pop = np.mean([tc in pop_topk for tc in true_cols])
        log.info("  recall@%-4d model=%.4f  popularity=%.4f  random=%.4f",
                  k, hit_model, hit_pop, k / n_items)

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
