"""Real-time serving.

Phase 4 of the roadmap. The heuristic ranker (online CTR + category affinity -
fatigue penalty) still decides what's returned and in what order. The trained
LightGBM model (see training/) runs in shadow alongside it -- scored and reported
as `model_score` on every item, but not used for ranking yet. That's the point:
compare the two on live traffic before trusting the model with the real decision.

Contract: p50 < 100ms. All the cost is Redis I/O, issued as ONE pipeline (the
model, once loaded, adds one in-process batched predict() call -- no extra I/O).
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import lightgbm as lgb
import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CANDIDATE_POOL = 200

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_DIR, "ranker.txt")
MODEL_META_PATH = os.path.join(MODEL_DIR, "ranker.meta.json")

# The model was trained on impressions at their actual displayed slate position,
# but at scoring time we don't know final position yet -- ranking determines it.
# Score every candidate as if shown in the best slot, so the model's relative
# ordering of items reflects item/user quality, not an arbitrary retrieval-pool index.
SCORING_POSITION = 0

_r: redis.Redis | None = None
_booster: lgb.Booster | None = None
_model_features: list[str] | None = None
_category_codes: dict[str, int] | None = None


def _load_model() -> None:
    global _booster, _model_features, _category_codes
    if not (os.path.exists(MODEL_PATH) and os.path.exists(MODEL_META_PATH)):
        log.warning("no model at %s -- shadow scoring disabled", MODEL_PATH)
        return
    with open(MODEL_META_PATH) as f:
        meta = json.load(f)
    _booster = lgb.Booster(model_file=MODEL_PATH)
    _model_features = meta["features"]
    _category_codes = meta["category_codes"]
    log.info("loaded model from %s (%d features)", MODEL_PATH, len(_model_features))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _r
    _r = redis.from_url(REDIS_URL, decode_responses=True)
    _load_model()
    yield
    await _r.aclose()


app = FastAPI(title="Real-time Recommendation Engine", version="0.1.0", lifespan=lifespan)


class Recommendation(BaseModel):
    item_id: str
    score: float
    reason: str
    model_score: float | None = None  # shadow LightGBM score; None until a model is loaded


class RecommendResponse(BaseModel):
    user_id: str
    cold_start: bool
    latency_ms: float
    items: list[Recommendation]


@app.get("/health")
async def health() -> dict:
    await _r.ping()
    return {"status": "ok"}


async def fetch_user_features(user_id: str) -> tuple[dict, dict, list[str]]:
    pipe = _r.pipeline(transaction=False)
    pipe.hgetall(f"f:u:{user_id}:stats")
    pipe.hgetall(f"f:u:{user_id}:cat")
    pipe.lrange(f"f:u:{user_id}:recent", 0, 19)
    stats, cats, recent = await pipe.execute()
    return stats, cats, recent


def _model_feature_row(stats: dict, category_clicks: float, total_cat_clicks: float,
                        times_seen: str | None, cold_start: bool) -> list[float]:
    """One row in the exact column order training/train_ranker.py saved to ranker.meta.json."""
    # price_tier/category may briefly be absent for an item that hasn't had a fresh
    # impression since the consumer started writing them; LightGBM handles NaN as missing.
    category_code = _category_codes.get(stats.get("category", ""), float("nan"))
    price_tier = float(stats["price_tier"]) if "price_tier" in stats else float("nan")
    row = {
        "position": SCORING_POSITION,
        "is_new_user": int(cold_start),
        "price_tier": price_tier,
        "category": category_code,
        "item_impressions_before": float(stats.get("impressions", 0)),
        "item_clicks_before": float(stats.get("clicks", 0)),
        "item_ctr_before": (float(stats.get("clicks", 0)) + 1.0) / (float(stats.get("impressions", 0)) + 20.0),
        "user_clicks_before": total_cat_clicks,
        "user_category_clicks_before": category_clicks,
        "user_category_share_before": (category_clicks + 1.0) / (total_cat_clicks + 20.0),
        "user_item_impressions_before": float(times_seen or 0),
    }
    return [row[name] for name in _model_features]


def rank(candidates: list[tuple[str, float]], cats: dict, item_stats: list[dict],
         seen: list[str | None], cold_start: bool) -> list[Recommendation]:
    """Heuristic ranker decides order/return. The trained model (if loaded) scores
    the same candidates in shadow -- see module docstring."""
    total_cat_clicks = sum(float(v) for v in cats.values())
    out, feature_rows = [], []
    for (item_id, pop), stats, times_seen in zip(candidates, item_stats, seen):
        imps = float(stats.get("impressions", 0)) or 1.0
        clicks = float(stats.get("clicks", 0))
        ctr = (clicks + 1.0) / (imps + 20.0)          # Bayesian smoothing
        category_clicks = float(cats.get(stats.get("category", ""), 0))
        affinity = 1.0 + (category_clicks / total_cat_clicks if total_cat_clicks else 0.0)
        fatigue = 0.6 ** float(times_seen or 0)        # penalize items already seen a lot
        score = ctr * affinity * fatigue + (0.0002 * pop if cold_start else 0.0)
        reason = "popularity (cold-start)" if cold_start else "online_ctr x fatigue x affinity"
        out.append(Recommendation(item_id=item_id, score=round(score, 6), reason=reason))
        if _booster is not None:
            feature_rows.append(_model_feature_row(stats, category_clicks, total_cat_clicks,
                                                     times_seen, cold_start))

    if _booster is not None and feature_rows:
        model_scores = _booster.predict(np.array(feature_rows, dtype=float))
        for rec, model_score in zip(out, model_scores):
            rec.model_score = round(float(model_score), 6)

    return sorted(out, key=lambda x: x.score, reverse=True)


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(user_id: str, k: int = Query(10, ge=1, le=50)) -> RecommendResponse:
    t0 = time.perf_counter()

    stats, cats, recent = await fetch_user_features(user_id)
    cold_start = not stats or int(stats.get("impressions", 0)) < 5

    # RETRIEVAL: today = top popular in the last hour.
    # Phase 3: ANN over two-tower embeddings, with this ZSET as the fallback.
    candidates = await _r.zrevrange("f:pop:1h", 0, CANDIDATE_POOL - 1, withscores=True)
    if not candidates:
        return RecommendResponse(user_id=user_id, cold_start=True,
                                 latency_ms=(time.perf_counter() - t0) * 1000, items=[])

    pipe = _r.pipeline(transaction=False)
    for item_id, _ in candidates:
        pipe.hgetall(f"f:i:{item_id}:stats")
    for item_id, _ in candidates:
        pipe.get(f"f:u:{user_id}:seen:{item_id}")
    results = await pipe.execute()

    n = len(candidates)
    ranked = rank(candidates, cats, results[:n], results[n:], cold_start)

    return RecommendResponse(
        user_id=user_id,
        cold_start=cold_start,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        items=ranked[:k],
    )
