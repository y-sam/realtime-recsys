"""Real-time serving.

Contract: p50 < 100ms. All the cost is Redis I/O, issued as ONE pipeline.
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
from fastapi import FastAPI, Query, Response
from pydantic import BaseModel

from app import metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CANDIDATE_POOL = 200

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_DIR, "ranker.txt")
MODEL_META_PATH = os.path.join(MODEL_DIR, "ranker.meta.json")
TOWER_META_PATH = os.path.join(MODEL_DIR, "two_tower.meta.json")
ITEM_EMB_PATH = os.path.join(MODEL_DIR, "item_embeddings.npy")

RANKING_MODE = os.getenv("RANKING_MODE", "model")  # "model" or "heuristic"
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "blended")  # "blended" or "popularity_only"
TWO_TOWER_K = int(os.getenv("TWO_TOWER_K", "50"))

SCORING_POSITION = 0  # position is unknown before ranking; score as if shown in the best slot

_r: redis.Redis | None = None
_booster: lgb.Booster | None = None
_model_features: list[str] | None = None
_category_codes: dict[str, int] | None = None

_item_embeddings: np.ndarray | None = None
_item_ids: list[str] | None = None
_tower_categories: list[str] | None = None
_tower_user_stats: dict | None = None
_tower_w1 = _tower_b1 = _tower_w2 = _tower_b2 = None


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


def _load_two_tower() -> None:
    global _item_embeddings, _item_ids, _tower_categories, _tower_user_stats
    global _tower_w1, _tower_b1, _tower_w2, _tower_b2
    if not (os.path.exists(ITEM_EMB_PATH) and os.path.exists(TOWER_META_PATH)):
        log.warning("no two-tower model at %s -- retrieval stays popularity-only", ITEM_EMB_PATH)
        return
    with open(TOWER_META_PATH) as f:
        meta = json.load(f)
    _item_embeddings = np.load(ITEM_EMB_PATH)
    _item_ids = meta["item_ids"]
    _tower_categories = meta["categories"]
    _tower_user_stats = meta["user_stats"]
    w = meta["user_tower_weights"]
    _tower_w1, _tower_b1 = np.array(w["w1"]), np.array(w["b1"])
    _tower_w2, _tower_b2 = np.array(w["w2"]), np.array(w["b2"])
    log.info("loaded two-tower retrieval: %d items x %d dims", *_item_embeddings.shape)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _r
    _r = redis.from_url(REDIS_URL, decode_responses=True)
    _load_model()
    _load_two_tower()
    yield
    await _r.aclose()


app = FastAPI(title="Real-time Recommendation Engine", version="0.1.0", lifespan=lifespan)


class Recommendation(BaseModel):
    item_id: str
    score: float
    reason: str
    model_score: float | None = None
    category: str | None = None
    price_tier: int | None = None
    retrieved_via: str = "popularity"


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


def _two_tower_candidates(cats: dict, total_cat_clicks: float) -> list[tuple[str, float]]:
    user_numeric_names = (["user_clicks_before", "is_new_user"]
                           + [f"user_share_{c}" for c in _tower_categories])
    values = {"user_clicks_before": total_cat_clicks, "is_new_user": 0.0}
    for c in _tower_categories:
        values[f"user_share_{c}"] = (float(cats.get(c, 0)) + 1.0) / (total_cat_clicks + 20.0)

    x = np.array([[(values[name] - _tower_user_stats[name][0]) / _tower_user_stats[name][1]
                    for name in user_numeric_names]], dtype=np.float32)
    h = np.maximum(x @ _tower_w1.T + _tower_b1, 0.0)
    user_embedding = (h @ _tower_w2.T + _tower_b2)[0]

    scores = _item_embeddings @ user_embedding
    top_idx = np.argpartition(-scores, TWO_TOWER_K)[:TWO_TOWER_K]
    return [(_item_ids[i], float(scores[i])) for i in top_idx]


def rank(candidates: list[tuple[str, float]], cats: dict, item_stats: list[dict],
         seen: list[str | None], cold_start: bool) -> list[Recommendation]:
    use_model_ranking = _booster is not None and RANKING_MODE == "model" and not cold_start

    total_cat_clicks = sum(float(v) for v in cats.values())
    out, feature_rows = [], []
    for (item_id, pop), stats, times_seen in zip(candidates, item_stats, seen):
        imps = float(stats.get("impressions", 0)) or 1.0
        clicks = float(stats.get("clicks", 0))
        ctr = (clicks + 1.0) / (imps + 20.0)
        category_clicks = float(cats.get(stats.get("category", ""), 0))
        affinity = 1.0 + (category_clicks / total_cat_clicks if total_cat_clicks else 0.0)
        fatigue = 0.6 ** float(times_seen or 0)
        score = ctr * affinity * fatigue + (0.0002 * pop if cold_start else 0.0)
        reason = "popularity (cold-start)" if cold_start else "online_ctr x fatigue x affinity"
        price_tier = int(stats["price_tier"]) if "price_tier" in stats else None
        out.append(Recommendation(item_id=item_id, score=round(score, 6), reason=reason,
                                   category=stats.get("category"), price_tier=price_tier))
        if _booster is not None:
            feature_rows.append(_model_feature_row(stats, category_clicks, total_cat_clicks,
                                                     times_seen, cold_start))

    if _booster is not None and feature_rows:
        model_scores = _booster.predict(np.array(feature_rows, dtype=float))
        for rec, model_score in zip(out, model_scores):
            rec.model_score = round(float(model_score), 6)
            if use_model_ranking:
                rec.reason = "lightgbm_ranker"

    sort_key = (lambda x: x.model_score) if use_model_ranking else (lambda x: x.score)
    return sorted(out, key=sort_key, reverse=True)


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(user_id: str, k: int = Query(10, ge=1, le=50)) -> RecommendResponse:
    t0 = time.perf_counter()

    stats, cats, recent = await fetch_user_features(user_id)
    cold_start = not stats or int(stats.get("impressions", 0)) < 5
    metrics.REQUESTS_TOTAL.labels(cold_start=str(cold_start).lower()).inc()

    t_retrieval = time.perf_counter()
    candidates = await _r.zrevrange("f:pop:1h", 0, CANDIDATE_POOL - 1, withscores=True)

    ann_eligible = _item_embeddings is not None and RETRIEVAL_MODE == "blended" and not cold_start
    two_tower_ids: set[str] = set()
    if ann_eligible:
        metrics.ANN_ELIGIBLE_TOTAL.inc()
        total_cat_clicks = sum(float(v) for v in cats.values())
        seen_ids = {item_id for item_id, _ in candidates}
        added = [pair for pair in _two_tower_candidates(cats, total_cat_clicks)
                 if pair[0] not in seen_ids]
        two_tower_ids = {item_id for item_id, _ in added}
        candidates = candidates + added
        log.info("retrieval for %s: %d popularity + %d two-tower (new)",
                  user_id, len(seen_ids), len(added))
    metrics.STAGE_SECONDS.labels(stage="retrieval").observe(time.perf_counter() - t_retrieval)

    if not candidates:
        return RecommendResponse(user_id=user_id, cold_start=cold_start,
                                 latency_ms=(time.perf_counter() - t0) * 1000, items=[])

    t_ranking = time.perf_counter()
    pipe = _r.pipeline(transaction=False)
    for item_id, _ in candidates:
        pipe.hgetall(f"f:i:{item_id}:stats")
    for item_id, _ in candidates:
        pipe.get(f"f:u:{user_id}:seen:{item_id}")
    results = await pipe.execute()

    n = len(candidates)
    ranked = rank(candidates, cats, results[:n], results[n:], cold_start)
    metrics.STAGE_SECONDS.labels(stage="ranking").observe(time.perf_counter() - t_ranking)

    t_rerank = time.perf_counter()
    top_k = ranked[:k]
    for item in top_k:
        if item.item_id in two_tower_ids:
            item.retrieved_via = "two_tower"
    if ann_eligible and any(item.item_id in two_tower_ids for item in top_k):
        metrics.ANN_HIT_TOTAL.inc()
    if top_k:
        try:
            served_pipe = _r.pipeline(transaction=False)
            for item in top_k:
                served_pipe.zincrby(metrics.SERVED_ZSET, 1, item.item_id)
            await served_pipe.execute()
        except Exception:
            log.exception("failed to record served items -- coverage metric will undercount")
    metrics.STAGE_SECONDS.labels(stage="rerank").observe(time.perf_counter() - t_rerank)

    return RecommendResponse(
        user_id=user_id,
        cold_start=cold_start,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        items=top_k,
    )


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    await metrics.collect_extra_metrics(_r)
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
