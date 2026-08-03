"""Real-time serving.

Phase 4 of the roadmap. In this scaffold the ranker is heuristic (online CTR +
category affinity - fatigue penalty); swapping in two-tower + LightGBM happens
inside `rank()` without touching the rest of the service.

Contract: p50 < 100ms. All the cost is Redis I/O, issued as ONE pipeline.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Query
from pydantic import BaseModel

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CANDIDATE_POOL = 200

_r: redis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _r
    _r = redis.from_url(REDIS_URL, decode_responses=True)
    yield
    await _r.aclose()


app = FastAPI(title="Real-time Recommendation Engine", version="0.1.0", lifespan=lifespan)


class Recommendation(BaseModel):
    item_id: str
    score: float
    reason: str


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


def rank(candidates: list[tuple[str, float]], cats: dict, item_stats: list[dict],
         seen: list[str | None], cold_start: bool) -> list[Recommendation]:
    """Lightweight ranker. Replace with LightGBM/MLP loaded from MODEL_DIR."""
    out = []
    for (item_id, pop), stats, times_seen in zip(candidates, item_stats, seen):
        imps = float(stats.get("impressions", 0)) or 1.0
        clicks = float(stats.get("clicks", 0))
        ctr = (clicks + 1.0) / (imps + 20.0)          # Bayesian smoothing
        affinity = 1.0 + sum(float(v) for v in cats.values()) * 0.0  # placeholder for category affinity
        fatigue = 0.6 ** float(times_seen or 0)        # penalize items already seen a lot
        score = ctr * affinity * fatigue + (0.0002 * pop if cold_start else 0.0)
        reason = "popularity (cold-start)" if cold_start else "online_ctr x fatigue"
        out.append(Recommendation(item_id=item_id, score=round(score, 6), reason=reason))
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
