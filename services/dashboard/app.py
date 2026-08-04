"""Showcase UI for the recommendation engine. Ops metrics live in Grafana (:3000);
this hits the live API, Postgres, and Redis directly -- nothing here is mocked.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime

import pandas as pd
import plotly.express as px
import psycopg
import redis
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
PG_DSN = os.getenv("PG_DSN", "postgresql://rtrec:rtrec@postgres:5432/rtrec")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
N_ITEMS = int(os.getenv("N_ITEMS", "2000"))
N_USERS = int(os.getenv("N_USERS", "5000"))
MODEL_METRICS_PATH = os.getenv("MODEL_METRICS_PATH", "/models/ranker_metrics.json")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000/d/recsys-observability")

st.set_page_config(page_title="Real-time RecSys", page_icon="🎯", layout="wide")

_r = redis.from_url(REDIS_URL, decode_responses=True)

st.title("🎯 Real-time Recommendation Engine")
st.caption(
    "Everything below hits the live API, Postgres, and Redis directly -- "
    "this is the running system, not a mockup. For full latency/pipeline "
    f"metrics see the [Grafana dashboard]({GRAFANA_URL})."
)

tab_rec, tab_stream, tab_model = st.tabs(
    ["🎯 Live Recommendations", "📡 Event Stream", "📊 Catalog & Model"]
)

with tab_rec:
    st.subheader("Call the live /recommend endpoint")
    st.session_state.setdefault("user_id", "u_76")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col3:
        st.write("")
        st.write("")
        if st.button("🎲 Random warm user"):
            st.session_state["user_id"] = f"u_{random.randint(0, N_USERS - 1)}"
    with col4:
        st.write("")
        st.write("")
        if st.button("🧊 Random new user"):
            st.session_state["user_id"] = f"u_{random.randint(N_USERS + 1, N_USERS + 1_000_000)}"
    with col1:
        user_id = st.text_input("user_id", key="user_id")
    with col2:
        k = st.slider("k (items)", 1, 30, 10)

    try:
        resp = requests.get(f"{API_URL}/recommend", params={"user_id": user_id, "k": k}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        data = None

    if data:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("user_id", data["user_id"])
        m2.metric("Cold start?", "Yes" if data["cold_start"] else "No")
        m3.metric("Server latency", f"{data['latency_ms']:.1f} ms")
        m4.metric("Items returned", len(data["items"]))

        items = data["items"]
        if not items:
            st.warning("No recommendations returned (empty candidate pool).")
        else:
            df = pd.DataFrame(items)
            display_cols = ["item_id", "category", "price_tier", "retrieved_via",
                             "reason", "score", "model_score"]
            st.dataframe(df[[c for c in display_cols if c in df.columns]],
                         use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                if df["model_score"].notna().any():
                    long_df = df.melt(id_vars="item_id", value_vars=["score", "model_score"],
                                       var_name="ranker", value_name="value")
                    fig = px.bar(long_df, x="item_id", y="value", color="ranker", barmode="group",
                                 title="Heuristic score vs LightGBM model score, per item")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No model score on this response (cold-start users stay on the "
                             "heuristic fallback -- see ADR, RANKING_MODE in services/api).")
            with c2:
                if "retrieved_via" in df.columns:
                    counts = df["retrieved_via"].value_counts().reset_index()
                    counts.columns = ["source", "count"]
                    fig = px.pie(counts, names="source", values="count",
                                 title="Where these recommendations came from")
                    st.plotly_chart(fig, use_container_width=True)

with tab_stream:
    @st.fragment(run_every="3s")
    def live_stream() -> None:
        with psycopg.connect(PG_DSN, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) FILTER (WHERE event_type = 'impression'),
                       count(*) FILTER (WHERE event_type = 'click'),
                       count(*) FILTER (WHERE event_type = 'add_to_cart'),
                       count(*) FILTER (WHERE event_type = 'purchase')
                FROM events WHERE ts > now() - interval '1 minute'
            """)
            imp, clk, atc, pur = cur.fetchone()

            cur.execute("""
                SELECT count(*) FILTER (WHERE event_type = 'click')::float
                    / NULLIF(count(*) FILTER (WHERE event_type = 'impression'), 0)
                FROM events WHERE ts > now() - interval '1 hour'
            """)
            ctr = cur.fetchone()[0] or 0.0

            cur.execute("""
                SELECT date_trunc('minute', ts) AS minute, event_type, count(*)
                FROM events WHERE ts > now() - interval '15 minutes'
                GROUP BY 1, 2 ORDER BY 1
            """)
            rows = cur.fetchall()

        st.caption(f"Auto-refreshing every 3s -- last updated {datetime.now().strftime('%H:%M:%S')}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Impressions / min", imp)
        c2.metric("Clicks / min", clk)
        c3.metric("Add-to-carts / min", atc)
        c4.metric("Purchases / min", pur)
        c5.metric("Simulated CTR (1h)", f"{ctr:.2%}")
        st.caption(
            "⚠️ Simulated CTR reflects the raw event stream, **not this system's "
            "recommendations** -- the simulator samples independently and never calls "
            "/recommend. See docs/adr/0002-simulator-recommender-decoupling.md."
        )

        if rows:
            hist_df = pd.DataFrame(rows, columns=["minute", "event_type", "count"])
            fig = px.line(hist_df, x="minute", y="count", color="event_type", markers=True,
                          title="Raw events per minute (last 15m)")
            st.plotly_chart(fig, use_container_width=True, key=f"stream-{datetime.now().timestamp()}")

    live_stream()

with tab_model:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Offline ranker metrics")
        st.caption("As of the last training run -- not live. See training/train_ranker.py.")
        if os.path.exists(MODEL_METRICS_PATH):
            with open(MODEL_METRICS_PATH) as f:
                m = json.load(f)
            mc1, mc2 = st.columns(2)
            mc1.metric("Validation AUC", f"{m['auc']:.4f}")
            mc2.metric("Validation NDCG@10", f"{m['ndcg10']:.4f}")
            st.caption(f"Trained at {datetime.fromtimestamp(m['trained_at']).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info(f"No metrics file at {MODEL_METRICS_PATH} yet -- run training/train_ranker.py.")

    with col2:
        st.subheader("Live catalog coverage")
        st.caption("Distinct items actually served in recommendations vs. the full catalog.")
        served = _r.zrange("obs:served:1h", 0, -1, withscores=True)
        coverage = len(served) / N_ITEMS if N_ITEMS else 0.0
        st.metric("Distinct items served", f"{len(served)} / {N_ITEMS}", f"{coverage:.1%} coverage")

    if served:
        top = sorted(served, key=lambda x: -x[1])[:15]
        top_df = pd.DataFrame(top, columns=["item_id", "times_served"])
        fig = px.bar(top_df, x="item_id", y="times_served", title="Most-served items (top 15)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recommendations served yet in this window -- try the Live Recommendations tab.")
