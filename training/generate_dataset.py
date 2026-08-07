#!/usr/bin/env python3
"""
Synthetic event generator for the realtime-recsys offline store.

Produces an `items` dimension table and a narrow `events` log with the
behavioural mechanics that make the dataset worth training on:

  * Zipf item popularity, with ~10% of the head rotating over the window
  * within-session fatigue: nth impression of an item gets 0.55^(n-1) CTR
  * position bias: slot 0 clicks ~3x as often as slot 19
  * diurnal + weekend volume seasonality
  * cold-start users (arrive mid-window) and cold-start items (never impressed)
  * realistic click (0-5s) and purchase (20-90s, median ~45s) lags

Design notes
------------
Generation is per-day and resumable. Each day is generated deterministically
from (SEED, day_ordinal), written to a CSV, then loaded in a single transaction
that truncates the day's partition, COPYs, and records progress. Re-running
after a crash resumes at the first incomplete day and cannot double-load.

Rows are emitted as a narrow event log: an impression that gets clicked
produces two rows, one that converts produces three. `impression_id` ties them
together, which is what makes the fatigue signal recoverable at training time.

Usage
-----
    export RECSYS_DSN='postgresql://user:pass@localhost:5432/recsys'
    python generate_dataset.py                 # full 28-day load
    python generate_dataset.py --days 3 --scale 0.01   # quick smoke test
    python generate_dataset.py --dry-run --out ./csv   # CSVs only, no DB
    python generate_dataset.py --validate-only         # re-run checks
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 20260803

N_DAYS = 28
N_USERS = 50_000
N_ITEMS = 20_000
N_HELDOUT_ITEMS = 500       # in `items`, never in `events` -> cold-start set
N_CATEGORIES = 40
N_BRANDS = 500

IMPRESSIONS_PER_DAY = 714_000        # ~20M over 28 days
SESSION_SIZE_MIN, SESSION_SIZE_MAX = 6, 8
SESSION_MAX_SPAN_S = 30 * 60
SESSION_GAP_MEAN_S = 90              # Poisson-ish spacing between impressions

BASE_CTR = 0.05
PURCHASE_RATE = 0.04                 # of clicks
FATIGUE_DECAY = 0.55
POSITION_SLOTS = 20
POSITION_RATIO = 3.0                 # slot 0 CTR / slot 19 CTR
REPEAT_PROB = 0.25                   # chance an impression re-shows an earlier item

ZIPF_S = 1.1
DRIFT_PER_DAY = 50                   # head items rotated out each day
HEAD_SIZE = 500

NEW_USER_SESSION_PROB = 0.05
ESTABLISHED_USER_FRAC = 0.60         # rest arrive spread across the window

WEEKEND_LIFT = 1.25
DIURNAL_TROUGH_HOUR = 4
DIURNAL_AMPLITUDE = 3.5              # peak/trough ratio

SURFACES = np.array(["home", "search", "category", "detail"])
SURFACE_P = np.array([0.40, 0.25, 0.20, 0.15])
DEVICES = np.array(["mobile", "desktop", "tablet"])
DEVICE_P = np.array([0.65, 0.30, 0.05])

EVENT_COLUMNS = [
    "event_id", "impression_id", "event_type", "user_id", "item_id",
    "session_id", "surface", "device", "position", "is_new_user",
    "category", "price_tier", "value", "ts", "ingested_at",
]

INDEXES = {
    "events_user_ts_idx": "CREATE INDEX events_user_ts_idx ON events (user_id, ts DESC)",
    "events_item_ts_idx": "CREATE INDEX events_item_ts_idx ON events (item_id, ts DESC)",
    "events_type_ts_idx": "CREATE INDEX events_type_ts_idx ON events (event_type, ts DESC)",
}


# --------------------------------------------------------------------------
# Static catalogue (deterministic from SEED, identical on every run)
# --------------------------------------------------------------------------

@dataclass
class Catalog:
    item_ids: np.ndarray        # (N_ITEMS,) str
    category: np.ndarray        # (N_ITEMS,) str
    brand: np.ndarray
    price_tier: np.ndarray      # (N_ITEMS,) int8
    created_at: np.ndarray      # (N_ITEMS,) datetime64[s]
    quality: np.ndarray         # (N_ITEMS,) float, per-item CTR multiplier
    active: np.ndarray          # (N_ACTIVE,) int, indices eligible for events
    user_ids: np.ndarray        # (N_USERS,) str
    user_arrival_day: np.ndarray  # (N_USERS,) int, -1 = established


def build_catalog(window_start: dt.date) -> Catalog:
    rng = np.random.default_rng(SEED)

    idx = np.arange(N_ITEMS)
    item_ids = np.array([f"i{i:06d}" for i in idx])

    # Category is a deterministic function of item_id, as the schema assumes.
    cat_idx = np.array([hash_int(s) % N_CATEGORIES for s in item_ids])
    category = np.array([f"cat_{c:02d}" for c in cat_idx])
    brand = np.array([f"brand_{b:03d}" for b in rng.integers(0, N_BRANDS, N_ITEMS)])

    # Price tier correlates with category so tiers are not uniform noise.
    tier_bias = rng.uniform(0.5, 4.5, N_CATEGORIES)[cat_idx]
    price_tier = np.clip(
        np.round(tier_bias + rng.normal(0, 0.8, N_ITEMS)), 1, 5
    ).astype(np.int8)

    # Staggered creation: 60% exist before the window, the rest arrive during it,
    # so "new item" is a real condition rather than a synthetic flag.
    pre = rng.random(N_ITEMS) < 0.60
    offset_days = np.where(
        pre,
        -rng.integers(30, 400, N_ITEMS),
        rng.integers(0, N_DAYS, N_ITEMS),
    )
    base = np.datetime64(window_start, "s")
    created_at = base + offset_days.astype("timedelta64[D]").astype("timedelta64[s]")

    quality = rng.lognormal(mean=0.0, sigma=0.45, size=N_ITEMS)

    # Hold out the tail end of the id space entirely.
    heldout = rng.choice(N_ITEMS, size=N_HELDOUT_ITEMS, replace=False)
    mask = np.ones(N_ITEMS, dtype=bool)
    mask[heldout] = False
    active = idx[mask]

    user_ids = np.array([f"u{i:06d}" for i in range(N_USERS)])
    n_established = int(N_USERS * ESTABLISHED_USER_FRAC)
    arrival = np.full(N_USERS, -1, dtype=np.int32)
    late = np.arange(n_established, N_USERS)
    arrival[late] = rng.integers(0, N_DAYS, late.size)

    return Catalog(item_ids, category, brand, price_tier, created_at,
                   quality, active, user_ids, arrival)


def hash_int(s: str) -> int:
    """Stable string hash (Python's is salted per-process)."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------
# Popularity: Zipf ranks that drift across the window
# --------------------------------------------------------------------------

def popularity_probs(cat: Catalog, day: int) -> np.ndarray:
    """
    Zipf weights over active items, with the head rotating over time.

    Recomputed from scratch for each day (28 cheap iterations) so that any day
    can be regenerated identically without replaying prior state.
    """
    n = cat.active.size
    order = np.arange(n)
    for d in range(day + 1):
        rng = np.random.default_rng([SEED, 991, d])
        hi = min(4 * HEAD_SIZE, n)
        # Both index sets must be sampled without replacement and be disjoint,
        # otherwise the swap duplicates entries and `order` is no longer a
        # permutation (which silently produces NaN weights downstream).
        out_slots = rng.choice(HEAD_SIZE, size=DRIFT_PER_DAY, replace=False)
        in_slots = HEAD_SIZE + rng.choice(
            hi - HEAD_SIZE, size=DRIFT_PER_DAY, replace=False
        )
        tmp = order[in_slots].copy()
        order[in_slots] = order[out_slots]
        order[out_slots] = tmp

    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    w = ranks ** (-ZIPF_S)
    return w / w.sum()


# --------------------------------------------------------------------------
# Seasonality
# --------------------------------------------------------------------------

def diurnal_pmf() -> np.ndarray:
    hours = np.arange(24)
    phase = 2 * np.pi * (hours - DIURNAL_TROUGH_HOUR) / 24
    curve = 1 + (DIURNAL_AMPLITUDE - 1) / 2 * (1 - np.cos(phase))
    return curve / curve.sum()


def day_volume(day_date: dt.date, scale: float) -> int:
    lift = WEEKEND_LIFT if day_date.weekday() >= 5 else 1.0
    return max(1, int(IMPRESSIONS_PER_DAY * lift * scale))


# --------------------------------------------------------------------------
# Per-day event generation
# --------------------------------------------------------------------------

def generate_day(cat: Catalog, day: int, day_date: dt.date, scale: float):
    """
    Returns a dict of parallel arrays describing every event on `day_date`.
    Fully vectorised: no per-session Python loop.
    """
    rng = np.random.default_rng([SEED, day])

    target = day_volume(day_date, scale)
    avg_size = (SESSION_SIZE_MIN + SESSION_SIZE_MAX) / 2
    n_sessions = max(1, int(round(target / avg_size)))

    sizes = rng.integers(SESSION_SIZE_MIN, SESSION_SIZE_MAX + 1, n_sessions)
    n_imp = int(sizes.sum())
    sess_of = np.repeat(np.arange(n_sessions), sizes)

    # --- session-level attributes -----------------------------------------
    is_new = rng.random(n_sessions) < NEW_USER_SESSION_PROB
    arrivals_today = np.flatnonzero(cat.user_arrival_day == day)
    eligible = np.flatnonzero(
        (cat.user_arrival_day < 0) | (cat.user_arrival_day <= day)
    )
    if arrivals_today.size == 0:
        is_new[:] = False

    s_user = np.empty(n_sessions, dtype=np.int64)
    s_user[~is_new] = rng.choice(eligible, size=int((~is_new).sum()))
    if is_new.any():
        s_user[is_new] = rng.choice(arrivals_today, size=int(is_new.sum()))

    s_device = rng.choice(len(DEVICES), size=n_sessions, p=DEVICE_P)
    s_surface = rng.choice(len(SURFACES), size=n_sessions, p=SURFACE_P)

    # Session start hour follows the diurnal curve.
    hour = rng.choice(24, size=n_sessions, p=diurnal_pmf())
    sec_in_hour = rng.integers(0, 3600, n_sessions)
    midnight = np.datetime64(day_date, "s")
    s_start = midnight + (hour * 3600 + sec_in_hour).astype("timedelta64[s]")

    # --- impression timing -------------------------------------------------
    idx_in_sess = np.arange(n_imp) - np.repeat(
        np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes
    )
    gaps = rng.exponential(SESSION_GAP_MEAN_S, n_imp).astype(np.int64)
    gaps[idx_in_sess == 0] = 0
    offset = np.cumsum(gaps) - np.repeat(
        np.concatenate([[0], np.cumsum(gaps)[np.cumsum(sizes) - 1][:-1]]), sizes
    )
    offset = np.minimum(offset, SESSION_MAX_SPAN_S)
    ts = s_start[sess_of] + offset.astype("timedelta64[s]")

    # --- item selection ----------------------------------------------------
    probs = popularity_probs(cat, day)
    picked = rng.choice(cat.active.size, size=n_imp, p=probs)

    # Re-show an earlier item from the same session sometimes. This is what
    # gives the fatigue mechanic something to bite on.
    repeat = (rng.random(n_imp) < REPEAT_PROB) & (idx_in_sess > 0)
    back = 1 + (rng.random(n_imp) * idx_in_sess).astype(np.int64)
    src = np.arange(n_imp) - back
    picked[repeat] = picked[src[repeat]]

    item_idx = cat.active[picked]

    # --- occurrence index within (session, item) ---------------------------
    n_occurrence = cumcount(sess_of, picked)

    # --- position and click model -----------------------------------------
    position = rng.integers(0, POSITION_SLOTS, n_imp)
    alpha = np.log(POSITION_RATIO) / np.log(POSITION_SLOTS)
    pos_factor = (1.0 + position) ** (-alpha)

    fatigue = FATIGUE_DECAY ** n_occurrence
    q = cat.quality[item_idx]
    p_click = BASE_CTR * pos_factor * fatigue * q
    p_click *= BASE_CTR / max(p_click.mean(), 1e-12)     # hold overall CTR at target
    p_click = np.clip(p_click, 0.0, 0.95)

    clicked = rng.random(n_imp) < p_click
    click_idx = np.flatnonzero(clicked)
    click_lag = rng.uniform(0, 5, click_idx.size)

    purchased = rng.random(click_idx.size) < PURCHASE_RATE
    purch_of = click_idx[purchased]
    purch_lag = np.clip(rng.lognormal(np.log(45), 0.35, purch_of.size), 20, 90)
    tier_of_purchase = cat.price_tier[item_idx[purch_of]]
    value = np.round(
        tier_of_purchase * 25 * rng.lognormal(0, 0.4, purch_of.size), 2
    )

    return {
        "n_imp": n_imp,
        "sess_of": sess_of,
        "item_idx": item_idx,
        "s_user": s_user,
        "s_device": s_device,
        "s_surface": s_surface,
        "is_new": is_new,
        "position": position,
        "ts": ts,
        "click_idx": click_idx,
        "click_lag": click_lag,
        "purch_of": purch_of,
        "purch_lag": purch_lag,
        "value": value,
        "rng": rng,
    }


def cumcount(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Occurrence index of each (a, b) pair among earlier entries with the same
    pair, preserving original array order. Vectorised equivalent of
    pandas' groupby(...).cumcount().
    """
    n = a.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    key = a.astype(np.int64) * (b.max() + 1) + b.astype(np.int64)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(sorted_key[1:], sorted_key[:-1], out=new_group[1:])
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))
    counts_sorted = np.arange(n) - group_start
    out = np.empty(n, dtype=np.int64)
    out[order] = counts_sorted
    return out


# --------------------------------------------------------------------------
# CSV emission
# --------------------------------------------------------------------------

def uuids(n: int, rng) -> list[str]:
    raw = rng.integers(0, 1 << 32, size=(n, 4), dtype=np.uint64)
    return [
        str(uuid.UUID(int=int(a) << 96 | int(b) << 64 | int(c) << 32 | int(d)))
        for a, b, c, d in raw
    ]


def write_day_csv(cat: Catalog, g: dict, path: Path) -> dict:
    rng = g["rng"]
    n = g["n_imp"]

    imp_ids = uuids(n, rng)
    sess_ids_by_session = uuids(int(g["sess_of"].max()) + 1, rng)
    click_ids = uuids(g["click_idx"].size, rng)
    purch_ids = uuids(g["purch_of"].size, rng)

    item_idx = g["item_idx"]
    sess_of = g["sess_of"]
    user_of = g["s_user"][sess_of]
    dev_of = g["s_device"][sess_of]
    surf_of = g["s_surface"][sess_of]
    new_of = g["is_new"][sess_of]

    ts = g["ts"]
    ingest_lag = rng.uniform(0, 2, n)

    click_ts = ts[g["click_idx"]] + (
        g["click_lag"] * 1000
    ).astype("timedelta64[ms]")
    click_pos = {int(v): i for i, v in enumerate(g["click_idx"])}
    purch_ts = np.array(
        [click_ts[click_pos[int(v)]] for v in g["purch_of"]],
        dtype="datetime64[ms]",
    ) if g["purch_of"].size else np.array([], dtype="datetime64[ms]")
    purch_ts = purch_ts + (g["purch_lag"] * 1000).astype("timedelta64[ms]")

    # Sessions run up to 30 minutes and purchases lag up to 90s, so a small
    # number of rows generated for `day_date` actually carry a ts on the
    # following day. Those cannot go into this day's partition -- they are
    # written to a spill file and loaded through the parent table instead.
    day_date = ts[0].astype("datetime64[D]").astype(dt.date) if n else None
    spill_path = path.with_name(path.name.replace("events_", "spill_"))
    opener = gzip.open if path.suffix == ".gz" else open
    n_spill = 0

    with opener(path, "wt", newline="") as fh, \
            opener(spill_path, "wt", newline="") as sfh:
        w = csv.writer(fh)
        w.writerow(EVENT_COLUMNS)
        sw = csv.writer(sfh)
        sw.writerow(EVENT_COLUMNS)

        def emit(rec, when):
            nonlocal n_spill
            if when.astype("datetime64[D]").astype(dt.date) == day_date:
                w.writerow(rec)
            else:
                sw.writerow(rec)
                n_spill += 1

        def row(event_id, impression_id, etype, i, when, val, lag):
            ii = item_idx[i]
            return (
                event_id, impression_id, etype,
                cat.user_ids[user_of[i]], cat.item_ids[ii],
                sess_ids_by_session[sess_of[i]],
                SURFACES[surf_of[i]], DEVICES[dev_of[i]],
                int(g["position"][i]), "true" if new_of[i] else "false",
                cat.category[ii], int(cat.price_tier[ii]),
                f"{val:.2f}", iso(when), iso(when + np.timedelta64(int(lag * 1000), "ms")),
            )

        for i in range(n):
            emit(row(imp_ids[i], imp_ids[i], "impression", i,
                     ts[i], 0.0, ingest_lag[i]), ts[i])

        for k, i in enumerate(g["click_idx"]):
            i = int(i)
            emit(row(click_ids[k], imp_ids[i], "click", i,
                     click_ts[k], 0.0, rng.uniform(0, 2)), click_ts[k])

        for k, i in enumerate(g["purch_of"]):
            i = int(i)
            emit(row(purch_ids[k], imp_ids[i], "purchase", i,
                     purch_ts[k], float(g["value"][k]), rng.uniform(0, 2)),
                 purch_ts[k])

    if n_spill == 0:
        spill_path.unlink(missing_ok=True)

    return {
        "impressions": n,
        "clicks": int(g["click_idx"].size),
        "purchases": int(g["purch_of"].size),
        "spill": n_spill,
    }


def iso(when) -> str:
    return np.datetime_as_string(when.astype("datetime64[ms]"), unit="ms") + "+00:00"


def write_items_csv(cat: Catalog, path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "category", "brand", "price_tier", "created_at"])
        for i in range(N_ITEMS):
            w.writerow([
                cat.item_ids[i], cat.category[i], cat.brand[i],
                int(cat.price_tier[i]),
                np.datetime_as_string(cat.created_at[i], unit="s") + "+00:00",
            ])


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

DDL_ITEMS = """
CREATE TABLE IF NOT EXISTS items (
    item_id    TEXT PRIMARY KEY,
    category   TEXT        NOT NULL,
    brand      TEXT        NOT NULL,
    price_tier SMALLINT    NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
"""

DDL_PROGRESS = """
CREATE TABLE IF NOT EXISTS _gen_progress (
    day        DATE PRIMARY KEY,
    rows       BIGINT      NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def load_spill(conn, out: Path) -> int:
    """
    Load midnight-crossing rows through the parent table, after every day
    partition is in place. Doing this inside load_day would be wrong: day N's
    TRUNCATE would delete the spill rows day N-1 had already written into it.

    Routed through the parent (so Postgres picks the partition) and made
    idempotent with ON CONFLICT, since re-running must not double-insert.
    """
    paths = sorted(out.glob("spill_*"))
    if not paths:
        return 0
    cols = ", ".join(EVENT_COLUMNS)
    total = 0
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE _spill (LIKE events)")
        for p in paths:
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rt") as fh:
                cur.copy_expert(
                    f"COPY _spill ({cols}) FROM STDIN "
                    "WITH (FORMAT csv, HEADER true)", fh
                )
        cur.execute(
            f"INSERT INTO events ({cols}) SELECT {cols} FROM _spill "
            "ON CONFLICT (event_id, ts) DO NOTHING"
        )
        total = cur.rowcount
        cur.execute("DROP TABLE _spill")
    conn.commit()
    for p in paths:
        p.unlink()
    return total


def connect(dsn: str):
    try:
        import psycopg2
    except ImportError:
        sys.exit("psycopg2 not installed. pip install psycopg2-binary "
                 "(or run with --dry-run to emit CSVs only)")
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


def prepare_db(conn, days: int) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_ITEMS)
        cur.execute(DDL_PROGRESS)
        # Backfill partitions for the historical window, plus the forward ones.
        cur.execute("SELECT ensure_event_partitions_back(%s)", (days + 2,))
        cur.execute("SELECT ensure_event_partitions(7)")
    conn.commit()


def drop_indexes(conn) -> None:
    with conn.cursor() as cur:
        for name in INDEXES:
            cur.execute(f"DROP INDEX IF EXISTS {name}")
    conn.commit()
    print("  indexes dropped for bulk load")


def create_indexes(conn) -> None:
    with conn.cursor() as cur:
        for name, ddl in INDEXES.items():
            cur.execute("SELECT 1 FROM pg_class WHERE relname = %s", (name,))
            if cur.fetchone() is None:
                print(f"  building {name} ...")
                cur.execute(ddl)
    conn.commit()


def completed_days(conn) -> set:
    with conn.cursor() as cur:
        cur.execute("SELECT day FROM _gen_progress")
        return {r[0] for r in cur.fetchall()}


def load_items(conn, path: Path) -> None:
    with conn.cursor() as cur, open(path) as fh:
        cur.execute("TRUNCATE items")
        cur.copy_expert("COPY items FROM STDIN WITH (FORMAT csv, HEADER true)", fh)
    conn.commit()


def load_day(conn, day_date: dt.date, path: Path, n_rows: int) -> None:
    """
    Truncate the day's partition and COPY into it directly, in one transaction.
    Loading into the partition rather than the parent skips tuple routing.
    """
    part = f"events_{day_date:%Y%m%d}"
    cols = ", ".join(EVENT_COLUMNS)
    opener = gzip.open if path.suffix == ".gz" else open

    with conn.cursor() as cur, opener(path, "rt") as fh:
        cur.execute(f"TRUNCATE {part}")
        cur.copy_expert(
            f"COPY {part} ({cols}) FROM STDIN WITH (FORMAT csv, HEADER true)", fh
        )
        cur.execute(
            "INSERT INTO _gen_progress (day, rows) VALUES (%s, %s) "
            "ON CONFLICT (day) DO UPDATE SET rows = EXCLUDED.rows, "
            "finished_at = now()",
            (day_date, n_rows),
        )
    conn.commit()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

CHECKS = {
    "orphan_clicks": """
        SELECT count(*) FROM events c
        WHERE c.event_type = 'click' AND NOT EXISTS (
            SELECT 1 FROM events i
            WHERE i.impression_id = c.impression_id
              AND i.event_type = 'impression')
    """,
    "orphan_purchases": """
        SELECT count(*) FROM events p
        WHERE p.event_type = 'purchase' AND NOT EXISTS (
            SELECT 1 FROM events c
            WHERE c.impression_id = p.impression_id
              AND c.event_type = 'click')
    """,
    "missing_items": """
        SELECT count(*) FROM (
            SELECT DISTINCT item_id FROM events) e
        LEFT JOIN items i USING (item_id)
        WHERE i.item_id IS NULL
    """,
}

CTR_BY_POSITION = """
    SELECT i.position,
           count(*) AS impressions,
           count(c.event_id) AS clicks,
           count(c.event_id)::float / count(*) AS ctr
    FROM events i
    LEFT JOIN events c
      ON c.impression_id = i.impression_id AND c.event_type = 'click'
    WHERE i.event_type = 'impression'
    GROUP BY 1 ORDER BY 1
"""

FATIGUE = """
    WITH imp AS (
        SELECT impression_id, session_id, item_id, ts,
               row_number() OVER (PARTITION BY session_id, item_id ORDER BY ts) AS n
        FROM events WHERE event_type = 'impression'
    )
    SELECT imp.n,
           count(*) AS impressions,
           count(c.event_id) AS clicks,
           count(c.event_id)::float / count(*) AS ctr
    FROM imp
    LEFT JOIN events c
      ON c.impression_id = imp.impression_id AND c.event_type = 'click'
    WHERE imp.n <= 4
    GROUP BY 1 ORDER BY 1
"""

SUMMARY = """
    SELECT event_type, count(*) FROM events GROUP BY 1 ORDER BY 1
"""

COVERAGE = """
    SELECT count(DISTINCT user_id), count(DISTINCT item_id),
           count(DISTINCT session_id), min(ts)::date, max(ts)::date
    FROM events
"""

HOURLY = """
    SELECT extract(hour FROM ts)::int AS h, count(*)
    FROM events WHERE event_type = 'impression'
    GROUP BY 1 ORDER BY 1
"""


def validate(conn) -> bool:
    ok = True
    print("\n" + "=" * 62)
    print("VALIDATION")
    print("=" * 62)

    with conn.cursor() as cur:
        cur.execute(SUMMARY)
        print("\nrows by event_type:")
        for etype, n in cur.fetchall():
            print(f"  {etype:<12} {n:>12,}")

        cur.execute(COVERAGE)
        u, i, s, lo, hi = cur.fetchone()
        print(f"\nusers {u:,}  items {i:,}  sessions {s:,}  span {lo} .. {hi}")

        for name, sql in CHECKS.items():
            cur.execute(sql)
            n = cur.fetchone()[0]
            status = "PASS" if n == 0 else "FAIL"
            ok &= n == 0
            print(f"\n[{status}] {name}: {n}")

        # --- position bias --------------------------------------------------
        cur.execute(CTR_BY_POSITION)
        rows = cur.fetchall()
        first, last = rows[0][3], rows[-1][3]
        ratio = first / last if last else float("inf")
        trend_ok = ratio > 1.8
        ok &= trend_ok
        print(f"\n[{'PASS' if trend_ok else 'FAIL'}] position bias: "
              f"slot 0 CTR {first:.4f} vs slot {rows[-1][0]} CTR {last:.4f} "
              f"(ratio {ratio:.2f}, want > 1.8)")
        if not trend_ok:
            print("      -> position bias was not generated; the model would "
                  "learn an easier problem than intended")

        # --- fatigue --------------------------------------------------------
        cur.execute(FATIGUE)
        rows = cur.fetchall()
        print("\nfatigue decay (want each ratio near "
              f"{FATIGUE_DECAY}):")
        base = rows[0][3] if rows else 0
        fat_ok = len(rows) > 1
        for n, imps, clicks, ctr in rows:
            rel = ctr / base if base else 0
            expect = FATIGUE_DECAY ** (n - 1)
            # Deep repeats are rare; at small --scale they carry too few
            # impressions to say anything, so they are reported but not judged.
            if imps < 5000:
                print(f"  n={n}  impressions {imps:>10,}  ctr {ctr:.4f}  "
                      f"relative {rel:.3f} (expect {expect:.3f})  [low sample]")
                continue
            flag = "" if abs(rel - expect) < 0.12 else "  <-- off"
            if flag:
                fat_ok = False
            print(f"  n={n}  impressions {imps:>10,}  ctr {ctr:.4f}  "
                  f"relative {rel:.3f} (expect {expect:.3f}){flag}")
        ok &= fat_ok
        print(f"[{'PASS' if fat_ok else 'FAIL'}] fatigue")
        if not fat_ok:
            print("      -> fatigue mechanic missing or distorted")

        # --- diurnal --------------------------------------------------------
        cur.execute(HOURLY)
        rows = cur.fetchall()
        counts = [c for _, c in rows]
        amp = max(counts) / min(counts) if min(counts) else float("inf")
        di_ok = amp > 2.0
        ok &= di_ok
        peak = rows[counts.index(max(counts))][0]
        trough = rows[counts.index(min(counts))][0]
        print(f"\n[{'PASS' if di_ok else 'FAIL'}] diurnal: peak/trough {amp:.2f} "
              f"(peak {peak:02d}h, trough {trough:02d}h, want > 2.0)")

    print("\n" + "=" * 62)
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED -- see above")
    print("=" * 62)
    return ok


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("RECSYS_DSN"))
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of full daily volume (0.01 for a smoke test)")
    ap.add_argument("--out", default="./_gen", help="CSV staging directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="write CSVs, do not touch the database")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--keep-csv", action="store_true")
    ap.add_argument("--gzip", action="store_true", help="gzip staged CSVs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    today = dt.date.today()
    window_start = today - dt.timedelta(days=args.days - 1)
    cat = build_catalog(window_start)

    conn = None
    if not args.dry_run:
        if not args.dsn:
            sys.exit("set RECSYS_DSN or pass --dsn")
        conn = connect(args.dsn)

    if args.validate_only:
        ok = validate(conn)
        conn.close()
        return 0 if ok else 1

    if conn:
        prepare_db(conn, args.days)
        drop_indexes(conn)
        done = completed_days(conn)
    else:
        done = set()

    items_csv = out / "items.csv"
    if not items_csv.exists():
        write_items_csv(cat, items_csv)
    if conn:
        load_items(conn, items_csv)
        print(f"items loaded: {N_ITEMS:,} "
              f"({N_HELDOUT_ITEMS} held out of events)")

    ext = ".csv.gz" if args.gzip else ".csv"
    totals = {"impressions": 0, "clicks": 0, "purchases": 0}

    for day in range(args.days):
        day_date = window_start + dt.timedelta(days=day)
        if day_date in done:
            print(f"[{day + 1:>2}/{args.days}] {day_date} already loaded, skipping")
            continue

        g = generate_day(cat, day, day_date, args.scale)
        path = out / f"events_{day_date:%Y%m%d}{ext}"
        stats = write_day_csv(cat, g, path)
        n_rows = sum(stats.values())

        if conn:
            load_day(conn, day_date, path, n_rows)
        if not args.keep_csv and conn:
            path.unlink()

        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        print(f"[{day + 1:>2}/{args.days}] {day_date}  "
              f"imp {stats['impressions']:>8,}  "
              f"clk {stats['clicks']:>7,}  "
              f"buy {stats['purchases']:>6,}  "
              f"spill {stats['spill']:>5,}")

    print(f"\ntotal: {totals['impressions']:,} impressions, "
          f"{totals['clicks']:,} clicks, {totals['purchases']:,} purchases")

    if conn:
        n_spill = load_spill(conn, out)
        print(f"midnight-crossing rows loaded via parent: {n_spill:,}")
        print("\nrebuilding indexes ...")
        create_indexes(conn)
        ok = validate(conn)
        conn.close()
        return 0 if ok else 1

    print("\ndry run -- CSVs written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
