# ADR 0001 — Aggregate in SQL, transform in pandas

**Status:** accepted · **Date:** 2026-08-03

## Context

The batch feature pipeline computes 1h/1d/7d windowed aggregates per user and per
item, writes them to Parquet for training, and pushes them to Redis for serving.
Where that computation runs is a real choice, and the whole system is deployed on a
single small VPS shared with Redpanda, Postgres, Redis, and the streaming services.

Two obvious options:

- **All in pandas inside the Airflow task.** `SELECT *` over the window, then
  `groupby` in the worker.
- **All in SQL.** `GROUP BY` in Postgres, task only ships the result.

## Decision

Heavy aggregation runs in SQL. The task receives the aggregated result — thousands
of rows, not millions — and does the light work in pandas: pivoting windows into
columns, deriving smoothed ratios, serializing to Parquet, and writing to Redis.

## Rationale

**Memory is the binding constraint, not CPU.** A 7-day window at ~8 events/s is on
the order of 5M rows. Loading that into a worker costs multiples of its on-disk size
once `groupby` and `merge` allocate intermediates. Postgres aggregates within
`work_mem` and spills to disk when it must, returning only the grouped output. In
practice this is the difference between an 8GB host and a 16GB one — roughly 2x the
monthly bill for the same work.

**Move the computation to the data.** Aggregating in the worker means transferring
every raw row across the wire first. Colocated today that is nearly free; the moment
Postgres becomes a managed instance, it is billed egress. The SQL boundary keeps that
door open at no present cost.

**pandas still earns its place.** Reshaping a few thousand aggregated rows, computing
Bayesian-smoothed CTR, and writing Parquet is more readable and far more testable in
pandas than in SQL. The same transform code is reused when generating training
datasets.

## Consequences

- Window logic lives in `.sql` files, which are harder to unit test than Python.
  Mitigated by keeping the queries purely declarative — no business rules beyond
  grouping and counting.
- Two languages in one pipeline. The boundary is deliberately sharp: SQL never
  computes a derived metric, pandas never touches a raw event.
- If the event volume ever outgrows a single Postgres, the aggregation moves to a
  proper engine (Spark, DuckDB over Parquet). The task-side transform is unaffected,
  because it only ever sees aggregated input.
