-- Offline store: raw events, the source of truth for training data.
--
-- Design notes:
--   * Partitioned by day on ts. Retention becomes DROP PARTITION (instant, no bloat)
--     instead of DELETE (slow, leaves dead tuples, needs VACUUM).
--   * PRIMARY KEY (event_id, ts): Postgres requires the partition key in every
--     unique constraint. event_id alone is unique in practice; the pair gives us
--     idempotent re-ingestion when the sink replays after a crash.
--   * Indexes are created per-partition automatically by declaring them on the parent.

CREATE TABLE IF NOT EXISTS events (
    event_id    UUID        NOT NULL,
    event_type  TEXT        NOT NULL,
    user_id     TEXT        NOT NULL,
    item_id     TEXT        NOT NULL,
    session_id  UUID        NOT NULL,
    surface     TEXT        NOT NULL,
    device      TEXT        NOT NULL,
    position    SMALLINT    NOT NULL,
    is_new_user BOOLEAN     NOT NULL,
    category    TEXT        NOT NULL,
    price_tier  SMALLINT    NOT NULL,
    value       NUMERIC(10, 2) NOT NULL DEFAULT 0,
    ts          TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, ts)
) PARTITION BY RANGE (ts);

CREATE INDEX IF NOT EXISTS events_user_ts_idx ON events (user_id, ts DESC);
CREATE INDEX IF NOT EXISTS events_item_ts_idx ON events (item_id, ts DESC);
CREATE INDEX IF NOT EXISTS events_type_ts_idx ON events (event_type, ts DESC);


-- Creates daily partitions from today through `days_ahead`.
-- Idempotent: safe to call on every sink startup and from a scheduled job.
CREATE OR REPLACE FUNCTION ensure_event_partitions(days_ahead INT DEFAULT 7)
RETURNS void AS $$
DECLARE
    d          DATE;
    part_name  TEXT;
BEGIN
    FOR i IN 0..days_ahead LOOP
        d := (CURRENT_DATE + i);
        part_name := format('events_%s', to_char(d, 'YYYYMMDD'));
        IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = part_name) THEN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF events FOR VALUES FROM (%L) TO (%L)',
                part_name, d, d + 1
            );
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;


-- Drops partitions older than `keep_days`. Raw events are disposable once the
-- features derived from them have been materialized.
CREATE OR REPLACE FUNCTION drop_old_event_partitions(keep_days INT DEFAULT 30)
RETURNS void AS $$
DECLARE
    r         RECORD;
    cutoff    DATE := CURRENT_DATE - keep_days;
    part_date DATE;
BEGIN
    FOR r IN
        SELECT c.relname
        FROM pg_class c
        JOIN pg_inherits i ON i.inhrelid = c.oid
        JOIN pg_class p ON p.oid = i.inhparent
        WHERE p.relname = 'events'
    LOOP
        BEGIN
            part_date := to_date(right(r.relname, 8), 'YYYYMMDD');
        EXCEPTION WHEN others THEN
            CONTINUE;  -- not one of ours, leave it alone
        END;
        IF part_date < cutoff THEN
            EXECUTE format('DROP TABLE IF EXISTS %I', r.relname);
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;


SELECT ensure_event_partitions(7);
