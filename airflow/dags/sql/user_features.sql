-- Per-user windowed aggregates.
-- Purely declarative: grouping and counting only. Every derived metric
-- (ratios, smoothing, normalization) is computed task-side in pandas.
--
-- Params: %(window_end)s  (timestamptz, exclusive upper bound)
--         %(hours)s       (int, window length)

SELECT
    user_id,
    %(hours)s::int                                              AS window_hours,
    count(*) FILTER (WHERE event_type = 'impression')            AS impressions,
    count(*) FILTER (WHERE event_type = 'click')                 AS clicks,
    count(*) FILTER (WHERE event_type = 'add_to_cart')           AS add_to_carts,
    count(*) FILTER (WHERE event_type = 'purchase')              AS purchases,
    coalesce(sum(value) FILTER (WHERE event_type = 'purchase'), 0) AS revenue,
    count(DISTINCT item_id) FILTER (WHERE event_type = 'click')  AS distinct_items_clicked,
    count(DISTINCT session_id)                                   AS sessions,
    bool_or(is_new_user)                                         AS is_new_user,
    max(ts)                                                      AS last_seen_ts
FROM events
WHERE ts >= %(window_end)s - make_interval(hours => %(hours)s)
  AND ts <  %(window_end)s
GROUP BY user_id;
