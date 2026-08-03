-- Per-item windowed aggregates.
--
-- Conversions are attributed to the window they LAND in, not the one the click
-- happened in. The simulator delays purchases by ~45s, so a click near a window
-- boundary and its purchase fall in different windows. Correcting that requires
-- joining on session_id and attributing to click time -- a real attribution model,
-- deliberately out of scope here. The consequence is that short windows understate
-- conversion; the 7d window is the one to trust for training labels.

SELECT
    item_id,
    %(hours)s::int                                               AS window_hours,
    max(category)                                                AS category,
    max(price_tier)                                              AS price_tier,
    count(*) FILTER (WHERE event_type = 'impression')             AS impressions,
    count(*) FILTER (WHERE event_type = 'click')                  AS clicks,
    count(*) FILTER (WHERE event_type = 'purchase')               AS purchases,
    coalesce(sum(value) FILTER (WHERE event_type = 'purchase'), 0) AS revenue,
    count(DISTINCT user_id) FILTER (WHERE event_type = 'click')   AS distinct_users_clicked,
    avg(position) FILTER (WHERE event_type = 'impression')        AS avg_position
FROM events
WHERE ts >= %(window_end)s - make_interval(hours => %(hours)s)
  AND ts <  %(window_end)s
GROUP BY item_id;
