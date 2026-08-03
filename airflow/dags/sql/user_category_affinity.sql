-- Click counts per (user, category) over the widest window.
-- Pivoted into one column per category task-side.

SELECT
    user_id,
    category,
    count(*) AS clicks
FROM events
WHERE event_type = 'click'
  AND ts >= %(window_end)s - make_interval(hours => %(hours)s)
  AND ts <  %(window_end)s
GROUP BY user_id, category;
