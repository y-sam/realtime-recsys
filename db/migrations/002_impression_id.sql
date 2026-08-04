-- Ties click/add_to_cart/purchase events to the specific impression that caused them.
--
-- The fatigue mechanic re-impresses the same item within a session, so item_id alone
-- can't tell which impression a later click belongs to -- that ambiguity mislabels
-- exactly the signal the simulator exists to produce. Impression rows set this to
-- their own event_id; click/add_to_cart/purchase rows carry the originating
-- impression's id. Nullable because pre-migration rows have no impression to point to.
ALTER TABLE events ADD COLUMN impression_id UUID;
