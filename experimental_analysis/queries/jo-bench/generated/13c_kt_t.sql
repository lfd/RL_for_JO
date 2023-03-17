SELECT * FROM title AS t, kind_type AS kt WHERE kt.kind = 'movie' AND t.title != '' AND (t.title LIKE 'Champion%' OR t.title LIKE 'Loser%') AND kt.id = t.kind_id AND t.kind_id = kt.id;