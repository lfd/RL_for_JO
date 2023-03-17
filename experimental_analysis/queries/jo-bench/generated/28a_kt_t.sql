SELECT * FROM title AS t, kind_type AS kt WHERE kt.kind IN ('movie', 'episode') AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id;