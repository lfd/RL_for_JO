SELECT * FROM kind_type AS kt, title AS t WHERE kt.kind = 'movie' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id;