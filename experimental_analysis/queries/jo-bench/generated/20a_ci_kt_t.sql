SELECT * FROM kind_type AS kt, title AS t, cast_info AS ci WHERE kt.kind = 'movie' AND t.production_year > 1950 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = ci.movie_id AND ci.movie_id = t.id;