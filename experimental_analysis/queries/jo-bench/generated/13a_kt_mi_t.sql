SELECT * FROM kind_type AS kt, movie_info AS mi, title AS t WHERE kt.kind = 'movie' AND mi.movie_id = t.id AND t.id = mi.movie_id AND kt.id = t.kind_id AND t.kind_id = kt.id;