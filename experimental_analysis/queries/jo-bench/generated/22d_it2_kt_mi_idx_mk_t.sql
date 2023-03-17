SELECT * FROM movie_keyword AS mk, movie_info_idx AS mi_idx, info_type AS it2, title AS t, kind_type AS kt WHERE it2.info = 'rating' AND kt.kind IN ('movie', 'episode') AND mi_idx.info < '8.5' AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;