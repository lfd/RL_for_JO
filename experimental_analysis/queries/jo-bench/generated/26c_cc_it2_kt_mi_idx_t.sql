SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx, title AS t, kind_type AS kt, complete_cast AS cc WHERE it2.info = 'rating' AND kt.kind = 'movie' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;