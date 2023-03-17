SELECT * FROM title AS t, movie_info_idx AS mi_idx, info_type AS it2 WHERE it2.info = 'rating' AND mi_idx.info < '8.5' AND t.production_year > 2005 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;