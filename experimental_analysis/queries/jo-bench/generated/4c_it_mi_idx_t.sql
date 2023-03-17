SELECT * FROM title AS t, movie_info_idx AS mi_idx, info_type AS it WHERE it.info = 'rating' AND mi_idx.info > '2.0' AND t.production_year > 1990 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND it.id = mi_idx.info_type_id AND mi_idx.info_type_id = it.id;