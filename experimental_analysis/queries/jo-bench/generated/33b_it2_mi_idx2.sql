SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx2 WHERE it2.info = 'rating' AND mi_idx2.info < '3.0' AND it2.id = mi_idx2.info_type_id AND mi_idx2.info_type_id = it2.id;