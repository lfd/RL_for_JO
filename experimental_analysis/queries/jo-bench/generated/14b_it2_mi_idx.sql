SELECT * FROM movie_info_idx AS mi_idx, info_type AS it2 WHERE it2.info = 'rating' AND mi_idx.info > '6.0' AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;