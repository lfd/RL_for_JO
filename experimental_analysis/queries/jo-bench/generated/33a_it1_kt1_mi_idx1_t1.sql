SELECT * FROM movie_info_idx AS mi_idx1, info_type AS it1, title AS t1, kind_type AS kt1 WHERE it1.info = 'rating' AND kt1.kind IN ('tv series') AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id AND kt1.id = t1.kind_id AND t1.kind_id = kt1.id;