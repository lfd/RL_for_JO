SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx, complete_cast AS cc WHERE it2.info = 'rating' AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;