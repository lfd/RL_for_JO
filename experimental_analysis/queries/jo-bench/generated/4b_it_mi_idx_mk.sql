SELECT * FROM movie_keyword AS mk, movie_info_idx AS mi_idx, info_type AS it WHERE it.info = 'rating' AND mi_idx.info > '9.0' AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND it.id = mi_idx.info_type_id AND mi_idx.info_type_id = it.id;