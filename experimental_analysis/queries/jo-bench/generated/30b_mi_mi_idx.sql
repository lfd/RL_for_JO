SELECT * FROM movie_info AS mi, movie_info_idx AS mi_idx WHERE mi.info IN ('Horror', 'Thriller') AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;