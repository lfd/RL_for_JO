SELECT * FROM movie_info_idx AS mi_idx1, title AS t1 WHERE t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id;