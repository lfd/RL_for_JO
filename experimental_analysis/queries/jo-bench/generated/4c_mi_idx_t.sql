SELECT * FROM movie_info_idx AS mi_idx, title AS t WHERE mi_idx.info > '2.0' AND t.production_year > 1990 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id;