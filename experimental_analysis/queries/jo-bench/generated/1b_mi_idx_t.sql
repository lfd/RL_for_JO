SELECT * FROM movie_info_idx AS mi_idx, title AS t WHERE t.production_year BETWEEN 2005 AND 2010 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id;