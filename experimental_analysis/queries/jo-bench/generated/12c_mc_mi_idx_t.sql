SELECT * FROM movie_companies AS mc, title AS t, movie_info_idx AS mi_idx WHERE mi_idx.info > '7.0' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id;