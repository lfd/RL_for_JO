SELECT * FROM movie_companies AS mc, title AS t, movie_info_idx AS mi_idx, info_type AS it2 WHERE it2.info = 'rating' AND mi_idx.info > '8.0' AND t.production_year BETWEEN 2005 AND 2008 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mi_idx.info_type_id = it2.id AND it2.id = mi_idx.info_type_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id;