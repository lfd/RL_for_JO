SELECT * FROM title AS t, movie_info_idx AS mi_idx, movie_info AS mi, info_type AS it2 WHERE it2.info = 'rating' AND mi.info IN ('Drama', 'Horror') AND mi_idx.info > '8.0' AND t.production_year BETWEEN 2005 AND 2008 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mi_idx.info_type_id = it2.id AND it2.id = mi_idx.info_type_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;