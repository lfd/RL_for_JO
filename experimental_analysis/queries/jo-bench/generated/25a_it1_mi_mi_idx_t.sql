SELECT * FROM movie_info AS mi, title AS t, movie_info_idx AS mi_idx, info_type AS it1 WHERE it1.info = 'genres' AND mi.info = 'Horror' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;