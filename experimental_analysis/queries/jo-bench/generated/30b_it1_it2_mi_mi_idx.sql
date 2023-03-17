SELECT * FROM info_type AS it1, info_type AS it2, movie_info_idx AS mi_idx, movie_info AS mi WHERE it1.info = 'genres' AND it2.info = 'votes' AND mi.info IN ('Horror', 'Thriller') AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;