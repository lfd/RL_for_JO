SELECT * FROM info_type AS it1, movie_info_idx AS mi_idx, movie_info AS mi, info_type AS it2 WHERE it1.info = 'genres' AND it2.info = 'rating' AND mi.info IN ('Drama', 'Horror') AND mi_idx.info > '8.0' AND mi.info_type_id = it1.id AND it1.id = mi.info_type_id AND mi_idx.info_type_id = it2.id AND it2.id = mi_idx.info_type_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;