SELECT * FROM info_type AS it2, cast_info AS ci, name AS n, movie_info_idx AS mi_idx WHERE it2.info = 'rating' AND mi_idx.info > '8.0' AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;