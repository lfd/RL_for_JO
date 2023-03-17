SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx, movie_companies AS mc WHERE it2.info = 'rating' AND mi_idx.info > '8.0' AND mi_idx.info_type_id = it2.id AND it2.id = mi_idx.info_type_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id;