SELECT * FROM movie_info_idx AS mi_idx, movie_companies AS mc WHERE mi_idx.info > '7.0' AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id;