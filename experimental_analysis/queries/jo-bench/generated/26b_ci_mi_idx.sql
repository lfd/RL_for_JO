SELECT * FROM cast_info AS ci, movie_info_idx AS mi_idx WHERE mi_idx.info > '8.0' AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id;