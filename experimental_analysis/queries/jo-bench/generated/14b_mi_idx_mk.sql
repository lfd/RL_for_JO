SELECT * FROM movie_keyword AS mk, movie_info_idx AS mi_idx WHERE mi_idx.info > '6.0' AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id;