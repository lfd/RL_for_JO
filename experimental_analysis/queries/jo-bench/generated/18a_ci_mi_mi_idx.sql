SELECT * FROM movie_info AS mi, cast_info AS ci, movie_info_idx AS mi_idx WHERE ci.note IN ('(producer)', '(executive producer)') AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;