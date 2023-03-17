SELECT * FROM cast_info AS ci, movie_info AS mi, title AS t, movie_info_idx AS mi_idx WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND mi.info IN ('Horror', 'Thriller') AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;