SELECT * FROM movie_companies AS mc, title AS t, cast_info AS ci, movie_info_idx AS mi_idx WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id;