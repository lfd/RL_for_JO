SELECT * FROM movie_info_idx AS mi_idx, cast_info AS ci, complete_cast AS cc, movie_keyword AS mk WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id;