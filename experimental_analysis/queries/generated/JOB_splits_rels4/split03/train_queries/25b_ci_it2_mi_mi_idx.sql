SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx, cast_info AS ci, movie_info AS mi WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it2.info = 'votes' AND mi.info = 'Horror' AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;