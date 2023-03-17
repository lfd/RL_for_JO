SELECT * FROM movie_info_idx AS mi_idx, info_type AS it2, cast_info AS ci WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it2.info = 'votes' AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;