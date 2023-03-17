SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx, title AS t, cast_info AS ci, name AS n WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it2.info = 'votes' AND n.gender = 'm' AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;