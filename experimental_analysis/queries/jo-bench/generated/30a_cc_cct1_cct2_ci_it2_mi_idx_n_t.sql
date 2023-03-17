SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx, name AS n, cast_info AS ci, complete_cast AS cc, comp_cast_type AS cct2, comp_cast_type AS cct1, title AS t WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete+verified' AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it2.info = 'votes' AND n.gender = 'm' AND t.production_year > 2000 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;