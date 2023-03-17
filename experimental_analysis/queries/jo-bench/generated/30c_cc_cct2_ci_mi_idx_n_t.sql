SELECT * FROM name AS n, cast_info AS ci, title AS t, complete_cast AS cc, comp_cast_type AS cct2, movie_info_idx AS mi_idx WHERE cct2.kind = 'complete+verified' AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND n.gender = 'm' AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;