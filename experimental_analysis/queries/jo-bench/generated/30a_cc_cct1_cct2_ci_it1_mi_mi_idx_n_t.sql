SELECT * FROM info_type AS it1, movie_info AS mi, comp_cast_type AS cct1, name AS n, complete_cast AS cc, cast_info AS ci, title AS t, movie_info_idx AS mi_idx, comp_cast_type AS cct2 WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete+verified' AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND mi.info IN ('Horror', 'Thriller') AND n.gender = 'm' AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;