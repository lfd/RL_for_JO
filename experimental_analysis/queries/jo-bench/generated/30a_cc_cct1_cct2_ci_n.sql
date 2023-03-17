SELECT * FROM name AS n, cast_info AS ci, complete_cast AS cc, comp_cast_type AS cct2, comp_cast_type AS cct1 WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete+verified' AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND n.gender = 'm' AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;