SELECT * FROM name AS n, cast_info AS ci, title AS t WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND n.gender = 'm' AND t.id = ci.movie_id AND ci.movie_id = t.id AND n.id = ci.person_id AND ci.person_id = n.id;