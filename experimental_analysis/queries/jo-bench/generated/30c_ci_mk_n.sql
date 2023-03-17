SELECT * FROM movie_keyword AS mk, cast_info AS ci, name AS n WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND n.gender = 'm' AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id;