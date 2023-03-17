SELECT * FROM movie_info AS mi, cast_info AS ci, name AS n WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND mi.info IN ('Horror', 'Thriller') AND mi.note IS NULL AND n.gender IS NOT NULL AND n.gender = 'f' AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id;