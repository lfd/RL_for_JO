SELECT * FROM movie_info AS mi, cast_info AS ci, info_type AS it1, name AS n, title AS t WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND mi.info IN ('Horror', 'Thriller') AND mi.note IS NULL AND n.gender IS NOT NULL AND n.gender = 'f' AND t.production_year BETWEEN 2008 AND 2014 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;