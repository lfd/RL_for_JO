SELECT * FROM title AS t, movie_info AS mi, info_type AS it1, cast_info AS ci, name AS n WHERE ci.note IN ('(producer)', '(executive producer)') AND it1.info = 'budget' AND n.gender = 'm' AND n.name LIKE '%Tim%' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;