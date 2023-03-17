SELECT * FROM info_type AS it, person_info AS pi, cast_info AS ci, movie_link AS ml, name AS n, title AS t WHERE it.info = 'mini biography' AND n.name_pcode_cf BETWEEN 'A' AND 'F' AND (n.gender = 'm') AND pi.note IS NOT NULL AND t.production_year BETWEEN 1980 AND 2010 AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = n.id AND n.id = ci.person_id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ml.linked_movie_id = t.id AND t.id = ml.linked_movie_id AND it.id = pi.info_type_id AND pi.info_type_id = it.id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;