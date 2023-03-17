SELECT * FROM info_type AS it, person_info AS pi, cast_info AS ci, movie_link AS ml, aka_name AS an, title AS t WHERE an.name IS NOT NULL AND (an.name LIKE '%a%' OR an.name LIKE 'A%') AND it.info = 'mini biography' AND pi.note IS NOT NULL AND t.production_year BETWEEN 1980 AND 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND ml.linked_movie_id = t.id AND t.id = ml.linked_movie_id AND it.id = pi.info_type_id AND pi.info_type_id = it.id AND pi.person_id = an.person_id AND an.person_id = pi.person_id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;