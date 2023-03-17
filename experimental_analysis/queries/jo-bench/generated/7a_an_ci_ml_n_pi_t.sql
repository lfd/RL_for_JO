SELECT * FROM name AS n, person_info AS pi, aka_name AS an, cast_info AS ci, title AS t, movie_link AS ml WHERE an.name LIKE '%a%' AND n.name_pcode_cf BETWEEN 'A' AND 'F' AND (n.gender = 'm') AND pi.note = 'Volker Boehm' AND t.production_year BETWEEN 1980 AND 1995 AND n.id = an.person_id AND an.person_id = n.id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = n.id AND n.id = ci.person_id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ml.linked_movie_id = t.id AND t.id = ml.linked_movie_id AND pi.person_id = an.person_id AND an.person_id = pi.person_id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;