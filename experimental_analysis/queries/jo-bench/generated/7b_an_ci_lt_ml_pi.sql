SELECT * FROM person_info AS pi, aka_name AS an, cast_info AS ci, movie_link AS ml, link_type AS lt WHERE an.name LIKE '%a%' AND lt.link = 'features' AND pi.note = 'Volker Boehm' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND pi.person_id = an.person_id AND an.person_id = pi.person_id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;