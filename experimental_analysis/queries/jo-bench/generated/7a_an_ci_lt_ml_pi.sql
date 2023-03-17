SELECT * FROM link_type AS lt, movie_link AS ml, cast_info AS ci, aka_name AS an, person_info AS pi WHERE an.name LIKE '%a%' AND lt.link = 'features' AND pi.note = 'Volker Boehm' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND pi.person_id = an.person_id AND an.person_id = pi.person_id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;