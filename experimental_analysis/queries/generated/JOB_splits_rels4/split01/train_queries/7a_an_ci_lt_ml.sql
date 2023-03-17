SELECT * FROM cast_info AS ci, link_type AS lt, movie_link AS ml, aka_name AS an WHERE an.name LIKE '%a%' AND lt.link = 'features' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;