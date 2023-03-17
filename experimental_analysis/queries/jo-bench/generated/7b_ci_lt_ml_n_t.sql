SELECT * FROM link_type AS lt, movie_link AS ml, title AS t, cast_info AS ci, name AS n WHERE lt.link = 'features' AND n.name_pcode_cf LIKE 'D%' AND n.gender = 'm' AND t.production_year BETWEEN 1980 AND 1984 AND ci.person_id = n.id AND n.id = ci.person_id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ml.linked_movie_id = t.id AND t.id = ml.linked_movie_id AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;