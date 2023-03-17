SELECT * FROM title AS t, movie_link AS ml, cast_info AS ci, name AS n WHERE n.name_pcode_cf BETWEEN 'A' AND 'F' AND (n.gender = 'm') AND t.production_year BETWEEN 1980 AND 1995 AND ci.person_id = n.id AND n.id = ci.person_id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ml.linked_movie_id = t.id AND t.id = ml.linked_movie_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;