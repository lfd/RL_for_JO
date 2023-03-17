SELECT * FROM title AS t, movie_link AS ml, cast_info AS ci, aka_name AS an WHERE an.name LIKE '%a%' AND t.production_year BETWEEN 1980 AND 1984 AND t.id = ci.movie_id AND ci.movie_id = t.id AND ml.linked_movie_id = t.id AND t.id = ml.linked_movie_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;