SELECT * FROM aka_name AS an, cast_info AS ci, movie_link AS ml WHERE an.name LIKE '%a%' AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = ml.linked_movie_id AND ml.linked_movie_id = ci.movie_id;