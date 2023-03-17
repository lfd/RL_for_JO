SELECT * FROM movie_companies AS mc1, link_type AS lt, movie_link AS ml, movie_companies AS mc2 WHERE lt.link LIKE '%follow%' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;