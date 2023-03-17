SELECT * FROM movie_keyword AS mk, title AS t1, movie_link AS ml, title AS t2 WHERE t1.id = mk.movie_id AND mk.movie_id = t1.id AND ml.movie_id = t1.id AND t1.id = ml.movie_id AND ml.linked_movie_id = t2.id AND t2.id = ml.linked_movie_id;