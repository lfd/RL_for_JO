SELECT * FROM movie_link AS ml, title AS t2 WHERE t2.production_year = 2007 AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id;