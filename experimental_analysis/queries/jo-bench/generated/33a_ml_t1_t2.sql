SELECT * FROM title AS t2, title AS t1, movie_link AS ml WHERE t2.production_year BETWEEN 2005 AND 2008 AND t1.id = ml.movie_id AND ml.movie_id = t1.id AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id;