SELECT * FROM title AS t1, movie_link AS ml WHERE t1.id = ml.movie_id AND ml.movie_id = t1.id;