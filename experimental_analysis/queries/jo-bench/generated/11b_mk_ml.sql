SELECT * FROM movie_link AS ml, movie_keyword AS mk WHERE ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id;