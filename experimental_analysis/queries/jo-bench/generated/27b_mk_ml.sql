SELECT * FROM movie_keyword AS mk, movie_link AS ml WHERE ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id;