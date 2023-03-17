SELECT * FROM movie_link AS ml, movie_keyword AS mk, title AS t WHERE t.production_year > 1950 AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id;