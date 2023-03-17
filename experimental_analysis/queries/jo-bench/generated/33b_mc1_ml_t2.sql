SELECT * FROM movie_companies AS mc1, title AS t2, movie_link AS ml WHERE t2.production_year = 2007 AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id;