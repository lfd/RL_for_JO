SELECT * FROM movie_keyword AS mk, title AS t1 WHERE t1.id = mk.movie_id AND mk.movie_id = t1.id;