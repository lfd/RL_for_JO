SELECT * FROM movie_keyword AS mk, aka_title AS at WHERE mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id;