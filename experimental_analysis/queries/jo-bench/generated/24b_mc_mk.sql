SELECT * FROM movie_companies AS mc, movie_keyword AS mk WHERE mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;