SELECT * FROM movie_companies AS mc, movie_keyword AS mk WHERE mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;