SELECT * FROM movie_companies AS mc, title AS t, movie_keyword AS mk WHERE t.production_year > 1990 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;