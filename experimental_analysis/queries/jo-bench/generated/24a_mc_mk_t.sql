SELECT * FROM movie_companies AS mc, movie_keyword AS mk, title AS t WHERE t.production_year > 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;