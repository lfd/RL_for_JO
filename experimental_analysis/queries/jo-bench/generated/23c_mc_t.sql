SELECT * FROM title AS t, movie_companies AS mc WHERE t.production_year > 1990 AND t.id = mc.movie_id AND mc.movie_id = t.id;