SELECT * FROM title AS t, aka_title AS at WHERE t.production_year > 2000 AND t.id = at.movie_id AND at.movie_id = t.id;