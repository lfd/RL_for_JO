SELECT * FROM title AS t, complete_cast AS cc WHERE t.production_year > 1990 AND t.id = cc.movie_id AND cc.movie_id = t.id;