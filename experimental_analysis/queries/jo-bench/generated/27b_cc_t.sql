SELECT * FROM complete_cast AS cc, title AS t WHERE t.production_year = 1998 AND t.id = cc.movie_id AND cc.movie_id = t.id;