SELECT * FROM movie_keyword AS mk, complete_cast AS cc, title AS t WHERE t.production_year > 1950 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id;