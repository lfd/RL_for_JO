SELECT * FROM title AS t, movie_keyword AS mk, complete_cast AS cc WHERE t.production_year > 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id;