SELECT * FROM movie_keyword AS mk, title AS t, cast_info AS ci WHERE t.production_year > 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;