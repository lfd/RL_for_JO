SELECT * FROM title AS t, cast_info AS ci WHERE t.production_year > 2005 AND t.id = ci.movie_id AND ci.movie_id = t.id;