SELECT * FROM title AS t, cast_info AS ci WHERE ci.note = '(voice)' AND t.production_year BETWEEN 2007 AND 2010 AND ci.movie_id = t.id AND t.id = ci.movie_id;